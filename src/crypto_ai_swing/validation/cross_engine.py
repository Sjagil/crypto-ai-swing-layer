from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.bridge.reference_runtime import reference_environment_status


@dataclass(frozen=True)
class ReplaySpec:
    market: str
    timeframe: str
    fast_ema: int
    slow_ema: int
    initial_cash_eur: float
    trade_notional_eur: float
    maximum_rows: int


def _stable_json_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str:
    subset = frame[["open", "high", "low", "close", "volume"]].copy()
    values = {
        "index": [str(x) for x in subset.index],
        "rows": subset.round(12).replace({np.nan: None}).values.tolist(),
    }
    return _stable_json_hash(values)


def _signal_schedule(frame: pd.DataFrame, *, fast: int, slow: int) -> pd.DataFrame:
    if fast <= 0 or slow <= 0 or fast >= slow:
        raise ValueError("EMA periods must satisfy 0 < fast < slow")
    close = frame["close"].astype(float)
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    raw_long = (fast_ema > slow_ema).astype(bool)
    target = raw_long.shift(1, fill_value=False).astype(bool)
    previous = target.shift(1, fill_value=False).astype(bool)
    return pd.DataFrame(
        {
            "target_long": target,
            "entry": target & ~previous,
            "exit": ~target & previous,
        },
        index=frame.index,
    )


def _native_fixed_size_replay(
    frame: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    initial_cash: float,
    trade_notional: float,
    fee_fraction: float,
    execution_drag_fraction: float,
) -> dict[str, Any]:
    cash = float(initial_cash)
    quantity = 0.0
    entry_count = 0
    exit_count = 0
    curve: list[float] = []
    first_close = float(frame["close"].iloc[0])
    fixed_quantity = trade_notional / first_close if first_close > 0 else 0.0

    for idx, row in frame.iterrows():
        close = float(row["close"])
        if bool(schedule.loc[idx, "entry"]) and quantity <= 0 and fixed_quantity > 0:
            fill = close * (1.0 + execution_drag_fraction)
            gross = fixed_quantity * fill
            fee = gross * fee_fraction
            if gross + fee <= cash:
                cash -= gross + fee
                quantity = fixed_quantity
                entry_count += 1
        if bool(schedule.loc[idx, "exit"]) and quantity > 0:
            fill = close * (1.0 - execution_drag_fraction)
            gross = quantity * fill
            fee = gross * fee_fraction
            cash += gross - fee
            quantity = 0.0
            exit_count += 1
        curve.append(cash + quantity * close)

    final_equity = curve[-1] if curve else cash
    peak = -math.inf
    max_drawdown = 0.0
    for value in curve:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - value) / peak)

    return {
        "engine": "native_fixed_size",
        "status": "COMPLETED",
        "entries": entry_count,
        "exits": exit_count,
        "orders": entry_count + exit_count,
        "fixed_quantity": fixed_quantity,
        "final_equity": final_equity,
        "total_return": final_equity / initial_cash - 1.0,
        "max_drawdown": max_drawdown,
        "open_position": quantity > 0,
    }


def _run_worker(
    python_path: Path,
    worker: Path,
    input_path: Path,
    output_path: Path,
    *,
    timeout: int = 600,
) -> dict[str, Any]:
    if not python_path.is_file():
        return {"status": "BLOCKED_RUNTIME", "error": f"Python missing: {python_path}"}
    proc = subprocess.run(
        [str(python_path), str(worker), str(input_path), str(output_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        return {
            "status": "BLOCKED_WORKER",
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-3000:],
            "stderr_tail": proc.stderr[-3000:],
        }
    if not output_path.is_file():
        return {"status": "BLOCKED_WORKER", "error": "Worker did not write output"}
    return json.loads(output_path.read_text(encoding="utf-8"))


def _cost_model(bridge: CryptoLibraryBridge) -> dict[str, Any]:
    economics = bridge.import_module("core.economics")
    cls = getattr(economics, "CanonicalCostModel", None)
    if cls is None:
        raise RuntimeError("core.economics.CanonicalCostModel unavailable")
    model = cls.from_settings(bridge.settings())
    return {
        "cost_model_version": model.cost_model_version,
        "maker_fee_fraction": float(model.maker_fee_fraction),
        "taker_fee_fraction": float(model.taker_fee_fraction),
        "spread_bps": float(model.spread_bps),
        "slippage_bps": float(model.slippage_bps),
        "calibration_status": model.calibration_status,
    }


def _pick_reference(status: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((row for row in status.get("references", []) if row.get("name") == name), None)


def run_cross_engine_validation(
    project_root: Path,
    crypto_repo_root: Path,
    *,
    market: str | None = None,
    timeframe: str | None = None,
    fast_ema: int | None = None,
    slow_ema: int | None = None,
    maximum_rows: int | None = None,
    venv_root: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    cfg = yaml.safe_load((root / "config/cross_engine.yaml").read_text(encoding="utf-8")) or {}
    defaults = dict(cfg.get("default") or {})
    parity_cfg = dict(cfg.get("parity") or {})
    spec = ReplaySpec(
        market=str(market or defaults.get("market") or "BTC-EUR").upper(),
        timeframe=str(timeframe or defaults.get("timeframe") or "1h"),
        fast_ema=int(fast_ema or defaults.get("fast_ema") or 20),
        slow_ema=int(slow_ema or defaults.get("slow_ema") or 60),
        initial_cash_eur=float(defaults.get("initial_cash_eur") or 10000.0),
        trade_notional_eur=float(defaults.get("trade_notional_eur") or 1000.0),
        maximum_rows=int(maximum_rows or defaults.get("maximum_rows") or 5000),
    )

    bridge = CryptoLibraryBridge(Path(crypto_repo_root))
    frame = bridge.ohlcv(spec.market, spec.timeframe, persist=False).copy().sort_index().tail(spec.maximum_rows)
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Canonical frame missing columns: {sorted(missing)}")
    if len(frame) < max(spec.slow_ema * 3, 100):
        raise RuntimeError(f"Insufficient canonical rows: {len(frame)}")

    schedule = _signal_schedule(frame, fast=spec.fast_ema, slow=spec.slow_ema)
    frame_hash = _frame_hash(frame)
    signal_hash = _stable_json_hash(
        {
            "index": [str(x) for x in schedule.index],
            "target_long": schedule["target_long"].astype(int).tolist(),
            "entry": schedule["entry"].astype(int).tolist(),
            "exit": schedule["exit"].astype(int).tolist(),
        }
    )
    costs = _cost_model(bridge)
    fee_fraction = float(costs["taker_fee_fraction"])
    execution_drag_fraction = (
        float(costs["slippage_bps"]) + float(costs["spread_bps"]) / 2.0
    ) / 10000.0
    native = _native_fixed_size_replay(
        frame,
        schedule,
        initial_cash=spec.initial_cash_eur,
        trade_notional=spec.trade_notional_eur,
        fee_fraction=fee_fraction,
        execution_drag_fraction=execution_drag_fraction,
    )

    status = reference_environment_status(
        root,
        include_disabled=False,
        venv_roots=[venv_root] if venv_root else None,
        repo_roots=[repo_root] if repo_root else None,
    )
    run_dir = root / "output/crypto_ai_swing/cross_engine" / f"{spec.market.replace('-', '_')}_{spec.timeframe}"
    run_dir.mkdir(parents=True, exist_ok=True)
    data_path = run_dir / "canonical_input.parquet"
    input_path = run_dir / "worker_input.json"
    vectorbt_out = run_dir / "vectorbt.json"
    nautilus_out = run_dir / "nautilus.json"
    work = frame.copy()
    work["entry"] = schedule["entry"].astype(bool)
    work["exit"] = schedule["exit"].astype(bool)
    work.to_parquet(data_path)
    input_path.write_text(
        json.dumps(
            {
                "schema_version": "crypto_ai_swing_cross_engine_worker_input_v1",
                "spec": asdict(spec),
                "data_path": str(data_path),
                "frame_hash": frame_hash,
                "signal_hash": signal_hash,
                "costs": costs,
                "fee_fraction": fee_fraction,
                "execution_drag_fraction": execution_drag_fraction,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    vectorbt_row = _pick_reference(status, "vectorbt")
    nautilus_row = _pick_reference(status, "nautilus_trader")
    workers = root / "scripts/workers"
    vectorbt = (
        _run_worker(Path(str(vectorbt_row.get("python"))), workers / "vectorbt_cross_engine_worker.py", input_path, vectorbt_out)
        if vectorbt_row and vectorbt_row.get("runtime_ready")
        else {"engine": "vectorbt", "status": "BLOCKED_REFERENCE_RUNTIME", "reference_status": vectorbt_row}
    )
    nautilus = (
        _run_worker(Path(str(nautilus_row.get("python"))), workers / "nautilus_cross_engine_worker.py", input_path, nautilus_out)
        if nautilus_row and nautilus_row.get("runtime_ready")
        else {"engine": "nautilus", "status": "BLOCKED_REFERENCE_RUNTIME", "reference_status": nautilus_row}
    )

    tolerance_bps = float(parity_cfg.get("total_return_abs_tolerance_bps") or 12.0)
    order_tolerance = int(parity_cfg.get("order_count_tolerance") or 2)
    comparisons: dict[str, Any] = {}
    for engine_name, evidence in (("vectorbt", vectorbt), ("nautilus", nautilus)):
        if evidence.get("status") != "COMPLETED":
            comparisons[engine_name] = {"evaluated": False, "pass": False, "reason": evidence.get("status")}
            continue
        delta_bps = abs(float(evidence.get("total_return") or 0.0) - float(native["total_return"])) * 10000.0
        order_delta = abs(int(evidence.get("orders") or 0) - int(native.get("orders") or 0))
        input_ok = evidence.get("frame_hash") == frame_hash
        signal_ok = evidence.get("signal_hash") == signal_hash
        fill_schedule_ok = bool(evidence.get("fill_schedule_match", True))
        comparisons[engine_name] = {
            "evaluated": True,
            "pass": bool(
                delta_bps <= tolerance_bps
                and order_delta <= order_tolerance
                and input_ok
                and signal_ok
                and fill_schedule_ok
            ),
            "return_delta_bps": delta_bps,
            "return_tolerance_bps": tolerance_bps,
            "order_delta": order_delta,
            "order_tolerance": order_tolerance,
            "input_hash_match": input_ok,
            "signal_hash_match": signal_ok,
            "fill_schedule_match": fill_schedule_ok,
        }

    completed = [x for x in comparisons.values() if x.get("evaluated")]
    parity_pass = bool(comparisons) and all(
        bool(item.get("evaluated")) and bool(item.get("pass"))
        for item in comparisons.values()
    )
    payload = {
        "schema_version": "crypto_ai_swing_cross_engine_validation_v1",
        "spec": asdict(spec),
        "frame_rows": len(frame),
        "frame_start": str(frame.index.min()),
        "frame_end": str(frame.index.max()),
        "frame_hash": frame_hash,
        "signal_hash": signal_hash,
        "cost_model": costs,
        "native": native,
        "vectorbt": vectorbt,
        "nautilus": nautilus,
        "comparisons": comparisons,
        "required_engine_count": len(comparisons),
        "completed_engine_count": len(completed),
        "parity_pass": parity_pass,
        "classification": "CROSS_ENGINE_PARITY_PASS" if parity_pass else "CROSS_ENGINE_PARITY_INCOMPLETE_OR_FAILED",
        "authority": "RESEARCH_ONLY",
        "mechanics_validation_only": True,
        "alpha_promotion_authorized": False,
        "live_decision_influence": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    latest = run_dir / "latest.json"
    latest.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    payload["output"] = str(latest)
    return payload
