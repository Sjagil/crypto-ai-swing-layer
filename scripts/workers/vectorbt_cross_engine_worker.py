from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import plotly
import vectorbt as vbt


def _stable_json_hash(payload) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str:
    subset = frame[["open", "high", "low", "close", "volume"]].copy()
    values = {
        "index": [str(x) for x in subset.index],
        "rows": subset.round(12).where(subset.notna(), None).values.tolist(),
    }
    return _stable_json_hash(values)


def _signal_hash(frame: pd.DataFrame) -> str:
    entries = frame["entry"].astype(bool).tolist()
    exits = frame["exit"].astype(bool).tolist()
    target = []
    state = False
    for entry, exit_ in zip(entries, exits, strict=True):
        if entry:
            state = True
        if exit_:
            state = False
        target.append(int(state))
    return _stable_json_hash(
        {
            "index": [str(x) for x in frame.index],
            "target_long": target,
            "entry": [int(x) for x in entries],
            "exit": [int(x) for x in exits],
        }
    )


def main() -> int:
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    frame = pd.read_parquet(payload["data_path"]).sort_index()

    close = frame["close"].astype(float)
    entries = frame["entry"].astype(bool)
    exits = frame["exit"].astype(bool)
    spec = payload["spec"]
    fixed_quantity = float(spec["trade_notional_eur"]) / float(close.iloc[0])

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=float(spec["initial_cash_eur"]),
        size=fixed_quantity,
        fees=float(payload["fee_fraction"]),
        slippage=float(payload["execution_drag_fraction"]),
        freq=str(spec["timeframe"]),
    )

    records = pf.orders.records_arr
    observed_timestamps = sorted(
        int(pd.Timestamp(frame.index[int(row["idx"])]).value)
        for row in records
    )
    expected_timestamps = sorted(
        [int(pd.Timestamp(ts).value) for ts in frame.index[entries]]
        + [int(pd.Timestamp(ts).value) for ts in frame.index[exits]]
    )

    result = {
        "schema_version": "crypto_ai_swing_cross_engine_worker_v2",
        "engine": "vectorbt",
        "engine_version": getattr(vbt, "__version__", None),
        "plotly_version": getattr(plotly, "__version__", None),
        "status": "COMPLETED",
        "frame_hash": _frame_hash(frame),
        "signal_hash": _signal_hash(frame),
        "entries": int(entries.sum()),
        "exits": int(exits.sum()),
        "orders": len(records),
        "fill_schedule_match": observed_timestamps == expected_timestamps,
        "expected_fill_timestamps": expected_timestamps,
        "observed_fill_timestamps": observed_timestamps,
        "total_return": float(pf.total_return()),
        "max_drawdown": abs(float(pf.max_drawdown())),
        "final_value": float(pf.final_value()),
        "orders_submitted": 0,
        "authority": "RESEARCH_ONLY",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
