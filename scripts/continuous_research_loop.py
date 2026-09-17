#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crypto_ai_swing.agents.live_promotion import LiveModelGovernor
from crypto_ai_swing.agents.manager import AgentManager
from crypto_ai_swing.agents.tcn_gru_live_governor import TCNGRULiveGovernor
from crypto_ai_swing.models.tcn_gru import TCNGRUChallengerTrainer, default_tcn_gru_config
from crypto_ai_swing.research.finance_diagnostics import diagnose_markets
from crypto_ai_swing.research.native import NativeResearchBridge
from crypto_ai_swing.settings import Settings

DEFAULT_MARKETS = (
    "ADA-EUR", "AVAX-EUR", "BCH-EUR", "BNB-EUR", "BTC-EUR",
    "DOGE-EUR", "ETH-EUR", "HBAR-EUR", "HYPE-EUR", "LINK-EUR",
    "LTC-EUR", "NEAR-EUR", "SHIB-EUR", "SOL-EUR", "SUI-EUR",
    "TAO-EUR", "TRX-EUR", "UNI-EUR", "XLM-EUR", "XRP-EUR",
)


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, dict) else {}
    except Exception:
        return {}


def age_seconds(path: Path) -> float | None:
    if not path.is_file():
        return None
    return max(0.0, time.time() - path.stat().st_mtime)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def run_cycle(settings, markets: list[str], args) -> dict[str, Any]:
    output: dict[str, Any] = {"started_at": now(), "markets": markets}

    manager = AgentManager(settings, mode="shadow")
    output["agent_manager"] = manager.cycle(
        markets=markets,
        forward_database_path=None,
        force_train=False,
    )

    output["supervised_rl_promotion"] = LiveModelGovernor(settings).cycle()

    tcn_root = Path(settings.project_root) / "output/crypto_ai_swing/agents/tcn_gru"
    tcn_pointer = tcn_root / "latest.pointer.json"
    tcn_age = age_seconds(tcn_pointer)
    tcn_due = tcn_age is None or tcn_age >= float(args.tcn_retrain_seconds)
    if tcn_due:
        diagnostics_bridge = NativeResearchBridge(settings.crypto_repo_root)
        # The diagnostic module uses the same canonical historical bridge contract.
        # Instantiate it through AgentManager's bridge to avoid a second data source.
        output["finance_diagnostics"] = diagnose_markets(
            manager.trainer.crypto,
            markets,
            timeframe=args.tcn_timeframe,
            maximum_rows=20000,
        )
        cfg = default_tcn_gru_config(horizon_bars=args.tcn_horizon_bars)
        trainer = TCNGRUChallengerTrainer(settings, config=cfg)
        result = trainer.train(
            markets=markets,
            timeframe=args.tcn_timeframe,
            horizon_bars=args.tcn_horizon_bars,
            minimum_rows=args.tcn_minimum_rows,
            minimum_net_move_bps=args.minimum_net_move_bps,
            expires_days=30,
        )
        output["tcn_gru_training"] = {
            "status": result.status,
            "artifact_path": str(result.artifact_path),
            "manifest_path": str(result.manifest_path),
            "dataset_id": result.dataset_id,
            "rows": result.row_count,
            "markets": list(result.markets),
            "metrics": result.metrics,
        }
    else:
        output["tcn_gru_training"] = {
            "status": "NOT_DUE",
            "age_seconds": tcn_age,
        }

    output["tcn_gru_promotion"] = TCNGRULiveGovernor(settings).cycle()

    research_state = (
        Path(settings.project_root)
        / "output/crypto_ai_swing/research/continuous_factory_state.json"
    )
    state = read_json(research_state)
    last = state.get("last_factory_at_epoch")
    factory_due = last is None or time.time() - float(last) >= float(args.factory_seconds)
    if factory_due:
        bridge = NativeResearchBridge(settings.crypto_repo_root)
        raw = bridge.run_factory_campaign(
            maximum_rows=args.factory_maximum_rows,
            execute_exact=args.factory_exact,
        )
        output["strategy_factory"] = bridge.factory_summary(raw)
        atomic_json(
            research_state,
            {"last_factory_at_epoch": time.time(), "last_factory_at": now()},
        )
    else:
        output["strategy_factory"] = {
            "status": "NOT_DUE",
            "seconds_remaining": max(0.0, float(args.factory_seconds) - (time.time() - float(last))),
        }

    output["completed_at"] = now()
    atomic_json(
        Path(settings.project_root)
        / "output/crypto_ai_swing/research/continuous_machine_latest.json",
        output,
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", default=",".join(DEFAULT_MARKETS))
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--tcn-timeframe", default="1h")
    parser.add_argument("--tcn-horizon-bars", type=int, default=4)
    parser.add_argument("--tcn-minimum-rows", type=int, default=8000)
    parser.add_argument("--tcn-retrain-seconds", type=int, default=21600)
    parser.add_argument("--minimum-net-move-bps", type=float, default=65.0)
    parser.add_argument("--factory-seconds", type=int, default=21600)
    parser.add_argument("--factory-maximum-rows", type=int, default=50000)
    parser.add_argument("--factory-exact", action="store_true", default=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    settings = Settings.load(root)
    markets = list(dict.fromkeys(
        item.strip().upper()
        for item in args.markets.split(",")
        if item.strip()
    ))

    while True:
        try:
            payload = run_cycle(settings, markets, args)
            print(json.dumps(payload, indent=2, default=str), flush=True)
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            error = {
                "at": now(),
                "status": "ERROR",
                "error": f"{type(exc).__name__}:{str(exc)[:1000]}",
            }
            print(json.dumps(error, indent=2), flush=True)
            atomic_json(
                Path(settings.project_root)
                / "output/crypto_ai_swing/research/continuous_machine_error.json",
                error,
            )
        if args.once:
            return 0
        time.sleep(max(60, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
