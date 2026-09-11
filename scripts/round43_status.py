#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crypto_ai_swing.monitoring.research_drift import ResearchDriftMonitor
from crypto_ai_swing.research.evidence_maturation import (
    ProspectiveEvidenceMaturation,
)
from crypto_ai_swing.research.optimization_controller import (
    OptimizationController,
)
from crypto_ai_swing.settings import Settings


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, TypeError, ValueError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    settings = Settings.load(Path.cwd())
    evidence = ProspectiveEvidenceMaturation(settings)
    optimizer = OptimizationController(settings)
    drift = ResearchDriftMonitor(settings)
    root = (
        Path(settings.project_root)
        / "output/crypto_ai_swing/autonomy/round43"
    )
    latest = _read(root / "latest.json")
    heartbeat = _read(root / "heartbeat.json")
    payload = {
        "schema_version": "crypto_ai_swing_round43_status_v1",
        "runtime": latest,
        "heartbeat": heartbeat,
        "evidence": evidence.snapshot(persist=False),
        "optimization": optimizer.status(),
        "drift": drift.status(),
        "automatic_live_authority": False,
        "automatic_live_promotion": False,
    }
    if args.compact:
        payload = {
            "schema_version": payload["schema_version"],
            "runtime_state": latest.get("state", "NOT_BUILT"),
            "mode": latest.get("mode"),
            "maturation_stage": latest.get("maturation_stage"),
            "evidence_stage": payload["evidence"].get("research_stage"),
            "primary_outcomes": payload["evidence"].get(
                "primary_horizon_outcomes"
            ),
            "complete_required_horizons": payload["evidence"].get(
                "complete_required_horizon_observations"
            ),
            "canary_readiness": payload["evidence"].get(
                "canary_readiness"
            ),
            "paper_economic_status": (
                latest.get("paper_economics", {}).get("status")
            ),
            "paper_closed_trades": (
                latest.get("paper_economics", {})
                .get("overall", {})
                .get("closed_trades")
            ),
            "paper_cost_adjusted_pnl_eur": (
                latest.get("paper_economics", {})
                .get("overall", {})
                .get("cost_adjusted_pnl_eur")
            ),
            "paper_expectancy_bps": (
                latest.get("paper_economics", {})
                .get("overall", {})
                .get("expectancy_bps")
            ),
            "paper_profit_factor": (
                latest.get("paper_economics", {})
                .get("overall", {})
                .get("profit_factor")
            ),
            "economic_qualification": (
                latest.get("paper_economics", {})
                .get("economic_qualification")
            ),
            "drift_status": payload["drift"].get("status"),
            "heartbeat_at": heartbeat.get("heartbeat_at"),
            "automatic_live_authority": False,
            "automatic_live_promotion": False,
        }
    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
