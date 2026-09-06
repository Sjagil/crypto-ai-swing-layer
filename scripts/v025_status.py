#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import crypto_ai_swing
from crypto_ai_swing.settings import Settings


def read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def main() -> int:
    settings = Settings.load()
    root = Path(settings.project_root) / "output/crypto_ai_swing"
    calibration = read(root / "research/net_edge/latest.json")
    attribution = read(root / "research/attribution/latest.json")
    strategy = read(root / "research/strategy_lab/latest.json")
    edge = read(root / "agents/meta/latest.json")
    autonomy = read(root / "autonomy/latest.json")
    registry = read(root / "promotion/registry.json")

    segments = list(calibration.get("segments") or [])[:8]
    result = {
        "version": crypto_ai_swing.__version__,
        "autonomy": {
            "state": autonomy.get("state"),
            "errors": autonomy.get("errors"),
        },
        "attribution": {
            "status": attribution.get("status"),
            "observations": attribution.get("observations"),
            "mean_normal_net_bps": (
                (attribution.get("overall") or {}).get("normal_net") or {}
            ).get("mean_bps"),
        },
        "net_edge_calibration": {
            "status": calibration.get("status"),
            "qualified": calibration.get("qualified"),
            "observations": calibration.get("observations"),
            "overall": calibration.get("overall"),
            "exit_efficiency": calibration.get("exit_efficiency"),
            "oos_validation": calibration.get("oos_validation"),
            "top_segments": [
                {
                    "segment": row.get("segment"),
                    "observations": row.get("observations"),
                    "shrunk_normal_mean_bps": row.get("shrunk_normal_mean_bps"),
                    "shrunk_stressed_mean_bps": row.get("shrunk_stressed_mean_bps"),
                    "probability_mean_positive": (
                        row.get("posterior") or {}
                    ).get("probability_mean_positive"),
                }
                for row in segments
            ],
        },
        "strategy_lab": {
            "status": strategy.get("status"),
            "observations": strategy.get("observations"),
            "known_trial_count": strategy.get("known_trial_count"),
            "frozen_candidate": strategy.get("frozen_candidate"),
            "champion": strategy.get("champion"),
        },
        "agent_manager": {
            "status": edge.get("status"),
            "qualified": edge.get("qualified"),
            "observations": edge.get("observations"),
            "threshold": edge.get("threshold"),
            "net_edge_calibration": edge.get("net_edge_calibration"),
            "reason_codes": edge.get("reason_codes"),
        },
        "promotion": {
            "status": registry.get("status"),
            "champion": registry.get("champion"),
            "qualified_research_candidates": registry.get(
                "qualified_research_candidates"
            ),
            "net_edge_calibration": registry.get("net_edge_calibration"),
        },
        "live_decision_influence": False,
        "automatic_live_promotion": False,
    }
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
