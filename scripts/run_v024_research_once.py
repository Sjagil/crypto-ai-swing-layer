#!/usr/bin/env python3
from __future__ import annotations

import json

from crypto_ai_swing.agents.edge_manager import ResearchEdgeManager
from crypto_ai_swing.research.performance_attribution import PerformanceAttributionEngine
from crypto_ai_swing.research.promotion import ResearchPromotionRegistry
from crypto_ai_swing.research.strategy_challenger import StrategyChallengerLab
from crypto_ai_swing.settings import Settings


def main() -> int:
    settings = Settings.load()
    forward_cfg = dict(settings.autonomy.get("forward_evidence", {}) or {})
    database = settings.project_root / forward_cfg.get(
        "path", "output/crypto_ai_swing/forward/forward.sqlite"
    )
    attribution = PerformanceAttributionEngine(settings, mode="paper").refresh()
    strategy = StrategyChallengerLab(settings, mode="paper").refresh()
    edge = ResearchEdgeManager(settings, mode="paper").refresh_policy(database, force=True)
    registry = ResearchPromotionRegistry(settings).refresh()
    payload = {
        "schema_version": "v024_research_control_once_v1",
        "attribution": {
            "status": attribution.get("status"),
            "observations": attribution.get("observations"),
            "normal_net": (attribution.get("overall") or {}).get("normal_net"),
            "paper": attribution.get("paper"),
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
            "weights": edge.get("weights"),
            "threshold": edge.get("threshold"),
            "reason_codes": edge.get("reason_codes"),
        },
        "promotion": {
            "status": registry.get("status"),
            "champion": registry.get("champion"),
            "qualified_research_candidates": registry.get("qualified_research_candidates"),
        },
        "live_decision_influence": False,
        "automatic_live_promotion": False,
        "orders_submitted": 0,
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
