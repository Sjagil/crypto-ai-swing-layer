#!/usr/bin/env python3
from __future__ import annotations

import json

from crypto_ai_swing.agents.edge_manager import ResearchEdgeManager
from crypto_ai_swing.research.net_edge_calibration import NetEdgeCalibrator
from crypto_ai_swing.research.performance_attribution import PerformanceAttributionEngine
from crypto_ai_swing.research.promotion import ResearchPromotionRegistry
from crypto_ai_swing.research.strategy_challenger import StrategyChallengerLab
from crypto_ai_swing.settings import Settings


def main() -> int:
    settings = Settings.load()
    attribution = PerformanceAttributionEngine(settings, mode="paper")
    calibration = NetEdgeCalibrator(settings, mode="paper")
    strategy = StrategyChallengerLab(settings, mode="paper")
    edge = ResearchEdgeManager(settings, mode="paper")
    edge.calibrator = calibration

    result = {
        "schema_version": "v025_cost_edge_research_once_v1",
        "attribution": attribution.refresh(),
        "net_edge_calibration": calibration.refresh(force=True),
        "strategy_lab": strategy.refresh(),
    }
    edge.strategy_lab = strategy
    result["agent_manager"] = edge.refresh_policy(
        attribution.forward_database_path(),
        force=True,
    )
    result["promotion"] = ResearchPromotionRegistry(settings).refresh()
    result["live_decision_influence"] = False
    result["automatic_live_promotion"] = False
    result["orders_submitted"] = 0
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
