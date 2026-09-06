from __future__ import annotations

import json

from crypto_ai_swing.agents.edge_manager import ResearchEdgeManager
from crypto_ai_swing.research.net_edge_calibration import NetEdgeCalibrator
from crypto_ai_swing.research.performance_attribution import PerformanceAttributionEngine
from crypto_ai_swing.research.promotion import ResearchPromotionRegistry
from crypto_ai_swing.research.strategy_challenger import StrategyChallengerLab
from crypto_ai_swing.research.swing_geometry import SwingGeometryEngine
from crypto_ai_swing.settings import Settings


def main() -> int:
    settings = Settings.load()
    mode = "paper"
    attribution = PerformanceAttributionEngine(settings, mode=mode).refresh()
    calibration = NetEdgeCalibrator(settings, mode=mode).refresh(force=True)
    geometry = SwingGeometryEngine(settings, mode=mode).refresh(force=True)
    strategy = StrategyChallengerLab(settings, mode=mode).refresh()
    edge = ResearchEdgeManager(settings, mode=mode)
    try:
        forward_path = PerformanceAttributionEngine(
            settings, mode=mode
        ).forward_database_path()
        edge_policy = edge.refresh_policy(forward_path, force=True)
    finally:
        pass
    registry = ResearchPromotionRegistry(settings).refresh()
    payload = {
        "schema_version": "v026_multi_horizon_swing_research_once_v1",
        "attribution_4h": attribution,
        "net_edge_24h": calibration,
        "swing_geometry": geometry,
        "strategy_lab": strategy,
        "edge_policy": edge_policy,
        "registry": registry,
        "live_decision_influence": False,
        "automatic_live_promotion": False,
        "orders_submitted": 0,
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
