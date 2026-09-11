from __future__ import annotations

from types import SimpleNamespace

from crypto_ai_swing.research.economic_optimizer import (
    EconomicImprovementController,
)


def test_weak_paper_economics_triggers_bounded_work(tmp_path):
    settings = SimpleNamespace(
        project_root=tmp_path,
        autonomy={
            "economic_optimizer": {
                "minimum_closed_trades": 2,
                "force_train_cooldown_seconds": 60,
                "force_research_cooldown_seconds": 60,
            }
        },
    )
    report = EconomicImprovementController(settings).plan(
        {
            "status": "DEGRADED",
            "overall": {"closed_trades": 5},
            "research_priorities": [
                "IMPROVE_NET_EXPECTANCY"
            ],
        }
    )
    assert report["force_train"] is True
    assert report["force_research"] is True
    assert report["research_priorities"] == [
        "IMPROVE_NET_EXPECTANCY"
    ]
    assert report["automatic_live_authority"] is False
    assert report["orders_submitted"] == 0
