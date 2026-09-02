from __future__ import annotations

import numpy as np

from crypto_ai_swing.agents.training import _threshold_plan


def test_threshold_economic_eligibility_requires_positive_conservative_bound():
    probability = np.full(40, 0.70, dtype=float)
    realized = np.array([0.02] * 39 + [-0.55], dtype=float)
    markets = np.array(["A", "B", "C", "D", "E"] * 8, dtype=object)

    plan = _threshold_plan(
        probability,
        realized,
        markets,
        cost_floor=0.0,
        minimum_selected=40,
        minimum_markets=5,
        minimum_positive_market_fraction=0.50,
    )

    row = next(item for item in plan["grid"] if item["threshold"] == 0.70)
    assert row["mean_net"] > 0.0
    assert row["conservative_mean_net"] < 0.0
    assert row["economically_eligible"] is False


def test_threshold_plan_exposes_market_breadth_requirement():
    probability = np.full(40, 0.70, dtype=float)
    realized = np.full(40, 0.02, dtype=float)
    markets = np.array(
        ["A"] * 20 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5 + ["E"] * 5,
        dtype=object,
    )

    plan = _threshold_plan(
        probability,
        realized,
        markets,
        cost_floor=0.005,
        minimum_selected=40,
        minimum_markets=5,
        minimum_positive_market_fraction=0.50,
    )

    row = next(item for item in plan["grid"] if item["threshold"] == 0.70)
    assert row["positive_market_fraction"] == 1.0
    assert row["economically_eligible"] is True
    assert plan["minimum_positive_market_fraction"] == 0.50
