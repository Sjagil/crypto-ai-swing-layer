from __future__ import annotations

import numpy as np

from crypto_ai_swing.agents.hpo import _economic_alpha_score


def test_round47a2_economic_alpha_rewards_profitable_selection():
    y = np.asarray([1, 1, 1, 0, 0, 0] * 20)
    probability = np.asarray(
        [0.92, 0.86, 0.81, 0.30, 0.24, 0.18] * 20
    )
    markets = np.asarray(
        ["A", "B", "C", "D", "E", "F"] * 20,
        dtype=object,
    )
    good = np.asarray(
        [0.020, 0.017, 0.014, -0.010, -0.008, -0.006] * 20
    )
    bad = -good

    good_score = _economic_alpha_score(
        y,
        probability,
        good,
        markets,
        cost_floor=0.0065,
        minimum_selected=20,
        minimum_markets=3,
    )
    bad_score = _economic_alpha_score(
        y,
        probability,
        bad,
        markets,
        cost_floor=0.0065,
        minimum_selected=20,
        minimum_markets=3,
    )

    assert good_score > bad_score
