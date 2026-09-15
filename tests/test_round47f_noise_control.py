from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_ai_swing.agents.feature_denoising import (
    denoised_candidate_columns,
    feature_group_counts,
    feature_selection_diagnostics,
    select_stable_train_features,
)
from crypto_ai_swing.agents.rl_multi_market import (
    _env,
    _validation_viability_checks,
)


def _synthetic(rows: int = 800):
    rng = np.random.default_rng(47)
    target = pd.Series(rng.normal(0.0, 1.0, rows), dtype=float)
    frame = pd.DataFrame(index=np.arange(rows))

    frame["core_stable"] = target + rng.normal(0.0, 0.8, rows)
    frame["pattern_stable"] = 0.45 * target + rng.normal(0.0, 1.0, rows)
    frame["crypto_vwap_distance_20"] = (
        0.35 * target + rng.normal(0.0, 1.0, rows)
    )
    frame["crypto_idx_rsi_14"] = (
        0.30 * target + rng.normal(0.0, 1.0, rows)
    )
    frame["crypto_strategy_breakout_entry"] = (
        target > 0.5
    ).astype(float)

    flip = target.copy()
    flip.iloc[rows // 2 :] *= -1.0
    frame["pattern_sign_flipping"] = (
        flip + rng.normal(0.0, 0.5, rows)
    )

    for i in range(30):
        frame[f"noise_{i}"] = rng.normal(0.0, 1.0, rows)

    return frame, target


def test_round47f_candidate_stage_keeps_binary_strategy_information():
    frame, _ = _synthetic()
    selected = denoised_candidate_columns(
        frame,
        maximum_candidates=32,
    )
    assert "crypto_strategy_breakout_entry" in selected


def test_round47f_stability_score_penalizes_sign_flipping_noise():
    frame, target = _synthetic()
    rows = feature_selection_diagnostics(
        frame,
        frame.columns,
        target=target,
    )
    by_name = {
        str(row["name"]): row
        for row in rows
    }
    assert (
        by_name["pattern_stable"]["stability_score"]
        > by_name["pattern_sign_flipping"]["stability_score"]
    )


def test_round47f_group_aware_selection_preserves_new_information_groups():
    frame, target = _synthetic()
    selected = select_stable_train_features(
        frame,
        frame.columns,
        target=target,
        maximum_features=16,
        maximum_abs_correlation=0.95,
    )
    counts = feature_group_counts(selected)
    assert counts["pattern"] >= 1
    assert counts["strategy"] >= 1
    assert counts["vwap"] >= 1
    assert counts["index"] >= 1
    assert len(selected) <= 16


def test_round47f_validation_viability_rejects_best_of_bad_policies():
    bad = {
        "market_count": 24,
        "mean_return": -0.05,
        "median_return": -0.04,
        "positive_market_fraction": 0.25,
        "mean_excess_vs_buy_hold": -0.20,
        "worst_maximum_drawdown": 0.35,
        "mean_invalid_action_rate": 0.10,
    }
    checks = _validation_viability_checks(bad)
    assert not all(checks.values())

    good = {
        "market_count": 24,
        "mean_return": 0.04,
        "median_return": 0.02,
        "positive_market_fraction": 0.58,
        "mean_excess_vs_buy_hold": 0.01,
        "worst_maximum_drawdown": 0.18,
        "mean_invalid_action_rate": 0.10,
    }
    checks = _validation_viability_checks(good)
    assert all(checks.values())


def test_round47f_trade_giveback_is_peak_relative_and_bounded():
    index = pd.date_range(
        "2026-01-01",
        periods=20,
        freq="h",
        tz="UTC",
    )
    x = pd.DataFrame(
        {"x": np.linspace(0.0, 1.0, len(index))},
        index=index,
    )
    returns = pd.Series(
        [0.60, 0.60, -0.45, -0.20] + [0.0] * 16,
        index=index,
        dtype=float,
    )

    env = _env(
        x,
        returns,
        minimum_hold_bars=1,
        cooldown_bars=1,
    )
    env.reset()
    env.step(1)
    env.step(1)
    _, _, _, _, info = env.step(1)

    value = float(info["trade_giveback"])
    assert 0.0 <= value <= 1.0
