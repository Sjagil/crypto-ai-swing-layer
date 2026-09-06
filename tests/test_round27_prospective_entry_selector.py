from __future__ import annotations

import numpy as np

from crypto_ai_swing.research.entry_selector import (
    SELECTOR_FEATURES,
    ProspectiveSwingEntrySelector,
    fit_factor_expectancy,
    score_factor_expectancy,
)


def test_factor_expectancy_uses_orthogonal_factor_ridge():
    rng = np.random.default_rng(27)
    x = rng.normal(size=(120, 8))
    signal = 1.6 * x[:, 0] - 1.2 * x[:, 1] + 0.4 * x[:, 2]
    normal = 120.0 * signal + rng.normal(scale=35.0, size=120)
    stressed = normal - 25.0
    mfe = np.maximum(20.0, 210.0 + 80.0 * signal + rng.normal(scale=30.0, size=120))
    mae = np.maximum(20.0, 120.0 - 25.0 * signal + rng.normal(scale=20.0, size=120))
    labels = (normal > 0).astype(int)
    targets = np.column_stack([normal, stressed, mfe, mae])
    names = [f"x{i}" for i in range(x.shape[1])]

    model = fit_factor_expectancy(
        x,
        labels,
        targets,
        feature_names=names,
        minimum_feature_coverage=0.8,
        maximum_factors=6,
        bootstrap_draws=60,
        seed=27,
    )

    assert model["status"] == "READY"
    assert 1 <= model["factor_count"] <= 6
    assert model["linear_algebra_contract"]["solver"] == (
        "LINEAR_SOLVE_NO_EXPLICIT_MATRIX_INVERSE"
    )

    high = {name: 0.0 for name in names}
    high["x0"] = 2.0
    high["x1"] = -1.0
    low = {name: 0.0 for name in names}
    low["x0"] = -2.0
    low["x1"] = 1.0
    assert (
        score_factor_expectancy(model, high)["expected_normal_net_bps"]
        > score_factor_expectancy(model, low)["expected_normal_net_bps"]
    )


def test_selector_abstains_when_policy_is_not_qualified():
    selector = ProspectiveSwingEntrySelector.__new__(
        ProspectiveSwingEntrySelector
    )
    selector._latest = {
        "status": "COLLECTING",
        "qualified": False,
    }
    selector.horizons = (24, 72, 168)
    result = selector.evaluate_context({})
    assert result["action"] == "ABSTAIN"
    assert result["passes"] is False
    assert result["live_decision_influence"] is False


def test_decision_rejects_ood_and_weak_expectancy():
    selector = ProspectiveSwingEntrySelector.__new__(
        ProspectiveSwingEntrySelector
    )
    selector.minimum_probability = 0.62
    selector.minimum_expected_normal = 20.0
    selector.minimum_expected_stressed = 0.0
    selector.minimum_mfe_cost_multiple = 2.0
    selector.minimum_excursion_ratio = 1.2
    selector.minimum_direction_stability = 0.55

    # Deliberately use a collecting model: fail closed.
    result = selector._decision(
        {"status": "COLLECTING"},
        {},
        normal_cost_bps=60.0,
    )
    assert result["action"] == "ABSTAIN"
    assert result["passes"] is False


def test_selector_feature_contract_is_nontrivial():
    assert len(SELECTOR_FEATURES) > 20
    assert "interaction_mtf_technical" in SELECTOR_FEATURES
    assert "state_donchian55" in SELECTOR_FEATURES
