from __future__ import annotations

import numpy as np

from crypto_ai_swing.agents.component_features import (
    extract_component,
    extract_research_component,
)
from crypto_ai_swing.research.performance_attribution import _safe_spearman


def _shadow_context():
    return {
        "agents": {
            "alpha_probability": 0.72,
            "forecast_score": 0.64,
            "regime_score": 0.61,
            "execution_score": 0.57,
            "predicted_mae": 0.012,
            "diagnostics": {
                "head_qualifications": {
                    "alpha": False,
                    "return": False,
                    "regime": False,
                    "risk": False,
                    "execution": False,
                }
            },
        },
        "rl": {
            "qualified": False,
            "long_probability": 0.68,
            "score": 0.36,
        },
    }


def test_live_extractor_still_blocks_unqualified_heads():
    context = _shadow_context()
    for name in ("alpha", "forecast", "regime", "risk", "execution", "rl"):
        assert extract_component(context, name) is None


def test_research_extractor_observes_shadow_predictions():
    context = _shadow_context()
    assert np.isclose(extract_research_component(context, "alpha"), 0.72)
    assert np.isclose(extract_research_component(context, "forecast"), 0.64)
    assert np.isclose(extract_research_component(context, "regime"), 0.61)
    assert extract_research_component(context, "risk") is not None
    assert np.isclose(extract_research_component(context, "execution"), 0.57)
    assert np.isclose(extract_research_component(context, "rl"), 0.68)


def test_safe_spearman_handles_constant_vector():
    corr, defined = _safe_spearman(
        np.ones(20),
        np.arange(20, dtype=float),
    )
    assert corr == 0.0
    assert defined is False


def test_safe_spearman_keeps_monotonic_signal():
    corr, defined = _safe_spearman(
        np.arange(20, dtype=float),
        np.arange(20, dtype=float),
    )
    assert defined is True
    assert corr > 0.99
