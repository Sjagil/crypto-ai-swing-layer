from __future__ import annotations

import numpy as np

from crypto_ai_swing.agents.component_features import extract_component
from crypto_ai_swing.research.entry_selector import (
    SELECTOR_FEATURES,
    selector_feature_vector,
)
from crypto_ai_swing.research.swing_geometry import feature_vector


def _context():
    return {
        "agents": {
            "alpha_probability": 0.74,
            "diagnostics": {
                "head_qualifications": {
                    "alpha": False,
                    "regime": False,
                    "return": False,
                    "risk": False,
                    "execution": False,
                }
            },
        },
        "technical": {
            "technical_score": 0.80,
        },
        "mtf_challenger": {
            "score": 0.625,
            "cmc_regime_score": 0.70,
        },
        "timeframe_pipeline": {
            "states": {
                "1h": {"score": 0.60},
                "4h": {"score": 0.65},
            }
        },
    }


def test_round47c2_unqualified_alpha_remains_blocked_for_live_extractor():
    assert extract_component(_context(), "alpha") is None


def test_round47c2_research_geometry_observes_shadow_alpha():
    row = feature_vector(_context())
    assert np.isclose(row["alpha"], 0.74)


def test_round47c2_selector_registers_alpha_confluence_features():
    assert "alpha" in SELECTOR_FEATURES
    assert "interaction_alpha_technical" in SELECTOR_FEATURES
    assert "interaction_alpha_mtf" in SELECTOR_FEATURES
    assert "interaction_alpha_cmc" in SELECTOR_FEATURES


def test_round47c2_selector_builds_alpha_interactions_when_available():
    row = selector_feature_vector(_context())
    assert row["alpha"] is not None
    assert row["interaction_alpha_technical"] is not None
