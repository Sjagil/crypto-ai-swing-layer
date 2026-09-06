from __future__ import annotations

from crypto_ai_swing.agents.component_features import extract_component, extract_components
from crypto_ai_swing.research.strategy_challenger import candidate_passes


def test_unqualified_supervised_heads_are_removed_from_meta_manager():
    context = {
        "agents": {
            "alpha_probability": 0.99,
            "forecast_score": 0.99,
            "regime_score": 0.99,
            "execution_score": 0.8,
            "predicted_mae": 0.01,
            "diagnostics": {
                "head_qualifications": {
                    "alpha": False,
                    "return": False,
                    "regime": False,
                    "risk": True,
                    "execution": True,
                }
            },
        }
    }
    assert extract_component(context, "alpha") is None
    assert extract_component(context, "forecast") is None
    assert extract_component(context, "regime") is None
    assert extract_component(context, "risk") is not None
    assert extract_component(context, "execution") == 0.8


def test_unqualified_rl_is_zero_availability_not_fake_half_score():
    context = {"rl": {"qualified": False, "long_probability": 0.99}}
    assert extract_component(context, "rl") is None


def test_challenger_catalog_predicates_are_deterministic():
    row = {
        "components": {"mtf": 0.72, "execution": 0.61, "cmc": 0.69},
        "descriptors": {
            "strategy_family": "TREND_CONTINUATION",
            "breakout_state": "TREND_CONTINUATION",
            "overextended": False,
            "mtf_alignment": 0.80,
            "mtf_setup": 0.75,
            "mtf_trigger": 0.71,
            "h1_rsi": 70.0,
            "m15_rsi": 72.0,
        },
    }
    assert candidate_passes({"kind": "component_ge", "component": "mtf", "threshold": 0.70}, row)
    assert candidate_passes({"kind": "components_all_ge", "thresholds": {"mtf": 0.68, "execution": 0.55}}, row)
    assert candidate_passes({"kind": "not_overextended"}, row)
    assert candidate_passes({"kind": "family", "value": "TREND_CONTINUATION"}, row)
    assert not candidate_passes({"kind": "descriptor_le", "field": "h1_rsi", "threshold": 65.0}, row)


def test_context_component_extraction_has_no_live_authority_side_effects():
    context = {
        "mtf_challenger": {"score": 0.8, "cmc_regime_score": 0.7},
        "nlp_score": 0.2,
        "nlp_confidence": 0.5,
        "orderflow_score": 0.3,
    }
    components = extract_components(context)
    assert components["mtf"] == 0.8
    assert components["cmc"] == 0.7
    assert components["nlp"] is not None
    assert components["orderflow"] is not None
