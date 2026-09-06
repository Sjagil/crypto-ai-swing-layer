from __future__ import annotations

from types import MethodType

from crypto_ai_swing.research.net_edge_calibration import (
    NetEdgeCalibrator,
    exit_efficiency,
    segment_keys_from_context,
)
from crypto_ai_swing.research.strategy_challenger import candidate_passes


def context(*, state="DONCHIAN_55_CONFIRMED", mtf=0.8, spread=4.0):
    return {
        "mtf_challenger": {
            "score": mtf,
            "strategy_family": "BREAKOUT_CONFIRMATION",
            "diagnostics": {"spread_bps": spread},
        },
        "universe_screen": {
            "technical": {"breakout_state": state},
            "universe_spread_bps": spread,
        },
    }


def row(net: float, *, gross: float | None = None, mfe=120.0, mae=-40.0):
    gross_value = net + 60.0 if gross is None else gross
    return {
        "observed_at": "2026-09-01T00:00:00+00:00",
        "market": "TEST-EUR",
        "context": context(),
        "gross_return_bps": gross_value,
        "normal_net_bps": net,
        "stressed_net_bps": net - 25.0,
        "normal_cost_bps": 60.0,
        "stressed_cost_bps": 85.0,
        "mfe_bps": mfe,
        "mae_bps": mae,
    }


def calibrator_for_test() -> NetEdgeCalibrator:
    obj = NetEdgeCalibrator.__new__(NetEdgeCalibrator)
    obj.minimum_history = 4
    obj.minimum_segment = 2
    obj.prior_strength = 0.0
    obj.minimum_expected_net_bps = 10.0
    obj.minimum_stressed_net_bps = 0.0
    obj.minimum_qualification_selected = 30
    obj.minimum_qualification_total = 100
    obj.minimum_positive_fraction = 0.55
    obj.minimum_probability_positive = 0.95
    obj.bootstrap_draws = 1000
    obj._stochastic = MethodType(
        lambda self, normal, stressed: {
            "status": "COLLECTING",
            "passed": False,
        },
        obj,
    )
    return obj


def test_cost_segment_keys_are_deterministic():
    keys = segment_keys_from_context(context())
    assert keys[0] == "STATE_MTF::DONCHIAN_55_CONFIRMED::HIGH"
    assert "STATE::DONCHIAN_55_CONFIRMED" in keys
    assert "MTF::HIGH" in keys
    assert "SPREAD::LE_5" in keys
    assert keys[-1] == "GLOBAL"


def test_exit_efficiency_separates_path_opportunity_from_realized_return():
    rows = [
        row(-20.0, gross=40.0, mfe=150.0),
        row(30.0, gross=90.0, mfe=110.0),
    ]
    result = exit_efficiency(rows)
    assert result["status"] == "READY"
    assert result["mfe_covers_normal_cost_fraction"] == 1.0
    assert result["negative_realized_despite_mfe_cost_cover_fraction"] == 0.5
    assert result["mean_mfe_to_realized_giveback_bps"] > 0.0


def test_sequential_gate_does_not_use_current_rows_future_return():
    obj = calibrator_for_test()
    rows = [row(-50.0) for _ in range(4)] + [row(500.0)]
    result = obj._sequential_validation(rows)
    # The +500 bps row cannot select itself: the four prior matured rows are
    # negative and therefore its prior-only gate must remain closed.
    assert result["oos_selected"] == 0
    assert result["lookahead_policy"] == (
        "EACH_OOS_ROW_GATED_USING_ONLY_PRIOR_MATURED_ROWS"
    )


def test_sequential_gate_can_select_after_prior_positive_evidence():
    obj = calibrator_for_test()
    rows = [row(50.0) for _ in range(4)] + [row(40.0)]
    result = obj._sequential_validation(rows)
    assert result["oos_selected"] == 1
    assert result["normal_net"]["mean_bps"] == 40.0
    assert result["qualified"] is False


def test_new_challenger_all_of_requires_every_rule():
    candidate = {
        "kind": "all_of",
        "rules": [
            {"kind": "breakout_state", "value": "DONCHIAN_55_CONFIRMED"},
            {"kind": "component_ge", "component": "mtf", "threshold": 0.70},
            {"kind": "descriptor_le", "field": "spread_bps", "threshold": 10.0},
        ],
    }
    assert candidate_passes(candidate, context())
    assert not candidate_passes(candidate, context(spread=15.0))
