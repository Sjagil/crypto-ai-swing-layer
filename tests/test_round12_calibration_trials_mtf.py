from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crypto_ai_swing.agents.calibration import (
    CalibratedClassifier,
    fit_probability_calibrator,
    purged_calibration_selection_split,
)
from crypto_ai_swing.agents.trials import register_trial_family
from crypto_ai_swing.orchestration.timeframe_pipeline import _macro_blockers


class _FakeModel:
    def predict_proba(self, x):
        p = np.asarray(x["p"], dtype=float)
        return np.column_stack((1.0 - p, p))


def test_sigmoid_calibrator_is_classifier_compatible():
    raw = np.linspace(0.10, 0.90, 200)
    y = (np.linspace(0.0, 1.0, 200) > 0.60).astype(int)
    calibrator = fit_probability_calibrator(raw, y, method="sigmoid")
    wrapped = CalibratedClassifier(_FakeModel(), calibrator)
    x = pd.DataFrame({"p": [0.2, 0.5, 0.8]})
    result = wrapped.predict_proba(x)
    assert result.shape == (3, 2)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
    assert np.allclose(result.sum(axis=1), 1.0)


def test_calibration_selection_split_is_chronological_and_purged():
    times = pd.date_range("2025-01-01", periods=120, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "feature_time": np.repeat(times, 2),
            "target_alpha": np.tile([0, 1], len(times)),
        }
    )
    calibration, selection = purged_calibration_selection_split(
        frame,
        horizon_bars=4,
        calibration_fraction=0.5,
    )
    assert calibration["feature_time"].max() < selection["feature_time"].min()
    gap = selection["feature_time"].min() - calibration["feature_time"].max()
    assert gap.total_seconds() >= 4 * 3600


def _experiment(dataset_id: str):
    return {
        "schema_version": "agent_alpha_tournament_v4",
        "dataset_id": dataset_id,
        "feature_columns": ["a", "b"],
        "timeframe": "1h",
        "horizon_bars": 4,
        "minimum_net_move_bps": 65.0,
        "models": [
            "extra_trees",
            "hist_gradient_boosting",
            "logistic_balanced",
        ],
        "calibration_methods": ["raw", "sigmoid", "isotonic"],
        "thresholds": [
            round(float(value), 3)
            for value in np.arange(0.50, 0.751, 0.025)
        ],
    }


def test_trial_ledger_dedupes_new_dataset_same_hypotheses(tmp_path):
    root = tmp_path / "agents"
    first = register_trial_family(
        root,
        experiment=_experiment("dataset-a"),
        trial_count=99,
        historical_floor=33,
    )
    second = register_trial_family(
        root,
        experiment=_experiment("dataset-b"),
        trial_count=99,
        historical_floor=33,
    )

    assert first["global_known_trial_count"] == 99
    assert second["global_known_trial_count"] == 99
    assert second["evaluation_count"] == 2
    assert second["unique_hypothesis_count"] == 99
    assert second["newly_registered_hypotheses"] == 0


def test_trial_ledger_adds_materially_new_hypothesis(tmp_path):
    root = tmp_path / "agents"
    register_trial_family(
        root,
        experiment=_experiment("dataset-a"),
        trial_count=99,
        historical_floor=33,
    )
    changed = _experiment("dataset-b")
    changed["thresholds"] = changed["thresholds"] + [0.775]
    result = register_trial_family(
        root,
        experiment=changed,
        trial_count=108,
        historical_floor=33,
    )
    assert result["global_known_trial_count"] == 108
    assert result["unique_hypothesis_count"] == 108


def test_weekly_bearish_daily_weak_bullish_is_macro_conflict():
    blockers = _macro_blockers(
        {
            "1w": {"score": -0.70},
            "1d": {"score": 0.25},
        },
        {
            "weekly_strong_bearish_score": -0.50,
            "daily_bullish_confirmation_score": 0.45,
        },
    )
    assert "MTF_MACRO_CONFLICT_WEEKLY_BEARISH" in blockers


def test_mtf_source_contains_aggregate_permission_gates():
    source = (
        Path(__file__).resolve().parents[1]
        / "src/crypto_ai_swing/orchestration/timeframe_pipeline.py"
    ).read_text()
    assert "MTF_MACRO_AGGREGATE_NEGATIVE" in source
    assert "MTF_TREND_AGGREGATE_NEGATIVE" in source
    assert "minimum_macro_score_for_long" in source
    assert "minimum_trend_score_for_long" in source
