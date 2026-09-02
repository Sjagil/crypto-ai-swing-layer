from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _clip_probability(probability) -> np.ndarray:
    values = np.asarray(probability, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("probability contains non-finite values")
    return np.clip(values, 1e-6, 1.0 - 1e-6)


def _logit(probability) -> np.ndarray:
    p = _clip_probability(probability)
    return np.log(p / (1.0 - p))


@dataclass
class ProbabilityCalibrator:
    method: str
    estimator: Any | None = None

    def transform(self, probability) -> np.ndarray:
        p = _clip_probability(probability)
        if self.method == "raw":
            return p
        if self.method == "sigmoid":
            if self.estimator is None:
                raise ValueError("sigmoid calibrator is not fitted")
            return np.clip(
                self.estimator.predict_proba(_logit(p).reshape(-1, 1))[:, 1],
                1e-6,
                1.0 - 1e-6,
            )
        if self.method == "isotonic":
            if self.estimator is None:
                raise ValueError("isotonic calibrator is not fitted")
            return np.clip(
                np.asarray(self.estimator.predict(p), dtype=float),
                1e-6,
                1.0 - 1e-6,
            )
        raise ValueError(f"unsupported calibration method: {self.method}")


@dataclass
class CalibratedClassifier:
    # Joblib-safe wrapper around a fitted classifier and frozen calibrator.
    base_model: Any
    calibrator: ProbabilityCalibrator

    def predict_proba(self, x) -> np.ndarray:
        raw = self.base_model.predict_proba(x)[:, 1]
        calibrated = self.calibrator.transform(raw)
        return np.column_stack((1.0 - calibrated, calibrated))


def fit_probability_calibrator(
    raw_probability,
    y_true,
    *,
    method: str,
) -> ProbabilityCalibrator:
    p = _clip_probability(raw_probability)
    y = np.asarray(y_true, dtype=int).reshape(-1)
    if len(p) != len(y) or not len(y):
        raise ValueError("calibration arrays must be aligned and non-empty")
    if len(np.unique(y)) < 2:
        raise ValueError("calibration labels require both classes")

    selected = str(method).strip().lower()
    if selected == "raw":
        return ProbabilityCalibrator("raw", None)
    if selected == "sigmoid":
        estimator = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=17,
        )
        estimator.fit(_logit(p).reshape(-1, 1), y)
        return ProbabilityCalibrator("sigmoid", estimator)
    if selected == "isotonic":
        estimator = IsotonicRegression(
            y_min=1e-6,
            y_max=1.0 - 1e-6,
            out_of_bounds="clip",
        )
        estimator.fit(p, y)
        return ProbabilityCalibrator("isotonic", estimator)
    raise ValueError(f"unsupported calibration method: {method}")


def purged_calibration_selection_split(
    validation: pd.DataFrame,
    *,
    horizon_bars: int,
    calibration_fraction: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Chronologically separate calibrator fitting from model/threshold selection.
    if validation.empty:
        raise ValueError("validation frame is empty")
    fraction = float(calibration_fraction)
    if not 0.25 <= fraction <= 0.75:
        raise ValueError("calibration_fraction must be between 0.25 and 0.75")

    times = pd.DatetimeIndex(
        pd.to_datetime(validation["feature_time"], utc=True)
        .drop_duplicates()
        .sort_values()
    )
    if len(times) < 20:
        raise ValueError("insufficient validation timestamps for calibration split")

    cut = max(1, min(len(times) - 1, int(len(times) * fraction)))
    purge = max(1, int(horizon_bars))
    calibration_end = max(1, cut - purge)
    calibration_times = set(times[:calibration_end])
    selection_times = set(times[cut:])

    calibration = validation[
        pd.to_datetime(validation["feature_time"], utc=True).isin(calibration_times)
    ].copy()
    selection = validation[
        pd.to_datetime(validation["feature_time"], utc=True).isin(selection_times)
    ].copy()

    if len(calibration) < 100 or len(selection) < 100:
        raise ValueError(
            "calibration/selection split requires at least 100 rows per segment"
        )
    if calibration["target_alpha"].nunique() < 2:
        raise ValueError("calibration segment has one alpha class")
    if selection["target_alpha"].nunique() < 2:
        raise ValueError("selection segment has one alpha class")
    return calibration, selection
