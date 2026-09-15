from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


LABEL_NOISE_VERSION = "round47g_label_noise_v1"


@dataclass(frozen=True)
class LabelNoisePolicy:
    enabled: bool = True
    boundary_band_fraction_of_cost: float = 0.25
    minimum_band_bps: float = 10.0
    maximum_band_bps: float = 30.0
    minimum_confidence: float = 0.05
    confidence_power: float = 1.35
    fallback_min_weight: float = 0.25


def resolve_label_noise_policy(
    config: dict[str, Any] | None = None,
) -> LabelNoisePolicy:
    raw = dict(config or {})
    policy = LabelNoisePolicy(
        enabled=bool(raw.get("enabled", True)),
        boundary_band_fraction_of_cost=float(
            raw.get("boundary_band_fraction_of_cost", 0.25)
        ),
        minimum_band_bps=float(raw.get("minimum_band_bps", 10.0)),
        maximum_band_bps=float(raw.get("maximum_band_bps", 30.0)),
        minimum_confidence=float(raw.get("minimum_confidence", 0.05)),
        confidence_power=float(raw.get("confidence_power", 1.35)),
        fallback_min_weight=float(raw.get("fallback_min_weight", 0.25)),
    )
    if policy.boundary_band_fraction_of_cost < 0.0:
        raise ValueError("boundary_band_fraction_of_cost must be non-negative")
    if policy.minimum_band_bps < 0.0:
        raise ValueError("minimum_band_bps must be non-negative")
    if policy.maximum_band_bps < policy.minimum_band_bps:
        raise ValueError("maximum_band_bps must be >= minimum_band_bps")
    if not 0.0 <= policy.minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be in [0, 1]")
    if policy.confidence_power <= 0.0:
        raise ValueError("confidence_power must be positive")
    if policy.fallback_min_weight < 0.0:
        raise ValueError("fallback_min_weight must be non-negative")
    return policy


def label_noise_band_bps(
    *,
    cost_floor: float,
    config: dict[str, Any] | None = None,
) -> float:
    policy = resolve_label_noise_policy(config)
    if not policy.enabled:
        return 0.0
    cost_bps = abs(float(cost_floor)) * 10_000.0
    raw = cost_bps * policy.boundary_band_fraction_of_cost
    return float(
        np.clip(raw, policy.minimum_band_bps, policy.maximum_band_bps)
    )


def _net_return(
    frame: pd.DataFrame,
    *,
    cost_floor: float,
) -> np.ndarray:
    if "target_net_return" in frame.columns:
        return frame["target_net_return"].to_numpy(float)
    return frame["target_forward_return"].to_numpy(float) - float(cost_floor)


def alpha_label_confidence(
    frame: pd.DataFrame,
    *,
    cost_floor: float,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    policy = resolve_label_noise_policy(config)
    n = len(frame)
    if n == 0:
        return np.asarray([], dtype=float)
    if not policy.enabled:
        return np.ones(n, dtype=float)

    band_bps = label_noise_band_bps(cost_floor=cost_floor, config=config)
    if band_bps <= 0.0:
        return np.ones(n, dtype=float)

    distance_bps = np.abs(
        _net_return(frame, cost_floor=cost_floor)
    ) * 10_000.0
    x = np.clip(distance_bps / band_bps, 0.0, 1.0)
    smooth = x * x * (3.0 - 2.0 * x)
    shaped = np.power(smooth, policy.confidence_power)
    confidence = (
        policy.minimum_confidence
        + (1.0 - policy.minimum_confidence) * shaped
    )
    return np.clip(
        confidence,
        policy.minimum_confidence,
        1.0,
    ).astype(float)


def economic_alpha_sample_weight(
    frame: pd.DataFrame,
    *,
    cost_floor: float,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    realized = frame["target_forward_return"].to_numpy(float)
    mae = frame["target_mae"].to_numpy(float)
    mfe = frame["target_mfe"].to_numpy(float)
    net = realized - float(cost_floor)

    scale = np.clip(
        np.abs(net) / max(float(cost_floor), 1e-4),
        0.0,
        3.0,
    )
    path_quality = np.clip(
        (mfe - mae) / (mfe + mae + 1e-9),
        0.0,
        1.0,
    )
    economic = np.clip(
        1.0
        + 0.35 * np.sqrt(scale)
        + 0.25 * path_quality * (net > 0.0),
        0.75,
        2.50,
    )

    confidence = alpha_label_confidence(
        frame,
        cost_floor=cost_floor,
        config=config,
    )
    combined = economic * confidence
    finite = np.isfinite(combined)
    if finite.any():
        mean_weight = float(np.mean(combined[finite]))
        if mean_weight > 1e-12:
            combined = combined / mean_weight
    return np.clip(combined, 0.05, 3.00).astype(float)


def alpha_label_noise_summary(
    frame: pd.DataFrame,
    *,
    cost_floor: float,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = resolve_label_noise_policy(config)
    confidence = alpha_label_confidence(
        frame,
        cost_floor=cost_floor,
        config=config,
    )
    weights = economic_alpha_sample_weight(
        frame,
        cost_floor=cost_floor,
        config=config,
    )
    net = _net_return(frame, cost_floor=cost_floor)
    band_bps = label_noise_band_bps(cost_floor=cost_floor, config=config)
    distance_bps = np.abs(net) * 10_000.0
    ambiguous = (
        distance_bps < band_bps
        if band_bps > 0.0
        else np.zeros(len(frame), dtype=bool)
    )
    labels = (
        frame["target_alpha"].to_numpy(float)
        if "target_alpha" in frame.columns
        else (net > 0.0).astype(float)
    )
    total_weight = float(np.sum(weights))
    weighted_positive_fraction = (
        float(np.sum(weights * labels) / total_weight)
        if total_weight > 1e-12
        else None
    )
    return {
        "version": LABEL_NOISE_VERSION,
        "policy": asdict(policy),
        "cost_floor_bps": float(cost_floor) * 10_000.0,
        "boundary_band_bps": float(band_bps),
        "rows": int(len(frame)),
        "ambiguous_rows": int(np.sum(ambiguous)),
        "ambiguous_fraction": float(np.mean(ambiguous)) if len(frame) else 0.0,
        "mean_confidence": float(np.mean(confidence)) if len(confidence) else 1.0,
        "median_confidence": float(np.median(confidence)) if len(confidence) else 1.0,
        "minimum_observed_confidence": float(np.min(confidence)) if len(confidence) else 1.0,
        "mean_training_weight": float(np.mean(weights)) if len(weights) else 1.0,
        "raw_positive_fraction": float(np.mean(labels)) if len(labels) else None,
        "weighted_positive_fraction": weighted_positive_fraction,
        "validation_labels_mutated": False,
        "test_labels_mutated": False,
        "return_target_mutated": False,
        "risk_target_mutated": False,
        "regime_target_mutated": False,
    }


def fit_weighted_or_confident_subset(
    model,
    x,
    y,
    weights,
    *,
    fallback_min_weight: float = 0.25,
):
    sample_weight = np.asarray(weights, dtype=float)

    if hasattr(model, "steps") and getattr(model, "steps", None):
        step_name = model.steps[-1][0]
        try:
            return model.fit(
                x,
                y,
                **{f"{step_name}__sample_weight": sample_weight},
            )
        except (TypeError, ValueError):
            pass

    try:
        return model.fit(x, y, sample_weight=sample_weight)
    except (TypeError, ValueError):
        pass

    keep = (
        np.isfinite(sample_weight)
        & (sample_weight >= float(fallback_min_weight))
    )
    minimum_rows = max(40, int(0.25 * len(sample_weight)))
    if int(np.sum(keep)) >= minimum_rows:
        positions = np.flatnonzero(keep)
        x_fit = x.iloc[positions] if hasattr(x, "iloc") else np.asarray(x)[keep]
        y_fit = y.iloc[positions] if hasattr(y, "iloc") else np.asarray(y)[keep]
        if len(np.unique(np.asarray(y_fit))) >= 2:
            return model.fit(x_fit, y_fit)

    return model.fit(x, y)


__all__ = [
    "LABEL_NOISE_VERSION",
    "LabelNoisePolicy",
    "alpha_label_confidence",
    "alpha_label_noise_summary",
    "economic_alpha_sample_weight",
    "fit_weighted_or_confident_subset",
    "label_noise_band_bps",
    "resolve_label_noise_policy",
]
