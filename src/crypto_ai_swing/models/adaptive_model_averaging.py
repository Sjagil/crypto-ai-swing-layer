from __future__ import annotations

"""Evidence-weighted probability ensemble for the live swing stack.

The implementation is deliberately conservative: it borrows the model-averaging
idea from Bayesian model averaging, but does not call the weights exact posterior
model probabilities unless a true model-evidence calculation is available.
Weights are derived only from out-of-sample diagnostics and are normalized with a
softmax.  Missing or unqualified challengers get zero weight.
"""

from dataclasses import dataclass
from math import exp, log
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ModelVote:
    name: str
    probability: float | None
    qualified: bool
    metrics: Mapping[str, Any] | None = None


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _tcn_gru_evidence_score(metrics: Mapping[str, Any] | None) -> float:
    """Map OOS diagnostics to a bounded evidence score.

    Positive contributions require discrimination, calibration and after-cost
    economics.  This avoids giving a deep model a large weight merely because it
    is complex.
    """
    m = dict(metrics or {})
    test = dict(m.get("test") or {})
    prob = dict(test.get("probability") or {})
    econ = dict(test.get("economics") or {})

    auc = _finite(prob.get("auc"), 0.5) or 0.5
    brier = _finite(prob.get("brier"), 0.25) or 0.25
    rows = max(0.0, _finite(prob.get("rows"), 0.0) or 0.0)
    conservative = _finite(econ.get("conservative_mean_net"), 0.0) or 0.0
    market_balanced = _finite(econ.get("market_balanced_mean_net"), 0.0) or 0.0
    positive_fraction = _finite(econ.get("positive_market_fraction"), 0.5) or 0.5
    market_count = max(0.0, _finite(econ.get("market_count"), 0.0) or 0.0)

    auc_skill = np.clip((auc - 0.5) / 0.10, -1.0, 1.0)
    brier_skill = np.clip((0.25 - brier) / 0.08, -1.0, 1.0)
    sample_strength = np.clip(log(1.0 + rows) / log(1.0 + 5000.0), 0.0, 1.0)
    market_strength = np.clip(market_count / 10.0, 0.0, 1.0)
    econ_skill = np.tanh(75.0 * conservative) + np.tanh(75.0 * market_balanced)
    breadth_skill = np.clip((positive_fraction - 0.5) / 0.25, -1.0, 1.0)

    return float(
        0.25 * auc_skill
        + 0.20 * brier_skill
        + 0.15 * sample_strength
        + 0.10 * market_strength
        + 0.20 * econ_skill
        + 0.10 * breadth_skill
    )


def _generic_evidence_score(metrics: Mapping[str, Any] | None) -> float:
    m = dict(metrics or {})
    auc = max(
        _finite(m.get("alpha_auc"), 0.5) or 0.5,
        _finite(m.get("test_alpha_auc"), 0.5) or 0.5,
    )
    conservative = _finite(m.get("selected_conservative_mean_net"), 0.0) or 0.0
    market_balanced = _finite(m.get("selected_market_balanced_mean_net"), 0.0) or 0.0
    positive_fraction = _finite(m.get("selected_positive_market_fraction"), 0.5) or 0.5
    return float(
        0.35 * np.clip((auc - 0.5) / 0.10, -1.0, 1.0)
        + 0.30 * np.tanh(75.0 * conservative)
        + 0.20 * np.tanh(75.0 * market_balanced)
        + 0.15 * np.clip((positive_fraction - 0.5) / 0.25, -1.0, 1.0)
    )


def evidence_score(vote: ModelVote) -> float:
    if not vote.qualified or vote.probability is None:
        return float("-inf")
    if vote.name.lower() in {"tcn_gru", "tcn+gru", "temporal"}:
        return _tcn_gru_evidence_score(vote.metrics)
    return _generic_evidence_score(vote.metrics)


def blend_probabilities(
    votes: list[ModelVote],
    *,
    temperature: float = 0.75,
    minimum_probability: float = 1e-6,
) -> dict[str, Any]:
    valid: list[tuple[ModelVote, float]] = []
    for vote in votes:
        p = _finite(vote.probability)
        if not vote.qualified or p is None:
            continue
        p = float(np.clip(p, minimum_probability, 1.0 - minimum_probability))
        valid.append((ModelVote(vote.name, p, True, vote.metrics), evidence_score(vote)))

    if not valid:
        return {
            "probability": None,
            "weights": {},
            "evidence_scores": {},
            "model_count": 0,
            "method": "evidence_weighted_model_average_v1",
        }

    if len(valid) == 1:
        vote, score = valid[0]
        return {
            "probability": float(vote.probability),
            "weights": {vote.name: 1.0},
            "evidence_scores": {vote.name: score},
            "model_count": 1,
            "method": "evidence_weighted_model_average_v1",
        }

    temp = max(0.05, float(temperature))
    finite_scores = np.asarray([score for _, score in valid], dtype=float)
    finite_scores -= np.max(finite_scores)
    raw = np.exp(finite_scores / temp)
    weights = raw / raw.sum()
    probabilities = np.asarray([float(vote.probability) for vote, _ in valid], dtype=float)
    blended = float(np.sum(weights * probabilities))

    return {
        "probability": blended,
        "weights": {
            vote.name: float(weight)
            for (vote, _), weight in zip(valid, weights, strict=True)
        },
        "evidence_scores": {
            vote.name: float(score)
            for vote, score in valid
        },
        "model_count": len(valid),
        "method": "evidence_weighted_model_average_v1",
    }
