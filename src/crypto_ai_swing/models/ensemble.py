from __future__ import annotations

from crypto_ai_swing.contracts import ModelVote


def ensemble_score(votes: list[ModelVote], weights: dict[str, float]) -> tuple[float, float]:
    if not votes:
        return 0.0, 0.0
    numerator = 0.0
    conf = 0.0
    denom = 0.0
    for vote in votes:
        w = float(weights.get(vote.source, weights.get("default", 0.0)))
        if w <= 0:
            continue
        numerator += vote.score * w
        conf += vote.confidence * w
        denom += w
    if denom <= 0:
        return 0.0, 0.0
    return max(0.0, min(1.0, numerator / denom)), max(0.0, min(1.0, conf / denom))
