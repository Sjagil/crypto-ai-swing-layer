from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

SOFT_MACRO_BLOCKERS = frozenset(
    {
        "MTF_MACRO_LONG_PERMISSION_DENIED",
        "MTF_MACRO_AGGREGATE_NEGATIVE",
    }
)


@dataclass(frozen=True)
class RallyAssessment:
    enabled: bool
    score: float
    deep_scan_candidate: bool
    overextended: bool
    ret_8: float
    ret_24: float
    trend_component: float
    momentum_component: float
    execution_quality: float
    rsi_14: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_rally(
    screen_row: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> RallyAssessment:
    """Classify post-breakout acceleration without requiring breakout_20 > 0."""
    cfg = dict(config or {})
    enabled = bool(cfg.get("enabled", True))
    ret_8 = float(screen_row.get("ret_8", 0.0) or 0.0)
    ret_24 = float(screen_row.get("ret_24", 0.0) or 0.0)
    trend = float(screen_row.get("trend_component", 0.0) or 0.0)
    momentum = float(screen_row.get("momentum_component", 0.0) or 0.0)
    execution_quality = float(screen_row.get("execution_quality", 0.0) or 0.0)
    rsi = float(screen_row.get("rsi_14", 50.0) or 50.0)

    impulse = math.tanh(max(0.0, ret_8) * 30.0)
    persistence = math.tanh(max(0.0, ret_24) * 8.0)
    score = (
        0.30 * max(0.0, min(1.0, trend))
        + 0.30 * max(0.0, min(1.0, momentum))
        + 0.20 * impulse
        + 0.12 * persistence
        + 0.08 * max(0.0, min(1.0, execution_quality))
    )

    minimum_ret_8 = float(cfg.get("minimum_ret_8", 0.015))
    minimum_ret_24 = float(cfg.get("minimum_ret_24", 0.04))
    minimum_trend = float(cfg.get("minimum_trend_component", 0.45))
    minimum_momentum = float(cfg.get("minimum_momentum_component", 0.65))
    minimum_score = float(cfg.get("minimum_rally_score", 0.65))
    max_rsi = float(cfg.get("maximum_rsi_for_macro_override", 84.0))

    return RallyAssessment(
        enabled=enabled,
        score=float(max(0.0, min(1.0, score))),
        deep_scan_candidate=bool(
            enabled
            and (ret_8 >= minimum_ret_8 or ret_24 >= minimum_ret_24)
            and trend >= minimum_trend
            and momentum >= minimum_momentum
            and score >= minimum_score
        ),
        overextended=bool(rsi >= max_rsi),
        ret_8=ret_8,
        ret_24=ret_24,
        trend_component=trend,
        momentum_component=momentum,
        execution_quality=execution_quality,
        rsi_14=rsi,
    )


def macro_override_allowed(
    decision: Any,
    assessment: Mapping[str, Any] | RallyAssessment,
    *,
    mode: str,
    config: Mapping[str, Any] | None = None,
) -> bool:
    """Bounded SHADOW/PAPER-only exception for lagging soft macro blockers."""
    cfg = dict(config or {})
    if str(mode).lower() not in {"shadow", "paper"}:
        return False
    if not bool(cfg.get("paper_macro_override_enabled", True)):
        return False

    row = assessment.to_dict() if isinstance(assessment, RallyAssessment) else dict(assessment or {})
    if not bool(row.get("deep_scan_candidate")) or bool(row.get("overextended")):
        return False

    blockers = {str(x) for x in getattr(decision, "blockers", ())}
    if not blockers or not blockers.issubset(SOFT_MACRO_BLOCKERS):
        return False

    return bool(
        float(getattr(decision, "macro_score", -1.0))
        >= float(cfg.get("macro_override_minimum_macro_score", -0.20))
        and float(getattr(decision, "trend_score", -1.0))
        >= float(cfg.get("macro_override_minimum_trend_score", 0.30))
        and float(getattr(decision, "setup_score", -1.0))
        >= float(cfg.get("macro_override_minimum_setup_score", 0.35))
        and float(getattr(decision, "trigger_score", -1.0))
        >= float(cfg.get("macro_override_minimum_trigger_score", 0.15))
        and float(getattr(decision, "execution_score", -1.0))
        >= float(cfg.get("macro_override_minimum_execution_score", -0.10))
    )
