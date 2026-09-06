from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class MTFChallengerDecision:
    strategy_family: str
    score: float
    alignment_score: float
    execution_score: float
    cmc_regime_score: float
    family_scores: dict[str, float]
    diagnostics: dict[str, Any]
    authority: str = "RESEARCH_ONLY"
    live_decision_influence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _norm(value: Any, default: float = 0.5) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(x):
        return default
    return _clip01((x + 1.0) / 2.0)


def _state(states: Mapping[str, Mapping[str, Any]], tf: str) -> Mapping[str, Any]:
    return states.get(tf) or states.get("1W" if tf == "1w" else tf) or {}


def evaluate_mtf_challenger(
    timeframe_decision: Any,
    *,
    microstructure: Mapping[str, Any] | None = None,
    cmc_asset: Mapping[str, Any] | None = None,
    cmc_context: Mapping[str, Any] | None = None,
) -> MTFChallengerDecision:
    """Research-only strategy-family challenger over the existing MTF pipeline.

    It never grants authority. It creates a richer candidate score that can be
    evaluated prospectively before any influence is enabled.
    """
    states = dict(getattr(timeframe_decision, "states", {}) or {})
    micro = dict(microstructure or {})
    asset = dict(cmc_asset or {})
    context = dict(cmc_context or {})

    d1 = _state(states, "1d")
    w1 = _state(states, "1w")
    h4 = _state(states, "4h")
    h2 = _state(states, "2h")
    h1 = _state(states, "1h")
    m15 = _state(states, "15m")

    macro = _clip01(
        0.70 * _norm(d1.get("score"))
        + 0.30 * _norm(w1.get("score"))
    )
    trend = _clip01(
        0.62 * _norm(h4.get("score"))
        + 0.38 * _norm(h2.get("score"))
    )
    setup = _norm(h1.get("score"))
    trigger = _norm(m15.get("score"))

    h1_breakout = _norm(h1.get("breakout"))
    m15_breakout = _norm(m15.get("breakout"))
    h1_momentum = _norm(h1.get("momentum"))
    m15_momentum = _norm(m15.get("momentum"))
    h4_momentum = _norm(h4.get("momentum"))

    try:
        h1_rsi = float(h1.get("rsi", 50.0))
    except (TypeError, ValueError):
        h1_rsi = 50.0
    try:
        m15_rsi = float(m15.get("rsi", 50.0))
    except (TypeError, ValueError):
        m15_rsi = 50.0

    pullback_zone = _clip01(
        1.0
        - abs(h1_rsi - 52.0) / 22.0
    )
    trigger_recovery = _clip01(
        0.55 * trigger
        + 0.45 * _clip01((m15_rsi - 35.0) / 25.0)
    )

    spread = float(micro.get("spread_bps", 999.0) or 999.0)
    book = float(np.clip(float(micro.get("book_imbalance", 0.0) or 0.0), -1.0, 1.0))
    cvd = float(np.clip(
        float(micro.get("cvd_notional_ratio", micro.get("cvd_ratio", 0.0)) or 0.0),
        -1.0,
        1.0,
    ))
    spread_quality = float(np.exp(-max(0.0, spread) / 15.0))
    execution = _clip01(
        0.55 * spread_quality
        + 0.25 * ((book + 1.0) / 2.0)
        + 0.20 * ((cvd + 1.0) / 2.0)
    )

    breadth = dict(context.get("breadth_top250") or {})
    try:
        breadth_24h = float(breadth.get("positive_fraction_24h", 0.5) or 0.5)
    except (TypeError, ValueError):
        breadth_24h = 0.5
    fear = context.get("fear_and_greed")
    fear_value = None
    if isinstance(fear, Mapping):
        try:
            fear_value = float(fear.get("value"))
        except (TypeError, ValueError):
            fear_value = None

    try:
        asset_24h = float(asset.get("percent_change_24h", 0.0) or 0.0)
    except (TypeError, ValueError):
        asset_24h = 0.0
    try:
        asset_7d = float(asset.get("percent_change_7d", 0.0) or 0.0)
    except (TypeError, ValueError):
        asset_7d = 0.0

    asset_momentum = _clip01(
        0.5
        + 0.25 * np.tanh(asset_24h / 10.0)
        + 0.25 * np.tanh(asset_7d / 25.0)
    )
    fear_regime = 0.5
    if fear_value is not None:
        # Extreme greed is not treated as unconditional bullish alpha.
        fear_regime = _clip01(
            0.65
            - 0.25 * max(0.0, (fear_value - 80.0) / 20.0)
            + 0.10 * max(0.0, (50.0 - fear_value) / 50.0)
        )
    cmc_regime = _clip01(
        0.45 * breadth_24h
        + 0.35 * asset_momentum
        + 0.20 * fear_regime
    )

    trend_continuation = _clip01(
        0.24 * macro
        + 0.28 * trend
        + 0.20 * setup
        + 0.12 * trigger
        + 0.10 * h4_momentum
        + 0.06 * execution
    )
    breakout = _clip01(
        0.18 * macro
        + 0.22 * trend
        + 0.21 * h1_breakout
        + 0.16 * m15_breakout
        + 0.13 * h1_momentum
        + 0.10 * execution
    )
    pullback = _clip01(
        0.23 * macro
        + 0.27 * trend
        + 0.18 * pullback_zone
        + 0.14 * trigger_recovery
        + 0.10 * execution
        + 0.08 * cmc_regime
    )
    acceleration = _clip01(
        0.16 * macro
        + 0.19 * trend
        + 0.20 * h1_momentum
        + 0.18 * m15_momentum
        + 0.12 * trigger
        + 0.08 * execution
        + 0.07 * cmc_regime
    )

    family_scores = {
        "TREND_CONTINUATION": trend_continuation,
        "BREAKOUT_CONFIRMATION": breakout,
        "TREND_PULLBACK": pullback,
        "RALLY_ACCELERATION": acceleration,
    }
    family = max(family_scores, key=family_scores.get)
    base = family_scores[family]

    alignment = _clip01(
        0.30 * macro
        + 0.28 * trend
        + 0.22 * setup
        + 0.12 * trigger
        + 0.08 * execution
    )
    score = _clip01(
        0.72 * base
        + 0.16 * alignment
        + 0.07 * execution
        + 0.05 * cmc_regime
    )

    return MTFChallengerDecision(
        strategy_family=family,
        score=score,
        alignment_score=alignment,
        execution_score=execution,
        cmc_regime_score=cmc_regime,
        family_scores=family_scores,
        diagnostics={
            "macro": macro,
            "trend": trend,
            "setup": setup,
            "trigger": trigger,
            "h1_rsi": h1_rsi,
            "m15_rsi": m15_rsi,
            "spread_bps": spread,
            "breadth_24h": breadth_24h,
            "fear_greed": fear_value,
            "asset_change_24h": asset_24h,
            "asset_change_7d": asset_7d,
            "base_pipeline_entry_blocked": bool(
                getattr(timeframe_decision, "entry_blocked", False)
            ),
        },
    )
