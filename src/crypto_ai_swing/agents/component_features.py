from __future__ import annotations

from typing import Any, Mapping

import numpy as np


COMPONENTS = (
    "technical",
    "mtf",
    "alpha",
    "forecast",
    "regime",
    "risk",
    "execution",
    "rl",
    "cmc",
    "nlp",
    "orderflow",
)

DEFAULT_PRIORS = {
    "technical": 0.18,
    "mtf": 0.22,
    "alpha": 0.09,
    "forecast": 0.08,
    "regime": 0.07,
    "risk": 0.08,
    "execution": 0.09,
    "rl": 0.03,
    "cmc": 0.06,
    "nlp": 0.05,
    "orderflow": 0.05,
}


def clip01(value: Any, default: float | None = 0.5) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return float(np.clip(result, 0.0, 1.0))


def sigmoid(value: Any, default: float | None = 0.5) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(x):
        return default
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0))))


def normalize_signed(value: Any, default: float | None = None) -> float | None:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(raw):
        return default
    return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))


def _head_qualification(context: Mapping[str, Any], head: str) -> bool | None:
    agents = dict(context.get("agents") or {})
    diagnostics = dict(agents.get("diagnostics") or {})
    qualifications = dict(diagnostics.get("head_qualifications") or {})
    if head not in qualifications:
        return None
    return bool(qualifications.get(head))


def _qualified_or_unknown(context: Mapping[str, Any], head: str) -> bool:
    value = _head_qualification(context, head)
    return value is not False


def extract_component(context: Mapping[str, Any], name: str) -> float | None:
    """Extract one research component without granting it authority.

    Known-unqualified supervised heads are excluded instead of being silently
    treated as useful alpha. Historical observations without qualification
    metadata remain available for research and are reported as qualification
    unknown rather than rewritten.
    """
    if name == "technical":
        screen = dict(context.get("universe_screen") or {})
        value = screen.get("execution_adjusted_score")
        if value is not None:
            return sigmoid(value, None)
        technical = dict(context.get("technical") or {})
        value = technical.get("technical_score")
        if value is not None:
            return normalize_signed(value, None)
        old = dict(context.get("edge_manager") or {}).get("components") or {}
        return clip01(old.get("technical"), None)

    if name == "mtf":
        challenger = dict(context.get("mtf_challenger") or {})
        if challenger.get("score") is not None:
            return clip01(challenger.get("score"), None)
        value = context.get("mtf_score")
        if value is not None:
            return normalize_signed(value, None)
        return None

    if name in {"alpha", "forecast", "regime", "execution"}:
        head = {
            "alpha": "alpha",
            "forecast": "return",
            "regime": "regime",
            "execution": "execution",
        }[name]
        if not _qualified_or_unknown(context, head):
            return None
        agents = dict(context.get("agents") or {})
        field = {
            "alpha": "alpha_probability",
            "forecast": "forecast_score",
            "regime": "regime_score",
            "execution": "execution_score",
        }[name]
        return clip01(agents.get(field), None)

    if name == "risk":
        if not _qualified_or_unknown(context, "risk"):
            return None
        agents = dict(context.get("agents") or {})
        try:
            mae = abs(float(agents.get("predicted_mae")))
        except (TypeError, ValueError):
            return None
        if not np.isfinite(mae):
            return None
        # Smaller predicted adverse excursion is better. 3% is a bounded
        # research scale, not a trading threshold.
        return float(np.clip(np.exp(-mae / 0.03), 0.0, 1.0))

    if name == "rl":
        rl = dict(context.get("rl") or context.get("rl_preview") or {})
        if not bool(rl.get("qualified", False)):
            return None
        value = rl.get("long_probability")
        if value is None:
            score = rl.get("score")
            return normalize_signed(score, None) if score is not None else None
        return clip01(value, None)

    if name == "cmc":
        challenger = dict(context.get("mtf_challenger") or {})
        value = challenger.get("cmc_regime_score")
        return clip01(value, None) if value is not None else None

    if name == "nlp":
        value = context.get("nlp_score")
        if value is None:
            return None
        confidence = clip01(context.get("nlp_confidence"), 0.5) or 0.5
        signed = normalize_signed(value, None)
        if signed is None:
            return None
        return float(np.clip(0.5 + (signed - 0.5) * confidence, 0.0, 1.0))

    if name == "orderflow":
        value = context.get("orderflow_score")
        if value is None:
            return None
        return normalize_signed(value, None)

    return None


def extract_components(context: Mapping[str, Any]) -> dict[str, float | None]:
    return {name: extract_component(context, name) for name in COMPONENTS}


def extract_descriptors(context: Mapping[str, Any]) -> dict[str, Any]:
    screen = dict(context.get("universe_screen") or {})
    technical = dict(screen.get("technical") or context.get("technical") or {})
    challenger = dict(context.get("mtf_challenger") or {})
    diagnostics = dict(challenger.get("diagnostics") or {})
    rally = dict(screen.get("rally_capture") or context.get("rally_capture") or {})
    edge = dict(context.get("edge_manager") or {})
    return {
        "strategy_family": challenger.get("strategy_family"),
        "breakout_state": technical.get("breakout_state"),
        "overextended": bool(rally.get("overextended", False)),
        "mtf_alignment": clip01(challenger.get("alignment_score"), None),
        "mtf_execution": clip01(challenger.get("execution_score"), None),
        "mtf_macro": clip01(diagnostics.get("macro"), None),
        "mtf_trend": clip01(diagnostics.get("trend"), None),
        "mtf_setup": clip01(diagnostics.get("setup"), None),
        "mtf_trigger": clip01(diagnostics.get("trigger"), None),
        "h1_rsi": diagnostics.get("h1_rsi"),
        "m15_rsi": diagnostics.get("m15_rsi"),
        "spread_bps": diagnostics.get("spread_bps") or screen.get("universe_spread_bps"),
        "edge_manager_score": edge.get("research_score"),
    }
