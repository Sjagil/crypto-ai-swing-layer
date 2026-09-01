from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class AgentCouncilDecision:
    alpha_probability: float | None
    forecast_score: float | None
    regime_score: float | None
    predicted_return: float | None
    predicted_mae: float | None
    execution_score: float
    entry_blocked: bool
    blocker_codes: tuple[str, ...]
    live_influence: bool
    diagnostics: dict[str, Any]


def execution_context_score(context: dict[str, Any]) -> float:
    spread = max(0.0, float(context.get("spread_bps", 999.0) or 999.0))
    book = float(context.get("book_imbalance", 0.0) or 0.0)
    cvd = float(
        context.get("cvd_notional_ratio", context.get("cvd_ratio", 0.0)) or 0.0
    )
    micro = float(context.get("microprice_edge_bps", 0.0) or 0.0)
    spread_quality = float(np.exp(-spread / 15.0))
    return float(np.clip(
        0.50 * spread_quality
        + 0.22 * ((np.clip(book, -1, 1) + 1) / 2)
        + 0.18 * ((np.clip(cvd, -1, 1) + 1) / 2)
        + 0.10 * ((np.tanh(micro / 5.0) + 1) / 2),
        0,
        1,
    ))


def build_council_decision(
    predictions: dict[str, float | None],
    context: dict[str, Any],
    *,
    artifact_status: str,
    live_decision_influence: bool,
    mode: str,
    shadow_decision_qualified: bool = True,
    head_qualifications: dict[str, bool] | None = None,
) -> AgentCouncilDecision:
    alpha = predictions.get("alpha_probability")
    regime = predictions.get("regime_probability")
    ret = predictions.get("predicted_return")
    mae = predictions.get("predicted_mae")
    forecast = (
        float(np.clip(0.5 + 0.5 * np.tanh(float(ret) / 0.02), 0, 1))
        if ret is not None and np.isfinite(float(ret))
        else None
    )
    execution = execution_context_score(context)

    status = str(artifact_status or "RESEARCH_ONLY").upper()
    live_influence = bool(
        mode == "live"
        and status in {"CANARY", "ACTIVE"}
        and live_decision_influence
    )
    shadow_influence = bool(mode != "live" and shadow_decision_qualified)
    decision_influence = bool(live_influence or shadow_influence)

    if head_qualifications is None:
        # Backward compatibility for v1/v2 artifacts and existing unit tests.
        head_qualifications = {
            "alpha": bool(shadow_decision_qualified),
            "regime": bool(shadow_decision_qualified),
            "return": bool(shadow_decision_qualified),
            "risk": bool(shadow_decision_qualified),
            "execution": True,
        }
    head_qualifications = {
        "alpha": bool(head_qualifications.get("alpha", False)),
        "regime": bool(head_qualifications.get("regime", False)),
        "return": bool(head_qualifications.get("return", False)),
        "risk": bool(head_qualifications.get("risk", False)),
        "execution": bool(head_qualifications.get("execution", True)),
    }
    head_influence = {
        name: bool(decision_influence and qualified)
        for name, qualified in head_qualifications.items()
    }

    diagnostic_blockers: list[str] = []
    active_blockers: list[str] = []

    def add_blocker(code: str, *, active: bool) -> None:
        diagnostic_blockers.append(code)
        if active:
            active_blockers.append(code)

    if alpha is not None and float(alpha) < 0.35:
        add_blocker("AGENT_ALPHA_STRONG_NEGATIVE", active=head_influence["alpha"])
    if regime is not None and float(regime) < 0.30:
        add_blocker("AGENT_REGIME_UNFAVORABLE", active=head_influence["regime"])
    if mae is not None and float(mae) > 0.08:
        add_blocker("AGENT_PREDICTED_MAE_EXCESSIVE", active=head_influence["risk"])
    if execution < 0.20:
        add_blocker("AGENT_EXECUTION_CONTEXT_POOR", active=head_influence["execution"])

    return AgentCouncilDecision(
        alpha_probability=float(alpha) if alpha is not None else None,
        forecast_score=forecast,
        regime_score=float(regime) if regime is not None else None,
        predicted_return=float(ret) if ret is not None else None,
        predicted_mae=float(mae) if mae is not None else None,
        execution_score=execution,
        entry_blocked=bool(active_blockers),
        blocker_codes=tuple(diagnostic_blockers),
        live_influence=live_influence,
        diagnostics={
            "artifact_status": status,
            "mode": mode,
            "shadow_decision_qualified": bool(shadow_decision_qualified),
            "decision_influence": decision_influence,
            "head_qualifications": head_qualifications,
            "head_influence": head_influence,
            "active_blocker_codes": active_blockers,
            "advisory_blocker_codes": [
                code for code in diagnostic_blockers if code not in active_blockers
            ],
            "advisory_only": not any(head_influence.values()),
            "automatic_live_promotion": False,
        },
    )
