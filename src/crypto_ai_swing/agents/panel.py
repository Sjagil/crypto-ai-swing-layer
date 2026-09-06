from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
from crypto_ai_swing.agents.council import execution_context_score

@dataclass(frozen=True)
class AgentVote:
    name: str
    score: float | None
    veto: bool
    reasons: tuple[str, ...]
    authority: str = "ADVISORY_ONLY"
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _bounded(value):
    if value is None:
        return None
    value = float(value)
    return float(np.clip(value, 0, 1)) if np.isfinite(value) else None

def build_research_agent_panel(*, predictions: dict[str, float | None], context: dict[str, Any], bayesian_evidence: dict[str, Any] | None = None, stochastic_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    alpha = _bounded(predictions.get("alpha_probability"))
    regime = _bounded(predictions.get("regime_probability"))
    ret = predictions.get("predicted_return")
    mae = predictions.get("predicted_mae")
    ret_score = float(np.clip(0.5 + 0.5 * np.tanh(float(ret) / 0.02), 0, 1)) if ret is not None and np.isfinite(float(ret)) else None
    risk = float(np.clip(1 - float(mae) / 0.08, 0, 1)) if mae is not None and np.isfinite(float(mae)) else None
    execution = execution_context_score(context)
    bayes = _bounded((bayesian_evidence or {}).get("mean_posterior", {}).get("probability_mean_positive"))
    stochastic = (stochastic_evidence or {}).get("passed")
    votes = [
        AgentVote("alpha", alpha, False, ()),
        AgentVote("regime", regime, bool(regime is not None and regime < 0.30), ("UNFAVORABLE_REGIME",) if regime is not None and regime < 0.30 else ()),
        AgentVote("return", ret_score, False, ()),
        AgentVote("risk", risk, bool(risk is not None and risk < 0.15), ("PREDICTED_MAE_EXCESSIVE",) if risk is not None and risk < 0.15 else ()),
        AgentVote("execution", execution, execution < 0.20, ("EXECUTION_CONTEXT_POOR",) if execution < 0.20 else ()),
        AgentVote("bayesian_edge", bayes, bool(bayes is not None and bayes < 0.35), ("BAYESIAN_EDGE_NEGATIVE",) if bayes is not None and bayes < 0.35 else ()),
        AgentVote("stochastic_validation", 1.0 if stochastic is True else (0.0 if stochastic is False else None), stochastic is False, ("STOCHASTIC_VALIDATION_FAILED",) if stochastic is False else ()),
    ]
    scores = [v.score for v in votes if v.score is not None]
    vetoes = [reason for v in votes if v.veto for reason in v.reasons]
    return {"schema_version": "crypto_ai_swing_research_agent_panel_v1", "votes": [v.to_dict() for v in votes], "consensus_score": float(np.mean(scores)) if scores else None, "vetoes": vetoes, "advisory_entry_supported": bool(scores and not vetoes), "authority": "ADVISORY_ONLY", "live_decision_influence": False, "automatic_live_promotion": False}
