from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from crypto_ai_swing.contracts import Authority, ModelVote, Side, Signal


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        selected = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(selected):
        return None
    return selected


def _bounded(value: Any, default: float = 0.0) -> float:
    selected = _finite_float(value)
    if selected is None:
        selected = float(default)
    return max(0.0, min(1.0, selected))


@dataclass(frozen=True)
class DecisionPacket:
    """Single AI/research decision contract before canonical portfolio authority."""

    packet_id: str
    created_at: datetime
    market: str
    side: Side
    authority: Authority
    entry_price: float
    score: float
    confidence: float
    calibrated_win_probability: float | None
    predicted_return: float | None
    predicted_mae: float | None
    expected_edge_bps: float
    stop_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    strategy: str
    prospective_qualified: bool
    uncertainty: float
    probability_source: str | None = None
    votes: tuple[ModelVote, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("DecisionPacket created_at must be timezone-aware")
        if not self.market or "-" not in self.market:
            raise ValueError("DecisionPacket market must be canonical BASE-QUOTE")
        if self.entry_price <= 0:
            raise ValueError("DecisionPacket entry_price must be positive")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("DecisionPacket score must be in [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("DecisionPacket confidence must be in [0, 1]")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("DecisionPacket uncertainty must be in [0, 1]")
        if self.calibrated_win_probability is not None and not (
            0.0 <= self.calibrated_win_probability <= 1.0
        ):
            raise ValueError(
                "DecisionPacket calibrated_win_probability must be in [0, 1]"
            )
        if not 0.0 < self.stop_pct < 1.0:
            raise ValueError("DecisionPacket stop_pct must be in (0, 1)")
        if self.take_profit_pct <= 0.0:
            raise ValueError("DecisionPacket take_profit_pct must be positive")

    @classmethod
    def from_signal(
        cls,
        signal: Signal,
        *,
        context: Mapping[str, Any] | None,
        authority: Authority,
    ) -> DecisionPacket:
        ctx = dict(context or {})
        agents = dict(ctx.get("agents") or {})
        diagnostics = dict(agents.get("diagnostics") or {})
        head_influence = dict(diagnostics.get("head_influence") or {})

        # The pipeline already exposes ml_probability only when the calibrated
        # alpha head is qualified. Never let an unqualified raw head drive Kelly.
        qualified_probability = _finite_float(ctx.get("ml_probability"))
        probability_source: str | None = None
        if qualified_probability is not None:
            probability_source = "agents.calibrated_alpha"
        elif bool(head_influence.get("alpha")):
            qualified_probability = _finite_float(
                agents.get("alpha_probability")
            )
            if qualified_probability is not None:
                probability_source = "agents.calibrated_alpha"

        predicted_return = _finite_float(ctx.get("predicted_return"))
        predicted_mae = _finite_float(ctx.get("predicted_mae"))

        prospective_status = str(
            ctx.get("prospective_context_status") or "UNKNOWN"
        ).upper()
        prospective_qualified = prospective_status not in {
            "BLOCK_NEW_ENTRIES",
            "BLOCKED",
            "FAILED",
            "NOT_READY",
        }
        if prospective_status == "UNKNOWN":
            prospective_qualified = not bool(ctx.get("entry_blocked", False))

        uncertainty = _bounded(
            diagnostics.get("uncertainty"),
            default=1.0 - float(signal.confidence),
        )
        evidence = {
            "agent_head_influence": head_influence,
            "agent_diagnostics": diagnostics,
            "research_meta_score": ctx.get("research_meta_score"),
            "prospective_context_status": prospective_status,
            "entry_blocked": bool(ctx.get("entry_blocked", False)),
            "data_source": ctx.get("data_source"),
            "reliability_evidence": ctx.get("reliability_evidence"),
            "edge_manager": ctx.get("edge_manager"),
            "mtf_challenger": ctx.get("mtf_challenger"),
            "observed_spread_bps": ctx.get("spread_bps"),
        }
        metadata = {
            "signal_features": dict(signal.features),
            "context_keys": sorted(str(key) for key in ctx),
            "automatic_live_promotion": False,
        }
        return cls(
            packet_id=str(uuid.uuid4()),
            created_at=datetime.now(UTC),
            market=str(signal.market).upper(),
            side=signal.side,
            authority=authority,
            entry_price=float(signal.features.get("price", 0.0) or 0.0),
            score=float(signal.score),
            confidence=float(signal.confidence),
            calibrated_win_probability=qualified_probability,
            predicted_return=predicted_return,
            predicted_mae=predicted_mae,
            expected_edge_bps=float(signal.expected_edge_bps),
            stop_pct=float(signal.stop_pct),
            take_profit_pct=float(signal.take_profit_pct),
            trailing_stop_pct=float(signal.trailing_stop_pct),
            strategy=str(signal.strategy),
            prospective_qualified=prospective_qualified,
            uncertainty=uncertainty,
            probability_source=probability_source,
            votes=tuple(signal.votes),
            evidence=evidence,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["created_at"] = self.created_at.astimezone(UTC).isoformat()
        raw["side"] = self.side.value
        raw["authority"] = self.authority.value
        return raw

    def canonical_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
