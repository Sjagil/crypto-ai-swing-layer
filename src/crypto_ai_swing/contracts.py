from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
import hashlib
import json
import uuid


class Side(str, Enum):
    BUY = "BUY"
    EXIT = "EXIT"
    HOLD = "HOLD"


class Authority(str, Enum):
    RESEARCH = "RESEARCH"
    SHADOW = "SHADOW"
    PAPER = "PAPER"
    LIVE = "LIVE"


@dataclass(frozen=True)
class ModelVote:
    source: str
    score: float
    confidence: float
    horizon_bars: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Signal:
    market: str
    timestamp: datetime
    side: Side
    score: float
    confidence: float
    expected_edge_bps: float
    stop_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    strategy: str
    votes: tuple[ModelVote, ...] = ()
    features: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("Signal timestamp must be timezone-aware")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Signal score must be in [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Signal confidence must be in [0, 1]")


@dataclass(frozen=True)
class RiskPlan:
    market: str
    equity_eur: Decimal
    available_cash_eur: Decimal
    order_notional_eur: Decimal
    risk_eur: Decimal
    stop_pct: float
    portfolio_heat_after: float
    approved: bool
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class TradeIntent:
    intent_id: str
    created_at: datetime
    market: str
    side: Side
    notional_eur: Decimal
    expected_edge_bps: float
    estimated_round_trip_cost_bps: float
    net_edge_bps: float
    stop_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    strategy: str
    authority: Authority
    expires_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new(**kwargs: Any) -> "TradeIntent":
        return TradeIntent(intent_id=str(uuid.uuid4()), **kwargs)

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["side"] = self.side.value
        raw["authority"] = self.authority.value
        raw["notional_eur"] = str(self.notional_eur)
        raw["created_at"] = self.created_at.astimezone(timezone.utc).isoformat()
        raw["expires_at"] = self.expires_at.astimezone(timezone.utc).isoformat()
        return raw

    def canonical_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()
