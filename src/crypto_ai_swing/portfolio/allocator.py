from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from crypto_ai_swing.contracts import Signal, RiskPlan
from .risk import build_risk_plan


@dataclass
class Allocation:
    signal: Signal
    risk: RiskPlan


def allocate(
    signals: list[Signal],
    equity_eur: Decimal,
    cash_eur: Decimal,
    exposure_eur: Decimal,
    open_risk_eur: Decimal,
    config: dict,
) -> list[Allocation]:
    max_positions = int(config.get("portfolio", {}).get("max_positions", 5))
    candidates = sorted(
        (s for s in signals if s.side.value == "BUY"),
        key=lambda s: (s.score, s.expected_edge_bps),
        reverse=True,
    )[:max_positions]

    allocations: list[Allocation] = []
    running_cash = cash_eur
    running_exposure = exposure_eur
    running_risk = open_risk_eur

    for signal in candidates:
        plan = build_risk_plan(
            signal,
            equity_eur,
            running_cash,
            running_exposure,
            running_risk,
            config,
        )
        allocations.append(Allocation(signal, plan))
        if plan.approved:
            running_cash -= plan.order_notional_eur
            running_exposure += plan.order_notional_eur
            running_risk += plan.risk_eur
    return allocations
