from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
from crypto_ai_swing.contracts import RiskPlan, Signal


def build_risk_plan(
    signal: Signal,
    equity_eur: Decimal,
    available_cash_eur: Decimal,
    current_exposure_eur: Decimal,
    current_open_risk_eur: Decimal,
    config: dict,
) -> RiskPlan:
    portfolio = config.get("portfolio", {})
    trade = config.get("trade", {})
    blockers: list[str] = []

    risk_fraction = Decimal(str(trade.get("risk_per_trade_fraction", 0.0065)))
    max_position_fraction = Decimal(str(portfolio.get("max_single_position_fraction", 0.25)))
    min_cash_fraction = Decimal(str(portfolio.get("minimum_cash_fraction", 0.10)))
    max_total = Decimal(str(portfolio.get("max_total_exposure_fraction", 0.85)))
    hard_heat = Decimal(str(portfolio.get("hard_max_heat_fraction", 0.06)))

    risk_budget = equity_eur * risk_fraction
    stop = Decimal(str(max(signal.stop_pct, 1e-6)))
    raw_notional = risk_budget / stop
    max_single = equity_eur * max_position_fraction
    reserve = equity_eur * min_cash_fraction
    cash_cap = max(Decimal("0"), available_cash_eur - reserve)
    exposure_cap = max(Decimal("0"), equity_eur * max_total - current_exposure_eur)

    notional = min(raw_notional, max_single, cash_cap, exposure_cap)
    notional = notional.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    risk_eur = (notional * stop).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    heat_after = (
        (current_open_risk_eur + risk_eur) / equity_eur
        if equity_eur > 0
        else Decimal("1")
    )

    if signal.side.value != "BUY":
        blockers.append("NO_ENTRY_SIGNAL")
    if notional <= 0:
        blockers.append("NO_CAPACITY")
    if heat_after > hard_heat:
        blockers.append("PORTFOLIO_HEAT")
    if available_cash_eur <= reserve:
        blockers.append("CASH_RESERVE")

    return RiskPlan(
        market=signal.market,
        equity_eur=equity_eur,
        available_cash_eur=available_cash_eur,
        order_notional_eur=notional,
        risk_eur=risk_eur,
        stop_pct=signal.stop_pct,
        portfolio_heat_after=float(heat_after),
        approved=not blockers,
        blockers=tuple(blockers),
    )
