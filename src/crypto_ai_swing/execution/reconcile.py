from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass
class PositionState:
    market: str
    quantity: Decimal
    average_price: Decimal
    realized_pnl_eur: Decimal = Decimal("0")


def apply_fill(
    position: PositionState | None,
    market: str,
    side: str,
    quantity: Decimal,
    price: Decimal,
) -> PositionState | None:
    if quantity <= 0 or price <= 0:
        raise ValueError("Fill quantity and price must be positive")

    if side.upper() == "BUY":
        if position is None:
            return PositionState(market, quantity, price)
        total_cost = position.quantity * position.average_price + quantity * price
        total_qty = position.quantity + quantity
        position.average_price = total_cost / total_qty
        position.quantity = total_qty
        return position

    if side.upper() == "SELL":
        if position is None or quantity > position.quantity:
            raise ValueError("Sell fill exceeds managed position")
        position.realized_pnl_eur += quantity * (price - position.average_price)
        position.quantity -= quantity
        if position.quantity == 0:
            return None
        return position

    raise ValueError(f"Unsupported fill side: {side}")
