from __future__ import annotations

from decimal import Decimal

from crypto_ai_swing.execution.paper_certification import (
    simulate_round_trip,
)


def test_paper_round_trip_charges_fees_and_slippage():
    result = simulate_round_trip(
        entry_reference=Decimal("100"),
        exit_reference=Decimal("103"),
        notional_eur=Decimal("1000"),
        fee_bps_per_side=Decimal("25"),
        slippage_bps_per_side=Decimal("2"),
    )
    assert result.entry.fill_price > Decimal("100")
    assert result.exit.fill_price < Decimal("103")
    assert result.entry.fee_eur > 0
    assert result.exit.fee_eur > 0
    assert result.realized_pnl_eur > 0
