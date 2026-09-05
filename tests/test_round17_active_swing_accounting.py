from __future__ import annotations

import inspect
from decimal import Decimal
from types import SimpleNamespace

from crypto_ai_swing.orchestration.proactive import ProactiveTrader


class _State:
    def __init__(self, positions):
        self._positions = positions

    def positions(self):
        return dict(self._positions)


class _Crypto:
    def __init__(self, marks):
        self._marks = marks

    def market_bundle(self, market, timeframe, **kwargs):
        del timeframe, kwargs
        return SimpleNamespace(
            microstructure={"best_bid": str(self._marks[market])}
        )


def _trader(*, positions=None, marks=None):
    trader = object.__new__(ProactiveTrader)
    trader.mode = "paper"
    trader.settings = SimpleNamespace(
        proactive={"shadow_equity_eur": 10000}
    )
    if positions is not None:
        trader.state = _State(positions)
    trader.crypto = _Crypto(marks or {})
    return trader


def test_simulated_account_without_state_preserves_legacy_fallback():
    trader = _trader()
    equity, cash, exposure = trader._simulated_portfolio_account()

    assert equity == Decimal(10000)
    assert cash == Decimal(10000)
    assert exposure == Decimal(0)


def test_simulated_account_marks_open_position_and_consumes_cash():
    position = SimpleNamespace(
        amount=Decimal("0.02"),
        entry_price=Decimal(50000),
        stop_pct=0.02,
    )
    trader = _trader(
        positions={"BTC-EUR": position},
        marks={"BTC-EUR": Decimal(51000)},
    )

    equity, cash, exposure = trader._simulated_portfolio_account()

    assert cash == Decimal(9000)
    assert exposure == Decimal(1020)
    assert equity == Decimal(10020)


def test_simulated_account_falls_back_to_cost_basis_when_mark_fails():
    position = SimpleNamespace(
        amount=Decimal("0.02"),
        entry_price=Decimal(50000),
        stop_pct=0.02,
    )
    trader = _trader(
        positions={"BTC-EUR": position},
        marks={},
    )

    equity, cash, exposure = trader._simulated_portfolio_account()

    assert cash == Decimal(9000)
    assert exposure == Decimal(1000)
    assert equity == Decimal(10000)


def test_cycle_wires_calculated_open_risk_and_position_capacity():
    source = inspect.getsource(ProactiveTrader.cycle)
    assert "open_risk_eur=open_risk_eur" in source
    assert "MAX_POSITIONS_REACHED" in source
    assert "remaining_position_slots" in source
