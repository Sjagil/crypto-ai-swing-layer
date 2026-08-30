from decimal import Decimal
from types import SimpleNamespace

import pytest

from crypto_ai_swing.execution.bitvavo import BitvavoError
from crypto_ai_swing.orchestration.proactive import ProactiveTrader


class FailingPrivateClient:
    api_key = "test"
    api_secret = "test"

    def balances(self):
        raise BitvavoError("IP_NOT_WHITELISTED")


def make_trader(mode: str):
    trader = object.__new__(ProactiveTrader)
    trader.mode = mode
    trader.settings = SimpleNamespace(
        proactive={
            "shadow_equity_eur": 1234,
            "account": {
                "use_private_balances_in_shadow": False,
                "use_private_balances_in_paper": False,
                "require_private_balances_in_live": True,
            },
        }
    )
    trader.client = FailingPrivateClient()
    return trader


def test_shadow_does_not_require_private_bitvavo_account():
    trader = make_trader("shadow")

    equity, cash, exposure = trader._account(["BTC-EUR"])

    assert equity == Decimal("1234")
    assert cash == Decimal("1234")
    assert exposure == Decimal("0")


def test_paper_does_not_require_private_bitvavo_account():
    trader = make_trader("paper")

    equity, cash, exposure = trader._account(["BTC-EUR"])

    assert equity == Decimal("1234")
    assert cash == Decimal("1234")
    assert exposure == Decimal("0")


def test_live_fails_closed_when_private_account_is_unavailable():
    trader = make_trader("live")

    with pytest.raises(BitvavoError):
        trader._account(["BTC-EUR"])
