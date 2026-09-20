from decimal import Decimal
from types import SimpleNamespace

import pytest

from crypto_ai_swing.orchestration.proactive import ProactiveTrader


class FailingCanonicalAuthority:
    def account_snapshot(self, markets):
        raise RuntimeError("IP_NOT_WHITELISTED")


class ReadyCanonicalAuthority:
    def account_snapshot(self, markets):
        return {"status": "READY", "failures": [], "equity_eur": "1500", "cash_eur": "900", "exposure_eur": "600"}


def make_trader(mode: str, authority=None):
    trader = object.__new__(ProactiveTrader)
    trader.mode = mode
    trader.settings = SimpleNamespace(
        proactive={"shadow_equity_eur": 1234, "account": {"use_private_balances_in_shadow": False, "use_private_balances_in_paper": False, "require_private_balances_in_live": True}},
        execution={"costs": {"fee_bps_per_side": 25.0}},
        risk={"portfolio": {}},
    )
    trader.execution_authority = authority
    return trader


def test_shadow_does_not_require_private_exchange_account():
    equity, cash, exposure = make_trader("shadow")._account(["BTC-EUR"])
    assert (equity, cash, exposure) == (Decimal("1234"), Decimal("1234"), Decimal("0"))


def test_paper_does_not_require_private_exchange_account():
    equity, cash, exposure = make_trader("paper")._account(["BTC-EUR"])
    assert (equity, cash, exposure) == (Decimal("1234"), Decimal("1234"), Decimal("0"))


def test_live_fails_closed_when_canonical_account_is_unavailable():
    with pytest.raises(RuntimeError, match="Canonical crypto account snapshot unavailable"):
        make_trader("live", FailingCanonicalAuthority())._account(["BTC-EUR"])


def test_live_uses_canonical_account_snapshot():
    equity, cash, exposure = make_trader("live", ReadyCanonicalAuthority())._account(["BTC-EUR"])
    assert (equity, cash, exposure) == (Decimal("1500"), Decimal("900"), Decimal("600"))
