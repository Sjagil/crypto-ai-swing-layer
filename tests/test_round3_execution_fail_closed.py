from datetime import datetime, timedelta, timezone
from decimal import Decimal

import crypto_ai_swing.execution.router as router_module
from crypto_ai_swing.contracts import Authority, Side, TradeIntent
from crypto_ai_swing.execution.router import ExecutionRouter


class _BlockedCanonicalAuthority:
    submit_called = False

    def __init__(self, root):
        self.root = root

    def preflight(self, intent):
        return {
            "accepted": False,
            "status": "BLOCKED",
            "blockers": ["CANONICAL_TEST_BLOCK"],
            "orders_submitted": 0,
        }

    def submit_buy(self, intent):
        type(self).submit_called = True
        raise AssertionError("submit_buy must not run after blocked preflight")


def test_crypto_execution_authority_mode_has_no_direct_exchange_fallback(
    tmp_path,
    monkeypatch,
):
    config = {
        "execution": {"paper_enabled": True, "live_enabled": True},
        "execution_adapter": {
            "mode": "crypto_execution_authority",
            "intent_directory": "intents",
            "direct_bitvavo_fallback": False,
        },
    }
    monkeypatch.setattr(
        router_module,
        "CryptoAuthorityAdapter",
        _BlockedCanonicalAuthority,
    )
    _BlockedCanonicalAuthority.submit_called = False

    intent = TradeIntent.new(
        created_at=datetime.now(timezone.utc),
        market="BTC-EUR",
        side=Side.BUY,
        notional_eur=Decimal("10"),
        expected_edge_bps=100,
        estimated_round_trip_cost_bps=50,
        net_edge_bps=50,
        stop_pct=0.02,
        take_profit_pct=0.06,
        trailing_stop_pct=0.02,
        strategy="TEST",
        authority=Authority.LIVE,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
        metadata={},
    )
    result = ExecutionRouter(
        tmp_path,
        tmp_path,
        config,
    ).route(intent, mode="live")

    assert result.accepted is False
    assert result.blocker == "CANONICAL_TEST_BLOCK"
    assert _BlockedCanonicalAuthority.submit_called is False
