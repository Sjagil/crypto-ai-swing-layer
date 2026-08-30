from datetime import datetime, timedelta, timezone
from decimal import Decimal

from crypto_ai_swing.contracts import Authority, Side, TradeIntent
from crypto_ai_swing.execution.router import ExecutionRouter


def test_crypto_execution_authority_mode_has_no_direct_exchange_fallback(tmp_path, monkeypatch):
    config = {
        "execution": {"paper_enabled": True, "live_enabled": True},
        "execution_adapter": {
            "mode": "crypto_execution_authority",
            "intent_directory": "intents",
            "direct_bitvavo_fallback": False,
        },
    }
    # Make the existing environment gate pass far enough that the adapter boundary is tested.
    monkeypatch.setenv("LIVE_TRADING_ALLOWED", "true")
    monkeypatch.setenv("WM_REAL_ORDER_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("WM_I_UNDERSTAND_LIVE_RISK", "true")
    monkeypatch.setenv("WM_LIVE_APPROVAL", "APPROVED_LIVE_ACTIVE")
    monkeypatch.setenv("WITHDRAWALS_DISABLED", "true")
    monkeypatch.setenv("BITVAVO_TRADE_KEY_SCOPE", "read,trade")
    monkeypatch.setenv("BITVAVO_TRADE_API_KEY", "x")
    monkeypatch.setenv("BITVAVO_TRADE_API_SECRET", "y")
    monkeypatch.setenv("WM_OPERATOR_ID", "1")
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
    result = ExecutionRouter(tmp_path, tmp_path, config).route(intent, mode="live")
    assert result.accepted is False
    assert result.blocker == "CRYPTO_EXECUTION_AUTHORITY_SUBMISSION_NOT_MAPPED"
