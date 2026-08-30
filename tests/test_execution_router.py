from datetime import datetime, timezone, timedelta
from decimal import Decimal
from crypto_ai_swing.contracts import TradeIntent, Side, Authority
from crypto_ai_swing.execution.router import ExecutionRouter


def make_intent(authority):
    now = datetime.now(timezone.utc)
    return TradeIntent.new(
        created_at=now,
        market="BTC-EUR",
        side=Side.BUY,
        notional_eur=Decimal("10"),
        expected_edge_bps=100,
        estimated_round_trip_cost_bps=30,
        net_edge_bps=70,
        stop_pct=0.03,
        take_profit_pct=0.09,
        trailing_stop_pct=0.03,
        strategy="TEST",
        authority=authority,
        expires_at=now + timedelta(minutes=2),
    )


def test_live_is_disabled_by_default(tmp_path, monkeypatch):
    cfg = {
        "execution": {"paper_enabled": True, "live_enabled": False},
        "execution_adapter": {
            "intent_directory": "output/intents",
            "command": [],
            "live_command": [],
        },
    }
    router = ExecutionRouter(tmp_path, tmp_path, cfg)
    result = router.route(make_intent(Authority.LIVE), "live")
    assert not result.accepted
    assert result.blocker == "LIVE_DISABLED"
