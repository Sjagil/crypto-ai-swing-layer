from datetime import datetime, timezone
from decimal import Decimal
from crypto_ai_swing.contracts import Signal, Side
from crypto_ai_swing.portfolio.risk import build_risk_plan


def test_risk_plan_respects_caps():
    signal = Signal(
        market="BTC-EUR",
        timestamp=datetime.now(timezone.utc),
        side=Side.BUY,
        score=0.8,
        confidence=0.8,
        expected_edge_bps=100,
        stop_pct=0.03,
        take_profit_pct=0.09,
        trailing_stop_pct=0.03,
        strategy="TEST",
    )
    cfg = {
        "portfolio": {
            "max_single_position_fraction": 0.25,
            "minimum_cash_fraction": 0.10,
            "max_total_exposure_fraction": 0.85,
            "hard_max_heat_fraction": 0.06,
        },
        "trade": {"risk_per_trade_fraction": 0.0065},
    }
    plan = build_risk_plan(
        signal,
        Decimal("10000"),
        Decimal("10000"),
        Decimal("0"),
        Decimal("0"),
        cfg,
    )
    assert plan.order_notional_eur <= Decimal("2500")
    assert plan.approved
