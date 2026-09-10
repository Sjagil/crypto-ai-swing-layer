from datetime import UTC
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_ai_swing.bridge.canonical_portfolio import (
    CanonicalPortfolioRiskBridge,
)
from crypto_ai_swing.contracts import Authority, Side, Signal
from crypto_ai_swing.decision_packet import DecisionPacket
from crypto_ai_swing.settings import Settings


def test_round41_reaches_real_canonical_cost_risk_and_kelly_modules():
    settings = Settings.load(Path.cwd())
    bridge = CanonicalPortfolioRiskBridge(
        settings.crypto_repo_root,
        project_root=settings.project_root,
        risk_config=settings.risk,
    )
    index = pd.date_range(
        "2026-01-01",
        periods=120,
        freq="1h",
        tz=UTC,
    )
    close = 50000.0 * np.exp(np.linspace(0.0, 0.03, len(index)))
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.full(len(index), 100.0),
        },
        index=index,
    )
    signal = Signal(
        market="BTC-EUR",
        timestamp=index[-1].to_pydatetime(),
        side=Side.BUY,
        score=0.75,
        confidence=0.70,
        expected_edge_bps=160.0,
        stop_pct=0.02,
        take_profit_pct=0.06,
        trailing_stop_pct=0.02,
        strategy="ROUND41_INTEGRATION",
        edge_source="CALIBRATED_TEST",
        features={"price": float(close[-1])},
    )
    packet = DecisionPacket.from_signal(
        signal,
        context={
            "ml_probability": 0.60,
            "agents": {
                "alpha_probability": 0.60,
                "diagnostics": {"head_influence": {"alpha": True}},
            },
            "prospective_context_status": "READY",
            "data_source": "Sjagil/crypto",
        },
        authority=Authority.SHADOW,
    )
    result = bridge.assess(
        packet,
        equity_eur=Decimal(2000),
        cash_eur=Decimal(2000),
        exposure_eur=Decimal(0),
        open_risk_eur=Decimal(0),
        positions=[],
        frames={"BTC-EUR": frame},
    )

    assert result.canonical_risk["backend"].startswith(
        "Sjagil/crypto:risk.risk_manager.RiskManager"
    )
    assert result.canonical_cost_model_version.startswith(
        "canonical_cost_v1:"
    )
    assert result.kelly["applied"] is True
    assert result.kelly["canonical_risk_never_widened"] is True
    canonical_notional = (
        result.canonical_risk["approved_quantity"] * packet.entry_price
    )
    assert float(result.risk_plan.order_notional_eur) <= canonical_notional + 1e-8
