import os
from pathlib import Path

import pytest

from crypto_ai_swing.production.dataflow_audit import Round41DataFlowAudit
from crypto_ai_swing.settings import Settings


@pytest.mark.skipif(
    os.getenv("ROUND41_REAL_DATA") != "1",
    reason="set ROUND41_REAL_DATA=1 for real canonical public-data smoke",
)
def test_round41_real_canonical_public_dataflow():
    settings = Settings.load(Path.cwd())
    payload = Round41DataFlowAudit(settings).run(
        markets=["BTC-EUR", "ETH-EUR", "SOL-EUR"],
        timeframes=["15m", "1h", "4h"],
        network=True,
        require_trained_agents=False,
        deep_context=False,
    )
    assert payload["canonical_imports"]["ready"] is True
    assert payload["data"]["ready"] is True
    assert payload["data"]["market_bundle"]["ready"] is True
    assert payload["features"]["ready"] is True
    assert payload["canonical_portfolio_risk"]["ready"] is True
    assert payload["orders_submitted"] == 0
