from datetime import datetime, timezone

import pandas as pd

from crypto_ai_swing.strategies.swing import build_signal
from crypto_ai_swing.contracts import Side


def row():
    return pd.Series(
        {
            "trend_8_20": 1.0,
            "trend_20_50": 1.0,
            "trend_50_200": 1.0,
            "rsi_14": 55.0,
            "ret_8": 0.02,
            "breakout_20": 1.0,
            "volume_z_48": 1.0,
            "atr_pct": 0.02,
        }
    )


def test_crypto_repo_context_is_bounded():
    base = build_signal(
        "BTC-EUR",
        datetime.now(timezone.utc),
        row(),
        minimum_entry_score=0.0,
    )
    boosted = build_signal(
        "BTC-EUR",
        datetime.now(timezone.utc),
        row(),
        mtf_score=1.0,
        orderflow_score=1.0,
        minimum_entry_score=0.0,
    )
    assert boosted.score - base.score <= 0.1400001


def test_crypto_repo_context_gate_vetoes_entry():
    signal = build_signal(
        "BTC-EUR",
        datetime.now(timezone.utc),
        row(),
        mtf_score=1.0,
        orderflow_score=1.0,
        context_entry_blocked=True,
        minimum_entry_score=0.0,
    )
    assert signal.side == Side.HOLD
