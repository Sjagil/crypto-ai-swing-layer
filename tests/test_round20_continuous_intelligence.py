from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_ai_swing.data.features import build_features
from crypto_ai_swing.intelligence.technical import (
    multi_timeframe_snapshot,
    technical_snapshot,
)


def _frame(rows: int = 260) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    close = np.linspace(100.0, 150.0, rows) + np.sin(np.arange(rows) / 7.0)
    close[-1] = close[-2] * 1.03
    volume = np.full(rows, 1000.0)
    volume[-1] = 3500.0
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.003,
            "low": close * 0.997,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_extended_indicators_exist():
    features = build_features(_frame())
    required = {
        "macd_hist_atr",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
        "breakout_55",
        "volume_ratio_20",
        "obv_slope_20",
        "cmf_20",
        "bb_squeeze_ratio",
        "atr_expansion",
        "ema20_slope_5",
        "roc_12",
    }
    assert required.issubset(set(features.columns))


def test_technical_snapshot_is_advisory():
    snapshot = technical_snapshot(_frame(), timeframe="1h")
    assert snapshot["status"] == "READY"
    assert -1.0 <= snapshot["technical_score"] <= 1.0
    assert snapshot["live_decision_influence"] is False
    assert snapshot["authority"] == "ADVISORY_ONLY"


def test_mtf_snapshot_has_states():
    frame = _frame()
    payload = multi_timeframe_snapshot(
        {"15m": frame, "1h": frame, "4h": frame, "1d": frame}
    )
    assert payload["status"] == "READY"
    assert len(payload["states"]) == 4
    assert payload["live_decision_influence"] is False
