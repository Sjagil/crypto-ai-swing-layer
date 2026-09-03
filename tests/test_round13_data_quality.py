from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from crypto_ai_swing.data.quality import audit_ohlcv_frame


def _frame(periods: int = 100) -> pd.DataFrame:
    index = pd.date_range(
        "2026-01-01",
        periods=periods,
        freq="1h",
        tz="UTC",
    )
    close = np.linspace(100.0, 120.0, periods)
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 1.0,
            "low": np.minimum(open_, close) - 1.0,
            "close": close,
            "volume": np.full(periods, 1000.0),
        },
        index=index,
    )


def test_quality_passes_clean_complete_frame():
    frame = _frame()
    now = (
        frame.index[-1].to_pydatetime()
        + timedelta(hours=1, minutes=5)
    )
    report = audit_ohlcv_frame(
        frame,
        timeframe="1h",
        now=now,
        policy={
            "market_data": {
                "minimum_coverage": 0.95,
                "maximum_duplicate_ratio": 0.0,
                "maximum_ohlc_violation_ratio": 0.0,
                "maximum_negative_volume_ratio": 0.0,
                "maximum_gap_ratio": 0.02,
                "maximum_outlier_ratio": 0.02,
            }
        },
    )
    assert report["healthy"] is True
    assert report["coverage"] == 1.0
    assert report["gap_ratio"] == 0.0


def test_quality_blocks_gap_and_invalid_ohlc():
    frame = _frame()
    frame = frame.drop(frame.index[40:50])
    frame.iloc[
        10,
        frame.columns.get_loc("high"),
    ] = 1.0
    now = (
        frame.index[-1].to_pydatetime()
        + timedelta(hours=1, minutes=5)
    )
    report = audit_ohlcv_frame(
        frame,
        timeframe="1h",
        now=now,
        policy={
            "market_data": {
                "minimum_coverage": 0.95,
                "maximum_duplicate_ratio": 0.0,
                "maximum_ohlc_violation_ratio": 0.0,
                "maximum_negative_volume_ratio": 0.0,
                "maximum_gap_ratio": 0.02,
                "maximum_outlier_ratio": 0.02,
            }
        },
    )
    assert report["healthy"] is False
    assert "GAP_RATIO_EXCEEDED" in report["reason_codes"]
    assert (
        "OHLC_VIOLATION_RATIO_EXCEEDED"
        in report["reason_codes"]
    )
