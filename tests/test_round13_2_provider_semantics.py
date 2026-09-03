from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from crypto_ai_swing.data.provider_semantics import classify_provider_semantics


@dataclass
class _Record:
    timestamp: datetime


def _frame(index: list[str]) -> pd.DataFrame:
    idx = pd.to_datetime(index, utc=True)
    return pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        },
        index=idx,
    )


def _records(index: list[str]) -> list[_Record]:
    return [
        _Record(pd.Timestamp(value).to_pydatetime())
        for value in index
    ]


def test_provider_absent_without_finer_candles_is_zero_trade_not_corruption():
    local = _frame(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T02:00:00Z",
        ]
    )
    provider = _records(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T02:00:00Z",
        ]
    )
    result = classify_provider_semantics(
        local_frame=local,
        provider_records=provider,
        timeframe="1h",
        finer_records=[],
    )
    assert result["raw_missing_intervals"] == 1
    assert result["confirmed_zero_trade_count"] == 1
    assert result["recoverable_provider_gap_count"] == 0
    assert result["integrity_healthy"] is True


def test_provider_refetch_recovers_missing_local_candle_and_blocks_integrity():
    local = _frame(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T02:00:00Z",
        ]
    )
    provider = _records(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T02:00:00Z",
        ]
    )
    result = classify_provider_semantics(
        local_frame=local,
        provider_records=provider,
        timeframe="1h",
        finer_records=[],
    )
    assert result["recoverable_provider_gap_count"] == 1
    assert "RECOVERABLE_PROVIDER_GAP" in result["integrity_blockers"]
    assert result["integrity_healthy"] is False


def test_missing_high_tf_with_finer_candle_is_provider_inconsistency():
    local = _frame(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T02:00:00Z",
        ]
    )
    provider = _records(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T02:00:00Z",
        ]
    )
    finer = _records(["2026-01-01T01:15:00Z"])
    result = classify_provider_semantics(
        local_frame=local,
        provider_records=provider,
        timeframe="1h",
        finer_records=finer,
    )
    assert result["cross_timeframe_inconsistency_count"] == 1
    assert result["confirmed_zero_trade_count"] == 0
    assert result["integrity_healthy"] is False
