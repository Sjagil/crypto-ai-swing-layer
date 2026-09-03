from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge

TIMEFRAME_DELTA: dict[str, timedelta] = {
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}

FINER_TIMEFRAME: dict[str, str] = {
    "15m": "5m",
    "1h": "15m",
    "2h": "1h",
    "4h": "1h",
    "1d": "4h",
}


def _native_timeframe(timeframe: str) -> str:
    value = str(timeframe).strip()
    return "1W" if value.lower() == "1w" else value


def _utc_index(values: Iterable[Any]) -> pd.DatetimeIndex:
    index = pd.to_datetime(list(values), utc=True, errors="coerce")
    return pd.DatetimeIndex(index).dropna().sort_values().unique()


def _frame_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if frame is None or frame.empty:
        return pd.DatetimeIndex([], tz="UTC")
    if isinstance(frame.index, pd.DatetimeIndex):
        return _utc_index(frame.index)
    for candidate in ("timestamp", "datetime", "date", "time"):
        if candidate in frame.columns:
            return _utc_index(frame[candidate])
    raise ValueError("OHLCV frame has no timestamp index or timestamp column")


def _record_index(records: Iterable[Any]) -> pd.DatetimeIndex:
    timestamps = []
    for record in records:
        value = getattr(record, "timestamp", None)
        if value is None and isinstance(record, dict):
            value = record.get("timestamp")
        if value is not None:
            timestamps.append(value)
    return _utc_index(timestamps)


def _expected_index(
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeframe: str,
) -> pd.DatetimeIndex:
    return pd.date_range(
        start,
        end,
        freq=pd.Timedelta(TIMEFRAME_DELTA[timeframe]),
        tz="UTC",
    )


def _group_missing(
    missing: pd.DatetimeIndex,
    timeframe: str,
) -> list[dict[str, Any]]:
    if len(missing) == 0:
        return []
    delta = pd.Timedelta(TIMEFRAME_DELTA[timeframe])
    rows: list[dict[str, Any]] = []
    start = previous = missing[0]
    for current in missing[1:]:
        if current - previous != delta:
            rows.append(
                {
                    "start": start.isoformat(),
                    "end": previous.isoformat(),
                    "count": int((previous - start) / delta) + 1,
                }
            )
            start = current
        previous = current
    rows.append(
        {
            "start": start.isoformat(),
            "end": previous.isoformat(),
            "count": int((previous - start) / delta) + 1,
        }
    )
    return rows


def classify_provider_semantics(
    *,
    local_frame: pd.DataFrame,
    provider_records: Iterable[Any],
    timeframe: str,
    finer_records: Iterable[Any] | None = None,
) -> dict[str, Any]:
    selected_tf = str(timeframe)
    if selected_tf not in TIMEFRAME_DELTA:
        raise ValueError(f"unsupported semantic timeframe: {selected_tf}")

    local_index = _frame_index(local_frame)
    if len(local_index) < 2:
        raise ValueError("semantic classification requires at least two candles")

    expected = _expected_index(
        local_index.min(),
        local_index.max(),
        selected_tf,
    )
    local_missing = expected.difference(local_index)

    provider_index = _record_index(provider_records)
    provider_index = provider_index[
        (provider_index >= expected.min())
        & (provider_index <= expected.max())
    ]

    recoverable = local_missing.intersection(provider_index)
    provider_absent = local_missing.difference(provider_index)

    finer_index = _record_index(finer_records or [])
    cross_tf: list[pd.Timestamp] = []
    confirmed_zero_trade: list[pd.Timestamp] = []
    delta = pd.Timedelta(TIMEFRAME_DELTA[selected_tf])

    for timestamp in provider_absent:
        if len(finer_index):
            inside = finer_index[
                (finer_index >= timestamp)
                & (finer_index < timestamp + delta)
            ]
        else:
            inside = pd.DatetimeIndex([], tz="UTC")
        if len(inside):
            cross_tf.append(timestamp)
        else:
            confirmed_zero_trade.append(timestamp)

    cross_tf_index = pd.DatetimeIndex(cross_tf)
    zero_trade_index = pd.DatetimeIndex(confirmed_zero_trade)

    integrity_blockers: list[str] = []
    if len(recoverable):
        integrity_blockers.append("RECOVERABLE_PROVIDER_GAP")
    if len(cross_tf_index):
        integrity_blockers.append("PROVIDER_CROSS_TIMEFRAME_INCONSISTENCY")

    expected_count = max(1, len(expected))
    return {
        "schema_version": "crypto_ai_swing_provider_semantics_v1",
        "timeframe": selected_tf,
        "expected_intervals": len(expected),
        "observed_intervals": len(local_index),
        "raw_missing_intervals": len(local_missing),
        "provider_returned_intervals": len(provider_index),
        "recoverable_provider_gap_count": len(recoverable),
        "confirmed_zero_trade_count": len(zero_trade_index),
        "cross_timeframe_inconsistency_count": len(cross_tf_index),
        "observed_candle_ratio": float(len(local_index) / expected_count),
        "confirmed_zero_trade_ratio": float(
            len(zero_trade_index) / expected_count
        ),
        "integrity_healthy": not integrity_blockers,
        "integrity_blockers": integrity_blockers,
        "recoverable_ranges": _group_missing(recoverable, selected_tf),
        "confirmed_zero_trade_ranges": _group_missing(
            zero_trade_index,
            selected_tf,
        ),
        "cross_timeframe_inconsistency_ranges": _group_missing(
            cross_tf_index,
            selected_tf,
        ),
        "semantics": {
            "missing_candle_is_not_automatically_corruption": True,
            "provider_refetch_required": True,
            "cross_timeframe_confirmation_required": True,
            "forward_fill_used": False,
            "synthetic_raw_candles_used": False,
            "cross_provider_synthesis_used": False,
        },
    }


@dataclass(frozen=True)
class ProviderSemanticAuditor:
    settings: Any

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "crypto",
            CryptoLibraryBridge(self.settings.crypto_repo_root),
        )
        object.__setattr__(self, "loader", self.crypto.data_loader())

    def _download(
        self,
        *,
        market: str,
        timeframe: str,
        start,
        end,
    ) -> list[Any]:
        return list(
            self.crypto._run(
                self.loader.download_ohlcv(
                    provider="bitvavo",
                    market=str(market).upper(),
                    timeframe=_native_timeframe(timeframe),
                    start=start,
                    end=end,
                    resume=False,
                    persist=False,
                )
            )
        )

    def audit_series(
        self,
        *,
        market: str,
        timeframe: str,
        frame: pd.DataFrame,
    ) -> dict[str, Any]:
        selected_market = str(market).upper()
        selected_tf = str(timeframe)
        index = _frame_index(frame)
        if len(index) < 2:
            return {
                "market": selected_market,
                "timeframe": selected_tf,
                "status": "BLOCKED",
                "integrity_healthy": False,
                "integrity_blockers": ["INSUFFICIENT_LOCAL_HISTORY"],
                "orders_submitted": 0,
            }

        delta = TIMEFRAME_DELTA[selected_tf]
        start = index.min().to_pydatetime()
        end = (index.max() + pd.Timedelta(delta)).to_pydatetime()

        provider_records = self._download(
            market=selected_market,
            timeframe=selected_tf,
            start=start,
            end=end,
        )

        finer_tf = FINER_TIMEFRAME.get(selected_tf)
        finer_records: list[Any] = []
        if finer_tf is not None:
            finer_records = self._download(
                market=selected_market,
                timeframe=finer_tf,
                start=start,
                end=end,
            )

        result = classify_provider_semantics(
            local_frame=frame,
            provider_records=provider_records,
            timeframe=selected_tf,
            finer_records=finer_records,
        )
        return {
            "market": selected_market,
            "timeframe": selected_tf,
            "status": (
                "INTEGRITY_HEALTHY"
                if result["integrity_healthy"]
                else "INTEGRITY_BLOCKED"
            ),
            "finer_timeframe_probe": finer_tf,
            **result,
            "source": (
                "Sjagil/crypto:DataLoader.download_ohlcv"
                "+provider-semantic-cross-timeframe-reconciliation"
            ),
            "orders_generated": 0,
            "orders_submitted": 0,
        }


def aggregate_provider_semantics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(rows)
    healthy = sum(bool(row.get("integrity_healthy")) for row in rows)
    return {
        "schema_version": "crypto_ai_swing_provider_semantics_aggregate_v1",
        "total_series": total,
        "integrity_healthy_series": healthy,
        "integrity_blocked_series": total - healthy,
        "recoverable_provider_gap_count": sum(
            int(row.get("recoverable_provider_gap_count") or 0)
            for row in rows
        ),
        "confirmed_zero_trade_count": sum(
            int(row.get("confirmed_zero_trade_count") or 0)
            for row in rows
        ),
        "cross_timeframe_inconsistency_count": sum(
            int(row.get("cross_timeframe_inconsistency_count") or 0)
            for row in rows
        ),
        "integrity_all_healthy": bool(total) and healthy == total,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
