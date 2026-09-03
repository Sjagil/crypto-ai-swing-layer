from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd


FREQUENCY_ALIASES = {
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1d",
    "1w": "7d",
    "1W": "7d",
}
TIMEFRAME_SECONDS = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
    "1W": 604800,
}
REQUIRED = ("open", "high", "low", "close", "volume")


def _normalized_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(column).lower() for column in out.columns]
    if not isinstance(out.index, pd.DatetimeIndex):
        for candidate in ("timestamp", "datetime", "date", "time"):
            if candidate in out.columns:
                out[candidate] = pd.to_datetime(
                    out[candidate],
                    utc=True,
                    errors="coerce",
                )
                out = out.set_index(candidate)
                break
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("OHLCV requires a timestamp index")
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out.sort_index()


def audit_ohlcv_frame(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    policy: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Audit raw OHLCV without repairing or forward-filling it."""

    if frame is None or frame.empty:
        return {
            "schema_version": "crypto_ai_swing_frame_quality_v2",
            "status": "BLOCKED",
            "healthy": False,
            "reason_codes": ["DATASET_EMPTY"],
            "rows": 0,
            "timeframe": timeframe,
        }

    x = _normalized_index(frame)
    missing = [column for column in REQUIRED if column not in x.columns]
    if missing:
        return {
            "schema_version": "crypto_ai_swing_frame_quality_v2",
            "status": "BLOCKED",
            "healthy": False,
            "reason_codes": ["MISSING_OHLCV_COLUMNS"],
            "missing_columns": missing,
            "rows": int(len(x)),
            "timeframe": timeframe,
        }

    selected = dict(policy or {})
    limits = dict(selected.get("market_data", selected) or {})
    numeric = x.loc[:, REQUIRED].apply(pd.to_numeric, errors="coerce")
    row_count = max(1, len(numeric))

    duplicate_count = int(x.index.duplicated(keep=False).sum())
    duplicate_ratio = float(duplicate_count / row_count)
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    nonfinite_ratio = float((~finite.all(axis=1)).mean())

    values = numeric.replace([np.inf, -np.inf], np.nan)
    ohlc_bad = (
        (values["open"] <= 0)
        | (values["high"] <= 0)
        | (values["low"] <= 0)
        | (values["close"] <= 0)
        | (
            values["high"]
            < values[["open", "close", "low"]].max(axis=1)
        )
        | (
            values["low"]
            > values[["open", "close", "high"]].min(axis=1)
        )
    )
    ohlc_violation_ratio = float(ohlc_bad.fillna(True).mean())
    negative_volume_ratio = float(
        (values["volume"] < 0).fillna(True).mean()
    )

    unique_index = x.index[~x.index.duplicated(keep="last")]
    gap_ratio = 0.0
    coverage = 1.0
    missing_intervals = 0
    expected_rows = int(len(unique_index))
    frequency = FREQUENCY_ALIASES.get(timeframe)
    if frequency and len(unique_index) > 1:
        expected = pd.date_range(
            unique_index.min(),
            unique_index.max(),
            freq=frequency,
            tz="UTC",
        )
        expected_rows = max(1, len(expected))
        missing_intervals = int(len(expected.difference(unique_index)))
        gap_ratio = float(missing_intervals / expected_rows)
        coverage = float(min(1.0, len(unique_index) / expected_rows))

    close = values["close"]
    returns = (
        np.log(close.where(close > 0))
        .diff()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    outlier_ratio = 0.0
    if len(returns) >= 20:
        median = float(returns.median())
        mad = float((returns - median).abs().median())
        if mad > 0:
            robust_z = 0.67448975 * (returns - median).abs() / mad
            outlier_ratio = float((robust_z > 12.0).mean())

    reason_codes: list[str] = []
    latest_open = unique_index.max()
    latest_close = None
    age_seconds = None
    seconds = TIMEFRAME_SECONDS.get(timeframe)
    current = pd.Timestamp(now or datetime.now(timezone.utc))
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")
    if seconds is not None:
        latest_close = latest_open + pd.to_timedelta(seconds, unit="s")
        age_seconds = max(
            0.0,
            float((current - latest_close).total_seconds()),
        )
        if latest_close > current:
            reason_codes.append("INCOMPLETE_LAST_CANDLE")
        if age_seconds > seconds * 2.25:
            reason_codes.append("STALE_CANDLE_CHAIN")

    thresholds = {
        "minimum_coverage": float(limits.get("minimum_coverage", 0.95)),
        "maximum_duplicate_ratio": float(
            limits.get("maximum_duplicate_ratio", 0.0)
        ),
        "maximum_ohlc_violation_ratio": float(
            limits.get("maximum_ohlc_violation_ratio", 0.0)
        ),
        "maximum_negative_volume_ratio": float(
            limits.get("maximum_negative_volume_ratio", 0.0)
        ),
        "maximum_gap_ratio": float(
            limits.get("maximum_gap_ratio", 0.02)
        ),
        "maximum_outlier_ratio": float(
            limits.get("maximum_outlier_ratio", 0.02)
        ),
    }
    checks = (
        (
            coverage < thresholds["minimum_coverage"],
            "COVERAGE_BELOW_MINIMUM",
        ),
        (
            duplicate_ratio > thresholds["maximum_duplicate_ratio"],
            "DUPLICATE_RATIO_EXCEEDED",
        ),
        (
            ohlc_violation_ratio
            > thresholds["maximum_ohlc_violation_ratio"],
            "OHLC_VIOLATION_RATIO_EXCEEDED",
        ),
        (
            negative_volume_ratio
            > thresholds["maximum_negative_volume_ratio"],
            "NEGATIVE_VOLUME_RATIO_EXCEEDED",
        ),
        (
            gap_ratio > thresholds["maximum_gap_ratio"],
            "GAP_RATIO_EXCEEDED",
        ),
        (
            outlier_ratio > thresholds["maximum_outlier_ratio"],
            "OUTLIER_RATIO_EXCEEDED",
        ),
        (nonfinite_ratio > 0, "NONFINITE_ROWS_PRESENT"),
    )
    for failed, code in checks:
        if failed:
            reason_codes.append(code)
    reason_codes = list(dict.fromkeys(reason_codes))
    healthy = not reason_codes

    return {
        "schema_version": "crypto_ai_swing_frame_quality_v2",
        "status": "HEALTHY" if healthy else "BLOCKED",
        "healthy": healthy,
        "timeframe": timeframe,
        "rows": int(len(x)),
        "start": unique_index.min().isoformat(),
        "latest_open": latest_open.isoformat(),
        "latest_close": (
            latest_close.isoformat() if latest_close is not None else None
        ),
        "age_seconds": age_seconds,
        "expected_rows": expected_rows,
        "missing_intervals": missing_intervals,
        "coverage": coverage,
        "gap_ratio": gap_ratio,
        "duplicate_count": duplicate_count,
        "duplicate_ratio": duplicate_ratio,
        "nonfinite_ratio": nonfinite_ratio,
        "ohlc_violation_ratio": ohlc_violation_ratio,
        "negative_volume_ratio": negative_volume_ratio,
        "outlier_ratio": outlier_ratio,
        "reason_codes": reason_codes,
        "thresholds": thresholds,
        "mutated_input": False,
        "forward_fill_used": False,
    }


def aggregate_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    healthy = sum(bool(row.get("healthy")) for row in rows)
    counts: dict[str, int] = {}
    for row in rows:
        for code in row.get("reason_codes") or []:
            token = str(code)
            counts[token] = counts.get(token, 0) + 1
    return {
        "schema_version": "crypto_ai_swing_data_quality_aggregate_v2",
        "total_series": total,
        "healthy_series": healthy,
        "blocked_series": total - healthy,
        "healthy_fraction": float(healthy / total) if total else 0.0,
        "all_healthy": bool(total) and healthy == total,
        "reason_counts": dict(sorted(counts.items())),
    }
