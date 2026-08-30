from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ("open", "high", "low", "close", "volume")


def canonicalize_ohlcv(df: pd.DataFrame, timestamp_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]

    if timestamp_col:
        timestamp_col = timestamp_col.lower()
        out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=True)
        out = out.set_index(timestamp_col)
    elif not isinstance(out.index, pd.DatetimeIndex):
        for candidate in ("timestamp", "datetime", "date", "time"):
            if candidate in out.columns:
                out[candidate] = pd.to_datetime(out[candidate], utc=True)
                out = out.set_index(candidate)
                break

    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("OHLCV requires a DatetimeIndex or timestamp column")

    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out.index.name = "timestamp"

    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    out = out.loc[:, list(dict.fromkeys([*REQUIRED, *out.columns]))]
    out = out.sort_index()

    if out.index.has_duplicates:
        raise ValueError("Duplicate OHLCV timestamps are not allowed")
    if not out.index.is_monotonic_increasing:
        raise ValueError("OHLCV timestamps must be monotonic")

    numeric = out.loc[:, REQUIRED].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("OHLCV contains missing or non-finite values")
    if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (numeric["volume"] < 0).any():
        raise ValueError("Volume cannot be negative")
    if (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Invalid high price")
    if (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Invalid low price")

    out.loc[:, REQUIRED] = numeric
    return out


def quality_report(df: pd.DataFrame, expected_frequency: str | None = None) -> dict:
    canonical = canonicalize_ohlcv(df)
    report = {
        "rows": int(len(canonical)),
        "start": canonical.index.min().isoformat() if len(canonical) else None,
        "end": canonical.index.max().isoformat() if len(canonical) else None,
        "duplicate_ratio": 0.0,
        "ohlc_violation_ratio": 0.0,
        "negative_volume_ratio": 0.0,
    }
    if expected_frequency and len(canonical) > 1:
        expected = pd.date_range(
            canonical.index.min(), canonical.index.max(), freq=expected_frequency, tz="UTC"
        )
        missing = expected.difference(canonical.index)
        report["gap_ratio"] = float(len(missing) / max(1, len(expected)))
    return report
