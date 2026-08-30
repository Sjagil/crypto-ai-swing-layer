from __future__ import annotations

import pandas as pd


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df.resample(rule, label="right", closed="right").agg(agg).dropna()
    return out


def lagged_asof_join(
    base: pd.DataFrame,
    higher_features: pd.DataFrame,
    prefix: str,
    lag_bars: int = 1,
) -> pd.DataFrame:
    """Join published higher-timeframe features without using the current unfinished bar."""
    if not isinstance(base.index, pd.DatetimeIndex) or not isinstance(
        higher_features.index, pd.DatetimeIndex
    ):
        raise ValueError("Both frames must use DatetimeIndex")
    shifted = higher_features.shift(lag_bars).copy()
    shifted = shifted.add_prefix(prefix)
    left = base.sort_index().reset_index()
    right = shifted.sort_index().reset_index()
    ts_name = left.columns[0]
    right = right.rename(columns={right.columns[0]: ts_name})
    joined = pd.merge_asof(left, right, on=ts_name, direction="backward")
    return joined.set_index(ts_name)


def assert_no_future_join(base: pd.DataFrame, higher: pd.DataFrame) -> None:
    if base.index.tz is None or higher.index.tz is None:
        raise ValueError("Timezone-aware indices are required")
    if not base.index.is_monotonic_increasing or not higher.index.is_monotonic_increasing:
        raise ValueError("Indices must be sorted")
