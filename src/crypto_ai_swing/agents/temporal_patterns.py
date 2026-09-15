from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


PATTERN_VERSION = "round47e_temporal_pattern_intelligence_v1"
PATTERN_WINDOWS = (4, 8, 16, 32, 64)


def _numeric(frame: pd.DataFrame, name: str, default: float | None = None) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _rolling_dot(series: pd.Series, weights: Iterable[float]) -> pd.Series:
    weights_arr = np.asarray(tuple(weights), dtype=float)
    scale = float(np.sum(np.abs(weights_arr)))
    if scale <= 0.0:
        raise ValueError("pattern filter weights must have non-zero L1 norm")
    weights_arr = weights_arr / scale
    return (
        pd.to_numeric(series, errors="coerce")
        .rolling(len(weights_arr), min_periods=len(weights_arr))
        .apply(lambda values: float(np.dot(values, weights_arr)), raw=True)
    )


def _bool_mean(frame: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    rows = []
    for name in names:
        if name in frame:
            rows.append(frame[name].fillna(False).astype(bool).astype(float).rename(name))
    if not rows:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.concat(rows, axis=1).mean(axis=1)


def _column_mean(frame: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    rows = []
    for name in names:
        if name in frame:
            rows.append(pd.to_numeric(frame[name], errors="coerce").rename(name))
    if not rows:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.concat(rows, axis=1).mean(axis=1)


def _family_share(frame: pd.DataFrame, *, include_tokens: Iterable[str]) -> pd.Series:
    tokens = tuple(str(x).lower() for x in include_tokens)
    columns = [
        str(name)
        for name in frame.columns
        if str(name).startswith("crypto_strategy_family_")
        and str(name).endswith("_entry_share")
        and any(token in str(name).lower() for token in tokens)
    ]
    return _column_mean(frame, columns)


def _bounded_tanh(series: pd.Series, scale: float = 1.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.tanh(values.to_numpy(dtype=float) / max(scale, 1e-12)),
        index=series.index,
        dtype=float,
    )


def augment_temporal_pattern_features(features: pd.DataFrame) -> pd.DataFrame:
    """Backward-only temporal, price-action, indicator and strategy pattern features."""

    if features is None or features.empty:
        result = features.copy()
        result.attrs["temporal_pattern_version"] = PATTERN_VERSION
        result.attrs["temporal_pattern_feature_count"] = 0
        return result

    index = features.index
    columns: dict[str, pd.Series] = {}

    close = _numeric(features, "close")
    open_ = _numeric(features, "open")
    high = _numeric(features, "high")
    low = _numeric(features, "low")
    volume = _numeric(features, "volume")

    log_close = np.log(close.where(close > 0.0))
    log_ret = log_close.diff()
    sign_ret = np.sign(log_ret)
    abs_ret = log_ret.abs()

    candle_range = (high - low).replace(0.0, np.nan)
    signed_body = (close - open_) / candle_range
    lower_wick = (np.minimum(open_, close) - low) / candle_range
    upper_wick = (high - np.maximum(open_, close)) / candle_range
    wick_imbalance = lower_wick - upper_wick
    close_location = 2.0 * (close - low) / candle_range - 1.0
    volume_change = np.log(volume.where(volume > 0.0)).diff()

    for window in PATTERN_WINDOWS:
        min_periods = max(3, window // 2)
        net_move = log_close.diff(window)
        path = abs_ret.rolling(window, min_periods=min_periods).sum()

        columns[f"pattern_return_{window}"] = close.pct_change(window, fill_method=None)
        columns[f"pattern_path_efficiency_{window}"] = net_move.abs() / path.replace(0.0, np.nan)
        columns[f"pattern_direction_balance_{window}"] = sign_ret.rolling(
            window, min_periods=min_periods
        ).mean()
        columns[f"pattern_positive_bar_share_{window}"] = (
            (log_ret > 0.0).astype(float).rolling(window, min_periods=min_periods).mean()
        )
        columns[f"pattern_realized_volatility_{window}"] = log_ret.rolling(
            window, min_periods=min_periods
        ).std(ddof=0)
        columns[f"pattern_return_autocorr1_{window}"] = log_ret.rolling(
            window, min_periods=min_periods
        ).corr(log_ret.shift(1))
        columns[f"pattern_drawdown_{window}"] = (
            close / close.rolling(window, min_periods=min_periods).max() - 1.0
        )
        columns[f"pattern_drawup_{window}"] = (
            close / close.rolling(window, min_periods=min_periods).min() - 1.0
        )

        rolling_high = high.rolling(window, min_periods=min_periods).max()
        rolling_low = low.rolling(window, min_periods=min_periods).min()
        columns[f"pattern_range_position_{window}"] = (
            (close - rolling_low) / (rolling_high - rolling_low).replace(0.0, np.nan)
        )
        prior_high = high.shift(1).rolling(window, min_periods=min_periods).max()
        prior_low = low.shift(1).rolling(window, min_periods=min_periods).min()
        columns[f"pattern_prior_breakout_distance_{window}"] = close / prior_high.replace(
            0.0, np.nan
        ) - 1.0
        columns[f"pattern_prior_breakdown_distance_{window}"] = close / prior_low.replace(
            0.0, np.nan
        ) - 1.0

        vol_mean = volume.rolling(window, min_periods=min_periods).mean()
        vol_std = volume.rolling(window, min_periods=min_periods).std(ddof=0)
        columns[f"pattern_volume_z_{window}"] = (volume - vol_mean) / vol_std.replace(
            0.0, np.nan
        )
        columns[f"pattern_price_volume_corr_{window}"] = log_ret.rolling(
            window, min_periods=min_periods
        ).corr(volume_change)
        columns[f"pattern_signed_body_mean_{window}"] = signed_body.rolling(
            window, min_periods=min_periods
        ).mean()
        columns[f"pattern_wick_imbalance_mean_{window}"] = wick_imbalance.rolling(
            window, min_periods=min_periods
        ).mean()
        columns[f"pattern_close_location_mean_{window}"] = close_location.rolling(
            window, min_periods=min_periods
        ).mean()

    vol32 = log_ret.rolling(32, min_periods=16).std(ddof=0)
    normalized_ret = log_ret / vol32.replace(0.0, np.nan)

    filters = {
        "trend_8": np.linspace(-1.0, 1.0, 8),
        "impulse_8": np.asarray([-1.0] * 7 + [7.0]),
        "reversal_8": np.asarray([-1.0] * 4 + [1.0] * 4),
        "oscillation_8": np.asarray([-1.0, 1.0] * 4),
        "trend_16": np.linspace(-1.0, 1.0, 16),
        "impulse_16": np.asarray([-1.0] * 15 + [15.0]),
        "reversal_16": np.asarray([-1.0] * 8 + [1.0] * 8),
        "oscillation_16": np.asarray([-1.0, 1.0] * 8),
        "trend_32": np.linspace(-1.0, 1.0, 32),
        "reversal_32": np.asarray([-1.0] * 16 + [1.0] * 16),
    }
    for name, weights in filters.items():
        columns[f"pattern_filter_{name}"] = _rolling_dot(normalized_ret, weights)

    columns["pattern_bullish_candle_score"] = _bool_mean(
        features,
        (
            "hammer",
            "inverted_hammer",
            "bullish_engulfing",
            "morning_star_proxy",
            "three_white_soldiers",
            "bullish_pin_bar",
            "bullish_marubozu",
            "mad_zscore_reclaim",
            "keltner_lower_reclaim",
            "vwap_reclaim",
            "anchored_vwap_reclaim",
            "supertrend_bullish_flip",
            "vortex_bullish_cross",
            "mfi_bullish_reclaim",
            "macd_bullish_cross",
        ),
    )
    columns["pattern_bearish_candle_score"] = _bool_mean(
        features,
        (
            "shooting_star",
            "hanging_man",
            "bearish_engulfing",
            "evening_star_proxy",
            "three_black_crows",
            "bearish_pin_bar",
            "bearish_marubozu",
        ),
    )
    columns["pattern_candle_balance"] = (
        columns["pattern_bullish_candle_score"] - columns["pattern_bearish_candle_score"]
    )

    columns["pattern_bullish_structure_score"] = _bool_mean(
        features,
        (
            "bullish_bos",
            "bullish_choch",
            "fractal_high_breakout",
            "bullish_fractal_bos",
            "bullish_fractal_choch",
            "bullish_fractal_sweep",
        ),
    )
    columns["pattern_bearish_structure_score"] = _bool_mean(
        features,
        (
            "bearish_bos",
            "bearish_choch",
            "fractal_low_breakdown",
            "bearish_fractal_bos",
            "bearish_fractal_choch",
            "bearish_fractal_sweep",
        ),
    )
    columns["pattern_structure_balance"] = (
        columns["pattern_bullish_structure_score"] - columns["pattern_bearish_structure_score"]
    )

    entry = _numeric(features, "crypto_strategy_entry_consensus")
    exit_ = _numeric(features, "crypto_strategy_exit_consensus")
    columns["pattern_strategy_net_bias"] = entry - exit_
    columns["pattern_strategy_decisiveness"] = (entry - exit_).abs()
    columns["pattern_strategy_conflict"] = entry * exit_

    columns["pattern_family_trend_consensus"] = _family_share(
        features, include_tokens=("trend", "continuation")
    )
    columns["pattern_family_breakout_consensus"] = _family_share(
        features, include_tokens=("breakout", "donchian", "compression", "expansion")
    )
    columns["pattern_family_reversion_consensus"] = _family_share(
        features, include_tokens=("reversion", "range")
    )
    columns["pattern_family_relative_strength_consensus"] = _family_share(
        features, include_tokens=("relative_strength", "rotation", "rs_")
    )
    columns["pattern_family_liquidity_recovery_consensus"] = _family_share(
        features, include_tokens=("liquidity", "recovery", "failed")
    )
    columns["pattern_family_momentum_volume_consensus"] = _family_share(
        features, include_tokens=("momentum", "volume")
    )

    rsi = _numeric(features, "crypto_idx_rsi_14")
    mfi = _numeric(features, "crypto_idx_mfi_14")
    aroon = _numeric(features, "crypto_idx_aroon_spread_25")
    vortex = _numeric(features, "crypto_idx_vortex_spread_14")
    macd = _numeric(features, "crypto_idx_macd_atr")
    ppo = _numeric(features, "crypto_idx_ppo")
    supertrend = _numeric(features, "crypto_idx_supertrend_direction")
    choppiness = _numeric(features, "crypto_idx_choppiness_14")
    efficiency = _numeric(features, "crypto_idx_trend_efficiency_20")
    adx = _numeric(features, "crypto_idx_adx_14")
    vwap_distance = _numeric(features, "crypto_vwap_distance_20")
    anchored_vwap_distance = _numeric(features, "crypto_anchored_vwap_distance")
    vwap_reclaim = _numeric(features, "crypto_vwap_reclaim")
    anchored_reclaim = _numeric(features, "crypto_anchored_vwap_reclaim")

    directional_components = pd.concat(
        [
            rsi.rename("rsi"),
            mfi.rename("mfi"),
            aroon.rename("aroon"),
            vortex.rename("vortex"),
            _bounded_tanh(macd, 1.0).rename("macd"),
            _bounded_tanh(ppo, 0.02).rename("ppo"),
            supertrend.rename("supertrend"),
        ],
        axis=1,
    )
    columns["pattern_index_directional_confluence"] = directional_components.mean(
        axis=1, skipna=True
    )
    columns["pattern_vwap_location_confluence"] = pd.concat(
        [
            _bounded_tanh(vwap_distance, 0.02).rename("vwap"),
            _bounded_tanh(anchored_vwap_distance, 0.03).rename("anchored"),
        ],
        axis=1,
    ).mean(axis=1, skipna=True)

    columns["pattern_breakout_regime_fit"] = (
        columns["pattern_family_breakout_consensus"]
        * pd.concat([adx, efficiency, 1.0 - choppiness], axis=1).mean(
            axis=1, skipna=True
        )
    )
    columns["pattern_reversion_regime_fit"] = (
        columns["pattern_family_reversion_consensus"]
        * pd.concat([choppiness, 1.0 - efficiency], axis=1).mean(
            axis=1, skipna=True
        )
    )
    columns["pattern_vwap_reclaim_trend_fit"] = (
        pd.concat([vwap_reclaim, anchored_reclaim], axis=1).max(
            axis=1, skipna=True
        )
        * pd.concat([adx, efficiency, supertrend.clip(lower=0.0)], axis=1).mean(
            axis=1, skipna=True
        )
    )

    columns["pattern_price_action_confluence"] = pd.concat(
        [
            columns["pattern_candle_balance"],
            columns["pattern_structure_balance"],
            columns["pattern_index_directional_confluence"],
            columns["pattern_strategy_net_bias"],
            columns["pattern_vwap_location_confluence"],
        ],
        axis=1,
    ).mean(axis=1, skipna=True)

    columns["pattern_trend_vs_reversion"] = (
        columns["pattern_family_trend_consensus"]
        + columns["pattern_family_breakout_consensus"]
        - columns["pattern_family_reversion_consensus"]
    )

    out = pd.DataFrame(columns, index=index)
    out = out.replace([np.inf, -np.inf], np.nan)
    result = pd.concat([features, out], axis=1)
    result = result.loc[:, ~result.columns.duplicated(keep="last")]
    result.attrs.update(features.attrs)
    result.attrs.update(
        {
            "temporal_pattern_version": PATTERN_VERSION,
            "temporal_pattern_feature_count": int(len(out.columns)),
            "temporal_pattern_lookahead_safe": True,
            "temporal_pattern_labels_used": False,
            "temporal_pattern_execution_authority": "NONE",
            "orders_generated": 0,
            "orders_submitted": 0,
        }
    )
    return result


def pattern_feature_names(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        str(name) for name in frame.columns if str(name).startswith("pattern_")
    )


__all__ = [
    "PATTERN_VERSION",
    "PATTERN_WINDOWS",
    "augment_temporal_pattern_features",
    "pattern_feature_names",
]
