from __future__ import annotations

import numpy as np
import pandas as pd

from .canonical import canonicalize_ohlcv


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()

    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Mathematically valid edge cases:
    # no losses + positive gains -> RSI 100
    # no gains + positive losses -> RSI 0
    # no gains and no losses -> neutral RSI 50
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((gain == 0) & (loss > 0), 0.0)
    rsi = rsi.mask((gain == 0) & (loss == 0), 50.0)

    return rsi


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = canonicalize_ohlcv(df)
    out = pd.DataFrame(index=x.index)

    ret1 = x["close"].pct_change()
    out["ret_1"] = ret1
    out["ret_4"] = x["close"].pct_change(4)
    out["ret_8"] = x["close"].pct_change(8)
    out["ret_24"] = x["close"].pct_change(24)
    out["log_ret_1"] = np.log(x["close"]).diff()

    out["ema_8"] = x["close"].ewm(span=8, adjust=False).mean()
    out["ema_20"] = x["close"].ewm(span=20, adjust=False).mean()
    out["ema_50"] = x["close"].ewm(span=50, adjust=False).mean()
    out["ema_200"] = x["close"].ewm(span=200, adjust=False).mean()

    out["trend_8_20"] = out["ema_8"] / out["ema_20"] - 1.0
    out["trend_20_50"] = out["ema_20"] / out["ema_50"] - 1.0
    out["trend_50_200"] = out["ema_50"] / out["ema_200"] - 1.0

    out["rsi_14"] = _rsi(x["close"], 14)
    out["atr_14"] = _atr(x, 14)
    out["atr_pct"] = out["atr_14"] / x["close"]

    ema12 = x["close"].ewm(span=12, adjust=False).mean()
    ema26 = x["close"].ewm(span=26, adjust=False).mean()
    out["macd_line"] = ema12 - ema26
    out["macd_signal"] = out["macd_line"].ewm(
        span=9, adjust=False
    ).mean()
    out["macd_hist"] = out["macd_line"] - out["macd_signal"]
    out["macd_hist_atr"] = (
        out["macd_hist"] / out["atr_14"].replace(0.0, np.nan)
    )
    out["ema20_slope_5"] = out["ema_20"] / out["ema_20"].shift(5) - 1.0
    out["roc_12"] = x["close"].pct_change(12)

    up_move = x["high"].diff()
    down_move = -x["low"].diff()
    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0), 0.0
    )
    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0), 0.0
    )
    plus_smoothed = plus_dm.ewm(alpha=1 / 14, adjust=False).mean()
    minus_smoothed = minus_dm.ewm(alpha=1 / 14, adjust=False).mean()
    atr_safe = out["atr_14"].replace(0.0, np.nan)
    out["plus_di_14"] = 100.0 * plus_smoothed / atr_safe
    out["minus_di_14"] = 100.0 * minus_smoothed / atr_safe
    di_sum = (out["plus_di_14"] + out["minus_di_14"]).replace(
        0.0, np.nan
    )
    dx = (
        100.0
        * (out["plus_di_14"] - out["minus_di_14"]).abs()
        / di_sum
    )
    out["adx_14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    out["rv_24"] = ret1.rolling(24).std(ddof=0)
    out["rv_168"] = ret1.rolling(168).std(ddof=0)
    out["vol_regime"] = out["rv_24"] / out["rv_168"].replace(0.0, np.nan)
    downside = ret1.clip(upper=0.0)
    out["downside_rv_24"] = np.sqrt(downside.pow(2).rolling(24).mean())
    out["ewma_rv_24"] = ret1.ewm(span=24, adjust=False, min_periods=24).std(bias=True)
    out["skew_24"] = ret1.rolling(24).skew()
    out["kurtosis_24"] = ret1.rolling(24).kurt()
    out["momentum_vol_adj_8"] = out["ret_8"] / (out["rv_24"] * np.sqrt(8.0)).replace(0.0, np.nan)
    peak48 = x["close"].rolling(48, min_periods=2).max()
    out["drawdown_48"] = x["close"] / peak48 - 1.0
    atr_mean = out["atr_pct"].rolling(168).mean(); atr_std = out["atr_pct"].rolling(168).std(ddof=0)
    out["atr_z_168"] = (out["atr_pct"] - atr_mean) / atr_std.replace(0.0, np.nan)
    out["tail_q05_168"] = ret1.rolling(168).quantile(0.05)

    mid = x["close"].rolling(20).mean()
    std = x["close"].rolling(20).std(ddof=0)
    out["bb_width"] = (4 * std) / mid.replace(0.0, np.nan)
    out["bb_z"] = (x["close"] - mid) / std.replace(0.0, np.nan)

    high20 = x["high"].rolling(20).max().shift(1)
    low20 = x["low"].rolling(20).min().shift(1)
    out["breakout_20"] = x["close"] / high20 - 1.0
    out["distance_low_20"] = x["close"] / low20 - 1.0
    high55 = x["high"].rolling(55).max().shift(1)
    low55 = x["low"].rolling(55).min().shift(1)
    out["breakout_55"] = x["close"] / high55 - 1.0
    out["distance_low_55"] = x["close"] / low55 - 1.0

    vol_mean = x["volume"].rolling(48).mean()
    vol_std = x["volume"].rolling(48).std(ddof=0)
    out["volume_z_48"] = (
        (x["volume"] - vol_mean) / vol_std.replace(0.0, np.nan)
    )
    # Constant volume has zero deviation from its rolling mean.
    out["volume_z_48"] = out["volume_z_48"].mask(
        vol_std.eq(0) & vol_mean.notna(),
        0.0,
    )
    out["dollar_volume"] = x["close"] * x["volume"]
    out["dollar_volume_log"] = np.log1p(out["dollar_volume"])

    volume_mean20 = x["volume"].rolling(20).mean()
    out["volume_ratio_20"] = (
        x["volume"] / volume_mean20.replace(0.0, np.nan)
    )
    signed_volume = np.sign(x["close"].diff()).fillna(0.0) * x["volume"]
    obv = signed_volume.cumsum()
    out["obv_slope_20"] = (
        (obv - obv.shift(20))
        / x["volume"].rolling(20).sum().replace(0.0, np.nan)
    )
    money_flow_multiplier = (
        (2.0 * x["close"] - x["high"] - x["low"])
        / (x["high"] - x["low"]).replace(0.0, np.nan)
    )
    out["cmf_20"] = (
        (money_flow_multiplier * x["volume"]).rolling(20).sum()
        / x["volume"].rolling(20).sum().replace(0.0, np.nan)
    )
    out["bb_squeeze_ratio"] = (
        out["bb_width"]
        / out["bb_width"].rolling(120).median().replace(0.0, np.nan)
    )
    out["atr_expansion"] = (
        out["atr_pct"]
        / out["atr_pct"].rolling(120).median().replace(0.0, np.nan)
    )

    out["range_pct"] = (x["high"] - x["low"]) / x["close"]
    out["close_location"] = (
        (x["close"] - x["low"]) / (x["high"] - x["low"]).replace(0.0, np.nan)
    )

    out["price"] = x["close"]
    out["volume"] = x["volume"]
    return out.replace([np.inf, -np.inf], np.nan)
