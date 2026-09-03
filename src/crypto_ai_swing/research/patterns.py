from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crypto_ai_swing.data.canonical import canonicalize_ohlcv
from crypto_ai_swing.data.features import build_features


def latest_pattern_snapshot(frame: pd.DataFrame) -> dict[str, Any]:
    """Describe causal price/volume structure on the latest closed history."""

    x = canonicalize_ohlcv(frame)
    if len(x) < 60:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "rows": int(len(x)),
            "minimum_rows": 60,
            "patterns": {},
            "authority": "RESEARCH_ONLY",
        }

    features = build_features(x)
    latest = features.iloc[-1]
    candle = x.iloc[-1]
    previous = x.iloc[-2]

    open_price = float(candle["open"])
    high = float(candle["high"])
    low = float(candle["low"])
    close = float(candle["close"])
    candle_range = max(high - low, np.finfo(float).eps)
    body_fraction = abs(close - open_price) / candle_range
    upper_wick_fraction = (
        high - max(open_price, close)
    ) / candle_range
    lower_wick_fraction = (
        min(open_price, close) - low
    ) / candle_range

    ema20 = float(latest.get("ema_20", np.nan))
    ema50 = float(latest.get("ema_50", np.nan))
    ema200 = float(latest.get("ema_200", np.nan))
    breakout20 = float(latest.get("breakout_20", np.nan))
    vol_regime = float(latest.get("vol_regime", np.nan))
    volume_z = float(latest.get("volume_z_48", np.nan))
    close_location = float(latest.get("close_location", np.nan))
    ret4 = float(latest.get("ret_4", np.nan))
    ret24 = float(latest.get("ret_24", np.nan))

    bullish_structure = bool(
        np.isfinite(ema20)
        and np.isfinite(ema50)
        and close > ema20 > ema50
    )
    long_term_bullish = bool(
        np.isfinite(ema50)
        and np.isfinite(ema200)
        and close > ema50 > ema200
    )
    prior_high = (
        x["high"].rolling(20).max().shift(1)
    )
    prior_breakout = bool(
        len(x) >= 22
        and float(previous["close"]) > float(prior_high.iloc[-2])
    )

    patterns = {
        "bullish_structure": bullish_structure,
        "long_term_bullish": long_term_bullish,
        "breakout_20": bool(
            np.isfinite(breakout20) and breakout20 > 0.0
        ),
        "near_breakout_20": bool(
            np.isfinite(breakout20)
            and -0.005 <= breakout20 <= 0.0
        ),
        "pullback_to_ema20": bool(
            bullish_structure
            and np.isfinite(ema20)
            and abs(close / ema20 - 1.0) <= 0.0125
        ),
        "volatility_compression": bool(
            np.isfinite(vol_regime) and vol_regime <= 0.75
        ),
        "volatility_expansion": bool(
            np.isfinite(vol_regime) and vol_regime >= 1.35
        ),
        "volume_expansion": bool(
            np.isfinite(volume_z) and volume_z >= 1.0
        ),
        "bullish_rejection": bool(
            lower_wick_fraction >= 0.45
            and close_location >= 0.65
            and close >= open_price
        ),
        "bearish_rejection": bool(
            upper_wick_fraction >= 0.45
            and close_location <= 0.35
            and close <= open_price
        ),
        "momentum_continuation": bool(
            bullish_structure
            and np.isfinite(ret4)
            and np.isfinite(ret24)
            and ret4 > 0
            and ret24 > 0
        ),
        "failed_breakout": bool(
            prior_breakout and close < float(previous["close"])
        ),
    }

    constructive_keys = (
        "bullish_structure",
        "long_term_bullish",
        "breakout_20",
        "near_breakout_20",
        "pullback_to_ema20",
        "volatility_compression",
        "volume_expansion",
        "bullish_rejection",
        "momentum_continuation",
    )
    adverse_keys = ("bearish_rejection", "failed_breakout")

    return {
        "schema_version": "crypto_ai_swing_pattern_snapshot_v1",
        "status": "READY",
        "observed_at": x.index[-1].isoformat(),
        "rows": int(len(x)),
        "patterns": patterns,
        "constructive_pattern_count": sum(
            int(patterns[key]) for key in constructive_keys
        ),
        "adverse_pattern_count": sum(
            int(patterns[key]) for key in adverse_keys
        ),
        "diagnostics": {
            "body_fraction": body_fraction,
            "upper_wick_fraction": upper_wick_fraction,
            "lower_wick_fraction": lower_wick_fraction,
            "breakout_20": breakout20,
            "vol_regime": vol_regime,
            "volume_z_48": volume_z,
            "ret_4": ret4,
            "ret_24": ret24,
            "close_location": close_location,
        },
        "causality": {
            "closed_history_only": True,
            "future_features_used": False,
            "pattern_names_are_descriptive_only": True,
        },
        "authority": "RESEARCH_ONLY",
        "live_decision_influence": False,
    }
