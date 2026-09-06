from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from crypto_ai_swing.data.features import build_features


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _clip(value: float) -> float:
    return float(np.clip(float(value), -1.0, 1.0))


def technical_snapshot(
    frame: pd.DataFrame,
    *,
    timeframe: str | None = None,
) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {
            "status": "NO_DATA",
            "timeframe": timeframe,
            "authority": "ADVISORY_ONLY",
            "live_decision_influence": False,
        }

    features = build_features(frame).replace([np.inf, -np.inf], np.nan)
    if features.empty:
        return {
            "status": "NO_FEATURES",
            "timeframe": timeframe,
            "authority": "ADVISORY_ONLY",
            "live_decision_influence": False,
        }

    row = features.iloc[-1]
    atr = max(1e-8, _finite(row.get("atr_pct"), 0.01))
    trend = _clip(
        0.55 * np.tanh(_finite(row.get("trend_20_50")) / atr)
        + 0.45 * np.tanh(_finite(row.get("trend_50_200")) / atr)
    )
    momentum = _clip(
        np.tanh(_finite(row.get("momentum_vol_adj_8")) / 1.75)
    )
    breakout20 = _finite(row.get("breakout_20"))
    breakout55 = _finite(row.get("breakout_55"))
    breakout = _clip(
        0.60 * np.tanh(breakout20 / atr)
        + 0.40 * np.tanh(breakout55 / atr)
    )
    macd = _clip(np.tanh(_finite(row.get("macd_hist_atr"))))
    plus_di = _finite(row.get("plus_di_14"))
    minus_di = _finite(row.get("minus_di_14"))
    adx = _finite(row.get("adx_14"))
    adx_direction = _clip((plus_di - minus_di) / 35.0)
    adx_strength = float(np.clip((adx - 15.0) / 30.0, 0.0, 1.0))
    adx_component = _clip(adx_direction * adx_strength)
    rsi = _finite(row.get("rsi_14"), 50.0)
    rsi_component = _clip((rsi - 50.0) / 20.0)
    volume_z = _finite(row.get("volume_z_48"))
    volume_component = _clip(np.tanh(volume_z / 2.5))
    close_location = _finite(row.get("close_location"), 0.5)
    close_component = _clip(2.0 * close_location - 1.0)

    score = _clip(
        0.24 * trend
        + 0.17 * momentum
        + 0.20 * breakout
        + 0.10 * macd
        + 0.09 * adx_component
        + 0.08 * volume_component
        + 0.06 * rsi_component
        + 0.06 * close_component
    )

    squeeze_ratio = _finite(row.get("bb_squeeze_ratio"), 1.0)
    atr_expansion = _finite(row.get("atr_expansion"), 1.0)
    volume_ratio = _finite(row.get("volume_ratio_20"), 1.0)

    tags: list[str] = []
    state = "NEUTRAL"
    if (
        breakout55 > 0.0
        and trend > 0.10
        and volume_z >= 0.35
        and close_location >= 0.55
    ):
        state = "DONCHIAN_55_CONFIRMED"
        tags.append("BREAKOUT_55")
    elif (
        breakout20 > 0.0
        and trend > 0.05
        and close_location >= 0.52
    ):
        state = "DONCHIAN_20_BREAKOUT"
        tags.append("BREAKOUT_20")
    elif breakout20 >= -0.35 * atr and trend > 0.10:
        state = "BREAKOUT_WATCH"
        tags.append("NEAR_20_BAR_HIGH")
    elif squeeze_ratio < 0.75 and trend >= -0.10:
        state = "SQUEEZE_BUILDUP"
        tags.append("VOLATILITY_COMPRESSION")
    elif trend > 0.15 and 40.0 <= rsi <= 62.0:
        state = "TREND_PULLBACK"
        tags.append("PULLBACK_ZONE")
    elif trend > 0.20:
        state = "TREND_CONTINUATION"

    if adx >= 25.0:
        tags.append("ADX_TREND")
    if macd > 0.15:
        tags.append("MACD_POSITIVE")
    elif macd < -0.15:
        tags.append("MACD_NEGATIVE")
    if volume_ratio >= 1.5 or volume_z >= 1.0:
        tags.append("VOLUME_EXPANSION")
    if atr_expansion >= 1.25:
        tags.append("ATR_EXPANSION")
    if squeeze_ratio <= 0.65:
        tags.append("BB_SQUEEZE")

    try:
        latest_bar = pd.Timestamp(features.index[-1]).isoformat()
    except Exception:
        latest_bar = None

    return {
        "status": "READY",
        "timeframe": timeframe,
        "latest_bar": latest_bar,
        "technical_score": score,
        "breakout_state": state,
        "tags": list(dict.fromkeys(tags)),
        "components": {
            "trend": trend,
            "momentum": momentum,
            "breakout": breakout,
            "macd": macd,
            "adx": adx_component,
            "volume": volume_component,
            "rsi": rsi_component,
            "close_location": close_component,
        },
        "indicators": {
            "rsi_14": rsi,
            "atr_pct": atr,
            "adx_14": adx,
            "plus_di_14": plus_di,
            "minus_di_14": minus_di,
            "macd_hist_atr": _finite(row.get("macd_hist_atr")),
            "ema20_slope_5": _finite(row.get("ema20_slope_5")),
            "roc_12": _finite(row.get("roc_12")),
            "breakout_20": breakout20,
            "breakout_55": breakout55,
            "bb_squeeze_ratio": squeeze_ratio,
            "atr_expansion": atr_expansion,
            "volume_z_48": volume_z,
            "volume_ratio_20": volume_ratio,
            "obv_slope_20": _finite(row.get("obv_slope_20")),
            "cmf_20": _finite(row.get("cmf_20")),
        },
        "authority": "ADVISORY_ONLY",
        "live_decision_influence": False,
        "automatic_parameter_application": False,
    }


def multi_timeframe_snapshot(
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    aliases = {"1W": "1w"}
    states: dict[str, dict[str, Any]] = {}
    for raw, frame in frames.items():
        timeframe = aliases.get(str(raw), str(raw))
        snapshot = technical_snapshot(frame, timeframe=timeframe)
        if snapshot.get("status") == "READY":
            states[timeframe] = snapshot

    weights = {
        "15m": 0.08,
        "1h": 0.22,
        "2h": 0.15,
        "4h": 0.25,
        "1d": 0.20,
        "1w": 0.10,
    }
    weighted = [
        (
            _finite(states[timeframe].get("technical_score")),
            weight,
        )
        for timeframe, weight in weights.items()
        if timeframe in states
    ]
    aggregate = (
        _clip(
            sum(score * weight for score, weight in weighted)
            / sum(weight for _, weight in weighted)
        )
        if weighted
        else 0.0
    )
    return {
        "status": "READY" if states else "NO_DATA",
        "aggregate_score": aggregate,
        "bullish_timeframes": sum(
            _finite(state.get("technical_score")) > 0.10
            for state in states.values()
        ),
        "bearish_timeframes": sum(
            _finite(state.get("technical_score")) < -0.10
            for state in states.values()
        ),
        "breakout_timeframes": [
            f"{timeframe}:{state.get('breakout_state')}"
            for timeframe, state in states.items()
            if (
                "BREAKOUT" in str(state.get("breakout_state"))
                or "DONCHIAN" in str(state.get("breakout_state"))
                or "SQUEEZE" in str(state.get("breakout_state"))
            )
        ],
        "states": states,
        "authority": "ADVISORY_ONLY",
        "live_decision_influence": False,
    }
