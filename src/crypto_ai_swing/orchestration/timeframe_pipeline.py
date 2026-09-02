from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd

from crypto_ai_swing.data.features import build_features


@dataclass(frozen=True)
class MTFDecision:
    observed_at: str
    macro_score: float
    trend_score: float
    setup_score: float
    trigger_score: float
    execution_score: float
    mtf_score: float
    entry_blocked: bool
    blockers: tuple[str, ...]
    states: dict[str, dict[str, Any]]
    policy: dict[str, Any]

    def to_dict(self):
        return asdict(self)


def _state(tf, frame):
    if frame is None or frame.empty:
        return None
    features = build_features(frame).replace([np.inf, -np.inf], np.nan).dropna()
    if features.empty:
        return None
    row = features.iloc[-1]
    atr = max(1e-6, float(row.get("atr_pct", 0)))
    rv = max(1e-6, float(row.get("rv_24", 0)))
    trend = float(
        np.clip(
            0.6 * np.tanh(float(row.get("trend_20_50", 0)) / atr)
            + 0.4 * np.tanh(float(row.get("trend_50_200", 0)) / atr),
            -1,
            1,
        )
    )
    momentum = float(
        np.tanh(float(row.get("ret_8", 0)) / max(1e-6, rv * np.sqrt(8)))
    )
    breakout = float(np.tanh(float(row.get("breakout_20", 0)) / atr))
    rsi = float(row.get("rsi_14", 50))
    rsi_score = float(np.clip((rsi - 50) / 20, -1, 1))
    score = float(
        np.clip(
            0.45 * trend
            + 0.30 * momentum
            + 0.15 * breakout
            + 0.10 * rsi_score,
            -1,
            1,
        )
    )
    return {
        "timeframe": tf,
        "rows": len(frame),
        "score": score,
        "trend": trend,
        "momentum": momentum,
        "breakout": breakout,
        "rsi": rsi,
        "latest_bar": pd.Timestamp(features.index[-1]).isoformat(),
    }


def _weighted(states, weights):
    rows = [
        (states[key]["score"], weight)
        for key, weight in weights.items()
        if key in states
    ]
    return (
        float(
            np.clip(
                sum(value * weight for value, weight in rows)
                / sum(weight for _, weight in rows),
                -1,
                1,
            )
        )
        if rows
        else 0.0
    )


def _macro_blockers(
    states: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any] | None = None,
) -> list[str]:
    cfg = dict(policy or {})
    weekly_strong_bearish = float(
        cfg.get("weekly_strong_bearish_score", -0.50)
    )
    daily_confirmation = float(
        cfg.get("daily_bullish_confirmation_score", 0.45)
    )
    daily_strong_bearish = float(
        cfg.get("daily_strong_bearish_score", -0.50)
    )
    weekly_confirmation = float(
        cfg.get("weekly_bullish_confirmation_score", 0.45)
    )

    blockers: list[str] = []
    daily = states.get("1d")
    weekly = states.get("1w")
    if daily and daily["score"] < 0 and (
        not weekly or weekly["score"] <= 0
    ):
        blockers.append("MTF_MACRO_LONG_PERMISSION_DENIED")

    if (
        weekly
        and daily
        and weekly["score"] <= weekly_strong_bearish
        and daily["score"] < daily_confirmation
    ):
        blockers.append("MTF_MACRO_CONFLICT_WEEKLY_BEARISH")

    if (
        daily
        and daily["score"] <= daily_strong_bearish
        and (not weekly or weekly["score"] < weekly_confirmation)
    ):
        blockers.append("MTF_MACRO_DAILY_STRONGLY_BEARISH")

    return list(dict.fromkeys(blockers))


def evaluate_timeframe_pipeline(
    frames: Mapping[str, pd.DataFrame],
    micro: Mapping[str, Any],
    *,
    observed_at: datetime | pd.Timestamp,
    maximum_spread_bps: float = 35.0,
    policy: Mapping[str, Any] | None = None,
) -> MTFDecision:
    aliases = {"1W": "1w"}
    states = {}
    for raw, frame in frames.items():
        tf = aliases.get(str(raw), str(raw))
        row = _state(tf, frame)
        if row:
            states[tf] = row

    blockers = []
    for tf in ("1d", "4h", "1h", "15m"):
        if tf not in states:
            blockers.append(f"MTF_REQUIRED_FRAME_MISSING:{tf}")

    macro = _weighted(states, {"1d": 0.70, "1w": 0.30})
    trend = _weighted(states, {"4h": 0.62, "2h": 0.38})
    setup = states.get("1h", {}).get("score", -1.0)
    trigger = states.get("15m", {}).get("score", -1.0)

    blockers.extend(_macro_blockers(states, policy))
    permission_policy = dict(policy or {})
    minimum_macro = float(
        permission_policy.get("minimum_macro_score_for_long", 0.0)
    )
    minimum_trend = float(
        permission_policy.get("minimum_trend_score_for_long", 0.0)
    )
    if macro < minimum_macro:
        blockers.append("MTF_MACRO_AGGREGATE_NEGATIVE")

    h4 = states.get("4h")
    h2 = states.get("2h")
    if h4 and h4["score"] < 0 and (not h2 or h2["score"] <= 0):
        blockers.append("MTF_TREND_LONG_PERMISSION_DENIED")
    if trend < minimum_trend:
        blockers.append("MTF_TREND_AGGREGATE_NEGATIVE")
    if "1h" in states and setup < 0:
        blockers.append("MTF_SETUP_NOT_BULLISH")
    if "15m" in states and trigger < -0.35:
        blockers.append("MTF_TRIGGER_STRONGLY_ADVERSE")

    spread = float(micro.get("spread_bps", 999) or 999)
    book = float(
        np.clip(float(micro.get("book_imbalance", 0) or 0), -1, 1)
    )
    cvd = float(
        np.clip(
            float(
                micro.get(
                    "cvd_notional_ratio",
                    micro.get("cvd_ratio", 0),
                )
                or 0
            ),
            -1,
            1,
        )
    )
    execution = float(
        np.clip(
            0.55 * (1 - spread / max(1, maximum_spread_bps))
            + 0.25 * book
            + 0.20 * cvd,
            -1,
            1,
        )
    )
    if not np.isfinite(spread) or spread > maximum_spread_bps:
        blockers.append("MTF_EXECUTION_SPREAD_TOO_WIDE")

    mtf = float(
        np.clip(
            0.30 * macro
            + 0.27 * trend
            + 0.23 * setup
            + 0.12 * trigger
            + 0.08 * execution,
            -1,
            1,
        )
    )
    ts = pd.Timestamp(observed_at)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return MTFDecision(
        ts.isoformat(),
        macro,
        trend,
        setup,
        trigger,
        execution,
        mtf,
        bool(blockers),
        tuple(dict.fromkeys(blockers)),
        states,
        dict(policy or {}),
    )
