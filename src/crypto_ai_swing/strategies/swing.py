from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd

from crypto_ai_swing.contracts import ModelVote, Signal, Side


def deterministic_score(row: pd.Series) -> tuple[float, str]:
    trend = (
        0.45 * float(row.get("trend_8_20", 0.0) > 0)
        + 0.35 * float(row.get("trend_20_50", 0.0) > 0)
        + 0.20 * float(row.get("trend_50_200", 0.0) > 0)
    )

    rsi = float(row.get("rsi_14", np.nan))
    pullback = 1.0 if np.isfinite(rsi) and 42 <= rsi <= 68 else 0.35
    momentum = np.tanh(max(-0.1, min(0.1, float(row.get("ret_8", 0.0)))) * 10)
    momentum = (momentum + 1.0) / 2.0
    breakout = 1.0 if float(row.get("breakout_20", -1.0)) > 0 else 0.4
    volume = min(1.0, max(0.0, 0.5 + float(row.get("volume_z_48", 0.0)) / 6.0))

    score = 0.35 * trend + 0.20 * pullback + 0.20 * momentum + 0.15 * breakout + 0.10 * volume
    family = "BREAKOUT_RETEST" if breakout > 0.8 else "TREND_PULLBACK"
    return float(np.clip(score, 0.0, 1.0)), family


def build_signal(
    market: str,
    timestamp: datetime,
    row: pd.Series,
    ml_probability: float | None = None,
    forecast_score: float | None = None,
    rl_score: float | None = None,
    minimum_entry_score: float = 0.62,
) -> Signal:
    dscore, family = deterministic_score(row)
    votes = [ModelVote("deterministic", dscore, 0.75)]
    weighted = dscore * 0.45
    weight = 0.45

    if ml_probability is not None and np.isfinite(ml_probability):
        votes.append(ModelVote("supervised_ml", float(ml_probability), 0.70))
        weighted += float(ml_probability) * 0.30
        weight += 0.30
    if forecast_score is not None and np.isfinite(forecast_score):
        votes.append(ModelVote("forecast", float(forecast_score), 0.60))
        weighted += float(forecast_score) * 0.15
        weight += 0.15
    if rl_score is not None and np.isfinite(rl_score):
        votes.append(ModelVote("rl_challenger", float(rl_score), 0.40))
        weighted += float(rl_score) * 0.10
        weight += 0.10

    score = float(np.clip(weighted / max(weight, 1e-9), 0.0, 1.0))
    side = Side.BUY if score >= minimum_entry_score else Side.HOLD

    atr_pct = max(0.005, float(row.get("atr_pct", 0.02)))
    stop_pct = float(np.clip(1.8 * atr_pct, 0.008, 0.08))
    take_profit_pct = float(np.clip(3.0 * stop_pct, 0.025, 0.24))
    trailing_stop_pct = float(np.clip(1.4 * atr_pct, 0.008, 0.08))
    expected_edge_bps = max(0.0, (score - 0.5) * 300.0)

    return Signal(
        market=market,
        timestamp=timestamp,
        side=side,
        score=score,
        confidence=float(np.mean([v.confidence for v in votes])),
        expected_edge_bps=expected_edge_bps,
        stop_pct=stop_pct,
        take_profit_pct=take_profit_pct,
        trailing_stop_pct=trailing_stop_pct,
        strategy=family,
        votes=tuple(votes),
        features={
            k: float(row[k])
            for k in ("atr_pct", "rsi_14", "ret_8", "trend_20_50", "breakout_20")
            if k in row and np.isfinite(row[k])
        },
    )
