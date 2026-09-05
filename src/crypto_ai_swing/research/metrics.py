from __future__ import annotations

import math
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return abs(float(dd.min())) if len(dd) else 0.0


def trade_metrics(trade_returns: pd.Series) -> dict:
    r = pd.Series(trade_returns, dtype=float).dropna()
    wins = r[r > 0]
    losses = r[r < 0]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    expectancy = float(r.mean()) if len(r) else 0.0
    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()) if len(r) else 0.0,
        "profit_factor": pf,
        "expectancy": expectancy,
    }


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    r = pd.Series(returns, dtype=float).dropna()
    n = len(r)
    if n < 3 or float(r.std(ddof=1)) == 0:
        return 0.0
    sr = float(r.mean() / r.std(ddof=1))
    skew = float(r.skew())
    kurt = float(r.kurtosis() + 3.0)
    denom = math.sqrt(max(1e-12, 1 - skew * sr + ((kurt - 1) / 4) * sr * sr))
    z = (sr - benchmark_sr) * math.sqrt(max(1, n - 1)) / denom
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
