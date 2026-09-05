from __future__ import annotations

import pandas as pd

from .metrics import max_drawdown, trade_metrics, probabilistic_sharpe_ratio


def long_flat_backtest(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series | None = None,
    cost_bps_per_side: float = 25.0,
) -> dict:
    close = close.astype(float)
    entries = entries.reindex(close.index).fillna(False).astype(bool)
    exits = (
        exits.reindex(close.index).fillna(False).astype(bool)
        if exits is not None
        else pd.Series(False, index=close.index)
    )

    position = pd.Series(0.0, index=close.index)
    state = 0.0
    trades: list[float] = []
    entry_price = None

    for i, ts in enumerate(close.index):
        if state == 0 and entries.loc[ts]:
            state = 1.0
            entry_price = close.loc[ts] * (1.0 + cost_bps_per_side / 10000.0)
        elif state == 1 and exits.loc[ts]:
            exit_price = close.loc[ts] * (1.0 - cost_bps_per_side / 10000.0)
            trades.append(float(exit_price / entry_price - 1.0))
            state = 0.0
            entry_price = None
        position.iloc[i] = state

    ret = close.pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strategy_ret = position.shift(1).fillna(0.0) * ret
    strategy_ret -= turnover * (cost_bps_per_side / 10000.0)
    equity = (1.0 + strategy_ret).cumprod()

    metrics = trade_metrics(pd.Series(trades, dtype=float))
    metrics.update(
        {
            "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
            "max_drawdown": max_drawdown(equity),
            "psr": probabilistic_sharpe_ratio(strategy_ret),
            "equity_final": float(equity.iloc[-1]) if len(equity) else 1.0,
        }
    )
    return {"metrics": metrics, "equity": equity, "returns": strategy_ret}
