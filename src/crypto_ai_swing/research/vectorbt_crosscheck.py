from __future__ import annotations

import pandas as pd


def crosscheck(close: pd.Series, entries: pd.Series, exits: pd.Series, fee_fraction: float):
    try:
        import vectorbt as vbt
    except Exception as exc:
        raise RuntimeError("vectorbt is not available") from exc
    return vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        fees=fee_fraction,
        direction="longonly",
    )
