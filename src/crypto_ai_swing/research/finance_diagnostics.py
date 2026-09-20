from __future__ import annotations

"""Financial time-series diagnostics used by the continuous research node.

The diagnostics intentionally remain evidence/diagnostic inputs rather than hard
assumptions.  They implement the ADF and Ljung-Box checks emphasized in the
sequence-modeling chapters of *Machine Learning in Finance*.
"""

from typing import Any, Mapping

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge


def _returns(frame: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    close = close.where(close > 0).dropna()
    values = np.log(close).diff().replace([np.inf, -np.inf], np.nan).dropna()
    return values.astype(float)


def diagnose_frame(frame: pd.DataFrame, *, maximum_rows: int = 20000) -> dict[str, Any]:
    r = _returns(frame)
    if len(r) > maximum_rows:
        r = r.iloc[-maximum_rows:]
    if len(r) < 100:
        return {"status": "INSUFFICIENT_ROWS", "rows": int(len(r))}

    adf_stat, adf_p, used_lag, nobs, critical, _ = adfuller(
        r.to_numpy(),
        autolag="AIC",
    )
    lags = sorted({1, 4, 8, 12, min(24, max(1, len(r) // 20))})
    lags = [lag for lag in lags if lag < len(r) - 5]
    lb = acorr_ljungbox(r.to_numpy(), lags=lags, return_df=True)
    lb_sq = acorr_ljungbox(np.square(r.to_numpy()), lags=lags, return_df=True)

    return {
        "status": "READY",
        "rows": int(len(r)),
        "adf": {
            "statistic": float(adf_stat),
            "pvalue": float(adf_p),
            "used_lag": int(used_lag),
            "observations": int(nobs),
            "critical_values": {str(k): float(v) for k, v in critical.items()},
            "reject_unit_root_5pct": bool(adf_p < 0.05),
        },
        "ljung_box_returns": {
            str(int(index)): {
                "statistic": float(row["lb_stat"]),
                "pvalue": float(row["lb_pvalue"]),
            }
            for index, row in lb.iterrows()
        },
        "ljung_box_squared_returns": {
            str(int(index)): {
                "statistic": float(row["lb_stat"]),
                "pvalue": float(row["lb_pvalue"]),
            }
            for index, row in lb_sq.iterrows()
        },
        "return_autocorrelation_1": float(r.autocorr(1)),
        "squared_return_autocorrelation_1": float(r.pow(2).autocorr(1)),
        "volatility": float(r.std(ddof=1)),
        "mean_return": float(r.mean()),
    }


def diagnose_markets(
    bridge: CryptoLibraryBridge,
    markets: list[str],
    *,
    timeframe: str = "1h",
    maximum_rows: int = 20000,
) -> dict[str, Any]:
    frames = bridge.historical_ohlcv_many(
        markets,
        timeframe,
        provider="bitvavo",
    )
    results: dict[str, Any] = {}
    for market in markets:
        frame = frames.get(market)
        if frame is None or frame.empty:
            results[market] = {"status": "NO_DATA"}
            continue
        try:
            results[market] = diagnose_frame(frame, maximum_rows=maximum_rows)
        except Exception as exc:
            results[market] = {
                "status": "ERROR",
                "error": f"{type(exc).__name__}:{str(exc)[:300]}",
            }
    return {
        "schema_version": "finance_time_series_diagnostics_v1",
        "timeframe": timeframe,
        "markets": results,
        "diagnostic_only": True,
        "automatic_live_block": False,
    }
