from __future__ import annotations

import numpy as np
import pandas as pd


STABLE_SYMBOLS = {
    "USDT", "USDC", "DAI", "EURC", "FDUSD", "TUSD", "PYUSD", "USDE", "USDS"
}


def _pct_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.rank(pct=True, ascending=ascending).fillna(0.0)


def select_universe(
    native_markets: pd.DataFrame,
    cmc: pd.DataFrame | None,
    config: dict,
) -> pd.DataFrame:
    df = native_markets.copy()
    df.columns = [str(c).lower() for c in df.columns]

    quote = config.get("exchange", {}).get("quote", "EUR")
    if "quote" in df.columns:
        df = df[df["quote"].astype(str).str.upper() == quote.upper()]
    if config.get("exchange", {}).get("require_trading_status", True) and "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().isin({"trading", "active", "ok"})]

    if "market" not in df.columns:
        raise ValueError("native_markets requires market column")
    if "symbol" not in df.columns:
        df["symbol"] = df["market"].str.split("-").str[0]

    hard = config.get("hard_filters", {})
    if "quote_volume_24h" in df.columns:
        df = df[df["quote_volume_24h"] >= hard.get("minimum_24h_quote_volume_eur", 0)]
    if "spread_bps" in df.columns:
        df = df[df["spread_bps"] <= hard.get("maximum_spread_bps", np.inf)]

    if config.get("discovery", {}).get("exclude_stablecoins", True):
        df = df[~df["symbol"].str.upper().isin(STABLE_SYMBOLS)]

    if cmc is not None and not cmc.empty:
        meta = cmc.copy()
        meta.columns = [str(c).lower() for c in meta.columns]
        key = "symbol" if "symbol" in meta.columns else None
        if key:
            keep = [c for c in ("symbol", "cmc_id", "cmc_rank", "market_cap", "volume_24h") if c in meta.columns]
            df = df.merge(meta[keep], on="symbol", how="left")
            if "market_cap" in df.columns:
                df = df[
                    df["market_cap"].isna()
                    | (df["market_cap"] >= hard.get("minimum_market_cap_usd", 0))
                ]

    weights = config.get("ranking_weights", {})
    liq = _pct_rank(df.get("quote_volume_24h", pd.Series(0.0, index=df.index)), True)
    rs = _pct_rank(df.get("relative_strength", pd.Series(0.0, index=df.index)), True)
    tq = _pct_rank(df.get("trend_quality", pd.Series(0.0, index=df.index)), True)
    vq_raw = -abs(df.get("volatility_z", pd.Series(0.0, index=df.index)))
    vq = _pct_rank(vq_raw, True)
    mc = _pct_rank(df.get("market_cap", pd.Series(0.0, index=df.index)), True)
    cq = _pct_rank(df.get("context_quality", pd.Series(0.0, index=df.index)), True)

    df["universe_score"] = (
        liq * weights.get("liquidity", 0.30)
        + rs * weights.get("relative_strength", 0.25)
        + tq * weights.get("trend_quality", 0.20)
        + vq * weights.get("volatility_quality", 0.10)
        + mc * weights.get("market_cap_quality", 0.10)
        + cq * weights.get("context_quality", 0.05)
    )
    top_n = int(config.get("discovery", {}).get("top_n", 75))
    return df.sort_values("universe_score", ascending=False).head(top_n).reset_index(drop=True)
