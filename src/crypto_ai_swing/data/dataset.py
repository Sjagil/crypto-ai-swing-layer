from __future__ import annotations

import pandas as pd

from .features import build_features


def supervised_dataset(
    ohlcv: pd.DataFrame,
    horizon_bars: int = 8,
    minimum_edge_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    features = build_features(ohlcv)
    future_return = ohlcv["close"].shift(-horizon_bars) / ohlcv["close"] - 1.0
    target = (future_return * 10000.0 > minimum_edge_bps).astype(float)
    valid = features.notna().all(axis=1) & future_return.notna()
    return (
        features.loc[valid],
        target.loc[valid].astype(int),
        future_return.loc[valid],
    )


def point_in_time_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    chunks = []
    for market, frame in frames.items():
        feat = build_features(frame).copy()
        feat["market"] = market
        chunks.append(feat.reset_index())
    if not chunks:
        return pd.DataFrame()
    panel = pd.concat(chunks, ignore_index=True)
    return panel.sort_values(["timestamp", "market"]).set_index(["timestamp", "market"])
