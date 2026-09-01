from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

import numpy as np
import pandas as pd

from crypto_ai_swing.data.canonical import canonicalize_ohlcv
from crypto_ai_swing.data.features import build_features

DEFAULT_FEATURES: tuple[str, ...] = (
    "ret_1", "ret_4", "ret_8", "ret_24", "log_ret_1",
    "trend_8_20", "trend_20_50", "trend_50_200", "rsi_14", "atr_pct",
    "rv_24", "rv_168", "vol_regime", "bb_width", "bb_z", "breakout_20",
    "distance_low_20", "volume_z_48", "dollar_volume_log", "range_pct",
    "close_location",
)

@dataclass(frozen=True)
class AgentDataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    horizon_bars: int
    dataset_id: str
    time_start: str
    time_end: str
    markets: tuple[str, ...]

def _future_extreme(series: pd.Series, horizon: int, mode: str) -> pd.Series:
    table = pd.concat([series.shift(-step) for step in range(1, horizon + 1)], axis=1)
    return table.min(axis=1, skipna=False) if mode == "min" else table.max(axis=1, skipna=False)

def build_agent_dataset(
    frames: dict[str, pd.DataFrame],
    *,
    horizon_bars: int = 4,
    minimum_net_move_bps: float = 65.0,
    feature_columns: Iterable[str] = DEFAULT_FEATURES,
) -> AgentDataset:
    horizon = int(horizon_bars)
    if horizon <= 0:
        raise ValueError("horizon_bars must be positive")
    features = tuple(str(x) for x in feature_columns)
    chunks: list[pd.DataFrame] = []
    for market, raw in sorted(frames.items()):
        if raw is None or raw.empty:
            continue
        ohlcv = canonicalize_ohlcv(raw)
        feat = build_features(ohlcv)
        missing = [name for name in features if name not in feat.columns]
        if missing:
            raise ValueError(f"missing features: {missing}")
        future_close = ohlcv["close"].shift(-horizon)
        future_low = _future_extreme(ohlcv["low"], horizon, "min")
        future_high = _future_extreme(ohlcv["high"], horizon, "max")
        forward_return = future_close / ohlcv["close"] - 1.0
        mae = (1.0 - future_low / ohlcv["close"]).clip(lower=0.0)
        mfe = (future_high / ohlcv["close"] - 1.0).clip(lower=0.0)
        threshold = float(minimum_net_move_bps) / 10_000.0

        item = feat.loc[:, features].copy()
        item["target_forward_return"] = forward_return
        item["target_mae"] = mae
        item["target_mfe"] = mfe
        item["target_alpha"] = (forward_return > threshold).astype(float)
        item["target_regime_persistence"] = ((forward_return > 0) & (mfe > mae)).astype(float)
        item["market"] = str(market).upper()
        item["feature_time"] = item.index
        item["label_end_time"] = item.index.to_series().shift(-horizon)
        item = item.iloc[:-horizon] if len(item) > horizon else item.iloc[0:0]
        item = item.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[*features, "target_forward_return", "target_mae", "target_mfe",
                    "target_alpha", "target_regime_persistence", "label_end_time"]
        )
        chunks.append(item)
    if not chunks:
        raise ValueError("no causal training rows")
    frame = pd.concat(chunks).sort_values(["feature_time", "market"])
    if frame.empty:
        raise ValueError("dataset empty after causal filtering")
    feature_time = pd.to_datetime(frame["feature_time"], utc=True)
    label_end = pd.to_datetime(frame["label_end_time"], utc=True)
    if (label_end <= feature_time).any():
        raise ValueError("labels are not strictly future-only")
    identity = pd.util.hash_pandas_object(
        frame[[*features, "target_forward_return", "target_mae",
               "target_regime_persistence", "market", "feature_time", "label_end_time"]],
        index=False,
    ).to_numpy()
    digest = sha256(identity.tobytes()).hexdigest()
    return AgentDataset(
        frame=frame,
        feature_columns=features,
        horizon_bars=horizon,
        dataset_id=f"swing_agent_dataset_{digest}",
        time_start=pd.Timestamp(feature_time.min()).isoformat(),
        time_end=pd.Timestamp(feature_time.max()).isoformat(),
        markets=tuple(sorted(frame["market"].unique())),
    )

def purged_chronological_split(
    dataset: AgentDataset,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = dataset.frame.sort_values(["feature_time", "market"])
    times = pd.Index(pd.to_datetime(frame["feature_time"], utc=True).unique()).sort_values()
    if len(times) < 30:
        raise ValueError("insufficient timestamps for split")
    train_end = max(1, int(len(times) * train_fraction))
    val_end = max(train_end + 1, int(len(times) * (train_fraction + validation_fraction)))
    purge = max(1, dataset.horizon_bars)
    train_times = times[: max(1, train_end - purge)]
    val_start = min(len(times), train_end + purge)
    val_times = times[val_start : max(val_start, val_end - purge)]
    test_start = min(len(times), val_end + purge)
    test_times = times[test_start:]
    ft = pd.to_datetime(frame["feature_time"], utc=True)
    train = frame[ft.isin(train_times)].copy()
    val = frame[ft.isin(val_times)].copy()
    test = frame[ft.isin(test_times)].copy()
    if min(len(train), len(val), len(test)) <= 0:
        raise ValueError("purged split produced empty partition")
    if pd.Timestamp(train["label_end_time"].max()) >= pd.Timestamp(val["feature_time"].min()):
        raise ValueError("train labels overlap validation")
    if pd.Timestamp(val["label_end_time"].max()) >= pd.Timestamp(test["feature_time"].min()):
        raise ValueError("validation labels overlap test")
    return train, val, test
