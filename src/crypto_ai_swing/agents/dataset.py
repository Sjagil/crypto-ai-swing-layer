from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

import numpy as np
import pandas as pd

from crypto_ai_swing.data.canonical import canonicalize_ohlcv
from crypto_ai_swing.data.features import build_features

DEFAULT_FEATURES: tuple[str, ...] = (
    "ret_1",
    "ret_4",
    "ret_8",
    "ret_24",
    "log_ret_1",
    "trend_8_20",
    "trend_20_50",
    "trend_50_200",
    "rsi_14",
    "atr_pct",
    "rv_24",
    "rv_168",
    "vol_regime",
    "bb_width",
    "bb_z",
    "breakout_20",
    "distance_low_20",
    "volume_z_48",
    "dollar_volume_log",
    "range_pct",
    "close_location",
    "downside_rv_24",
    "ewma_rv_24",
    "skew_24",
    "kurtosis_24",
    "momentum_vol_adj_8",
    "drawdown_48",
    "atr_z_168",
    "tail_q05_168",
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


def _deterministic_candidate_sample(
    frame: pd.DataFrame,
    *,
    maximum_rows: int,
) -> pd.DataFrame:
    if len(frame) <= int(maximum_rows):
        return frame
    positions = np.linspace(0, len(frame) - 1, int(maximum_rows), dtype=int)
    return frame.iloc[positions]


def build_agent_dataset(
    frames: dict[str, pd.DataFrame],
    *,
    horizon_bars: int = 4,
    minimum_net_move_bps: float = 65.0,
    feature_columns: Iterable[str] | None = DEFAULT_FEATURES,
    canonical_bridge=None,
    feature_tables_override: dict[str, pd.DataFrame] | None = None,
    maximum_candidates: int = 180,
    consume_feature_tables_override: bool = False,
) -> AgentDataset:
    horizon = int(horizon_bars)
    if horizon <= 0:
        raise ValueError("horizon_bars must be positive")

    ohlcv_tables: dict[str, pd.DataFrame] = {}
    feature_tables: dict[str, pd.DataFrame] = {}
    for market, raw in sorted(frames.items()):
        if raw is None or raw.empty:
            continue
        ohlcv = canonicalize_ohlcv(raw)
        ohlcv_tables[market] = ohlcv
        override = (feature_tables_override or {}).get(str(market).upper())
        if override is not None:
            feature_tables[market] = override.reindex(ohlcv.index).copy()
        elif canonical_bridge is None:
            feature_tables[market] = build_features(ohlcv)
        else:
            from crypto_ai_swing.agents.canonical_features import canonical_model_frame

            feature_tables[market] = canonical_model_frame(
                canonical_bridge,
                ohlcv,
                market=market,
                timeframe=str(ohlcv.attrs.get("timeframe") or raw.attrs.get("timeframe") or "1h"),
                benchmark=None,
            )

    if not feature_tables:
        raise ValueError("no causal training rows")

    if feature_columns is None:
        from crypto_ai_swing.agents.canonical_features import candidate_columns

        market_count = max(1, len(feature_tables))
        per_market_rows = max(
            2_000,
            min(8_000, 120_000 // market_count),
        )
        samples = [
            _deterministic_candidate_sample(
                table,
                maximum_rows=per_market_rows,
            )
            for _, table in sorted(feature_tables.items())
        ]
        union = pd.concat(
            samples,
            axis=0,
            sort=False,
            copy=False,
        )
        features = candidate_columns(
            union,
            minimum_coverage=0.55,
            maximum_candidates=int(maximum_candidates),
        )
        del union
        del samples
    else:
        features = tuple(str(x) for x in feature_columns)
    if not features:
        raise ValueError("no usable causal feature columns")

    chunks: list[pd.DataFrame] = []
    for market, ohlcv in ohlcv_tables.items():
        feat = feature_tables[market]
        if canonical_bridge is None:
            missing = [name for name in features if name not in feat.columns]
            if missing:
                raise ValueError(f"missing features: {missing}")
        # Build the numerical feature block in one operation.
        # Assigning hundreds of columns individually fragments the
        # DataFrame and causes large transient memory amplification.
        selected = (
            feat.reindex(columns=list(features))
            .apply(pd.to_numeric, errors="coerce")
            .astype(np.float32, copy=False)
            .copy()
        )
        future_close = ohlcv["close"].shift(-horizon)
        future_low = _future_extreme(ohlcv["low"], horizon, "min")
        future_high = _future_extreme(ohlcv["high"], horizon, "max")
        forward_return = future_close / ohlcv["close"] - 1.0
        mae = (1.0 - future_low / ohlcv["close"]).clip(lower=0.0)
        mfe = (future_high / ohlcv["close"] - 1.0).clip(lower=0.0)
        threshold = float(minimum_net_move_bps) / 10_000.0
        past_return = ohlcv["close"] / ohlcv["close"].shift(horizon) - 1.0
        net_return = forward_return - threshold
        path_quality = ((mfe - mae) / (mfe + mae + 1e-9)).clip(-1.0, 1.0)
        regime_persistence = (past_return * forward_return > 0.0) & (
            forward_return.abs() >= threshold * 0.50
        )

        label_end_time = selected.index.to_series().shift(-horizon)

        metadata = pd.DataFrame(
            {
                "target_forward_return": forward_return.astype(np.float32, copy=False),
                "target_net_return": net_return.astype(np.float32, copy=False),
                "target_mae": mae.astype(np.float32, copy=False),
                "target_mfe": mfe.astype(np.float32, copy=False),
                "target_path_quality": path_quality.astype(np.float32, copy=False),
                "target_alpha": (net_return > 0.0).astype(np.float32),
                "target_regime_persistence": regime_persistence.astype(np.float32),
                "market": str(market).upper(),
                "feature_time": selected.index,
                "label_end_time": label_end_time,
            },
            index=selected.index,
        )

        item = pd.concat(
            [selected, metadata],
            axis=1,
            copy=False,
        )
        item = item.iloc[:-horizon] if len(item) > horizon else item.iloc[0:0]
        target_columns = [
            "target_forward_return",
            "target_net_return",
            "target_mae",
            "target_mfe",
            "target_path_quality",
            "target_alpha",
            "target_regime_persistence",
            "label_end_time",
        ]
        item = item.replace([np.inf, -np.inf], np.nan).dropna(subset=target_columns)
        chunks.append(item)
    if not chunks:
        raise ValueError("no causal training rows")
    frame = pd.concat(chunks, copy=False).sort_values(["feature_time", "market"])
    if frame.empty:
        raise ValueError("dataset empty after causal filtering")
    feature_time = pd.to_datetime(frame["feature_time"], utc=True)
    label_end = pd.to_datetime(frame["label_end_time"], utc=True)
    if (label_end <= feature_time).any():
        raise ValueError("labels are not strictly future-only")
    identity = pd.util.hash_pandas_object(
        frame[
            [
                *features,
                "target_forward_return",
                "target_net_return",
                "target_mae",
                "target_mfe",
                "target_path_quality",
                "target_regime_persistence",
                "market",
                "feature_time",
                "label_end_time",
            ]
        ],
        index=False,
    ).to_numpy()
    digest = sha256(identity.tobytes()).hexdigest()
    return AgentDataset(
        frame=frame,
        feature_columns=tuple(features),
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
    """
    Chronological train/validation/test split with two layers of protection:

    1. bar-count embargo around the nominal boundaries;
    2. label-aware purge using the ACTUAL label_end_time.

    The second layer is required for mixed/gappy market calendars because
    `horizon_bars` global timestamps are not guaranteed to cover the future
    label horizon of every individual market.

    No row is allowed into train when its label can see validation, and no
    validation row is allowed when its label can see test.
    """
    train_fraction = float(train_fraction)
    validation_fraction = float(validation_fraction)

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError(
            "train_fraction + validation_fraction must leave a test partition"
        )

    frame = dataset.frame.sort_values(
        ["feature_time", "market"]
    ).copy()

    feature_time = pd.to_datetime(
        frame["feature_time"],
        utc=True,
    )
    label_end_time = pd.to_datetime(
        frame["label_end_time"],
        utc=True,
    )

    if (label_end_time <= feature_time).any():
        raise ValueError("labels are not strictly future-only")

    times = pd.Index(
        feature_time.unique()
    ).sort_values()

    if len(times) < 30:
        raise ValueError("insufficient timestamps for split")

    train_end = max(
        1,
        int(len(times) * train_fraction),
    )
    val_end = max(
        train_end + 1,
        int(
            len(times)
            * (train_fraction + validation_fraction)
        ),
    )

    # Keep the original bar-count embargo as a first defensive layer.
    purge = max(
        1,
        int(dataset.horizon_bars),
    )

    train_times = times[
        : max(1, train_end - purge)
    ]

    val_start = min(
        len(times),
        train_end + purge,
    )
    val_stop = max(
        val_start,
        val_end - purge,
    )
    val_times = times[
        val_start:val_stop
    ]

    test_start = min(
        len(times),
        val_end + purge,
    )
    test_times = times[
        test_start:
    ]

    if (
        len(train_times) == 0
        or len(val_times) == 0
        or len(test_times) == 0
    ):
        raise ValueError(
            "purged split produced empty timestamp partition"
        )

    validation_start_time = pd.Timestamp(
        val_times[0]
    )
    test_start_time = pd.Timestamp(
        test_times[0]
    )

    # CRITICAL:
    # Membership in a timestamp partition is not enough.
    # Purge using the actual future label lifetime.
    train_mask = (
        feature_time.isin(train_times)
        & (label_end_time < validation_start_time)
    )

    validation_mask = (
        feature_time.isin(val_times)
        & (label_end_time < test_start_time)
    )

    test_mask = feature_time.isin(
        test_times
    )

    train = frame.loc[
        train_mask
    ].copy()
    validation = frame.loc[
        validation_mask
    ].copy()
    test = frame.loc[
        test_mask
    ].copy()

    if min(
        len(train),
        len(validation),
        len(test),
    ) <= 0:
        raise ValueError(
            "label-aware purged split produced empty partition"
        )

    # Final hard assertions remain in place.
    # We do NOT weaken these guards.
    train_label_max = pd.Timestamp(
        pd.to_datetime(
            train["label_end_time"],
            utc=True,
        ).max()
    )
    validation_feature_min = pd.Timestamp(
        pd.to_datetime(
            validation["feature_time"],
            utc=True,
        ).min()
    )

    validation_label_max = pd.Timestamp(
        pd.to_datetime(
            validation["label_end_time"],
            utc=True,
        ).max()
    )
    test_feature_min = pd.Timestamp(
        pd.to_datetime(
            test["feature_time"],
            utc=True,
        ).min()
    )

    if train_label_max >= validation_feature_min:
        raise ValueError(
            "train labels overlap validation after label-aware purge: "
            f"train_label_max={train_label_max.isoformat()} "
            f"validation_start={validation_feature_min.isoformat()}"
        )

    if validation_label_max >= test_feature_min:
        raise ValueError(
            "validation labels overlap test after label-aware purge: "
            f"validation_label_max={validation_label_max.isoformat()} "
            f"test_start={test_feature_min.isoformat()}"
        )

    return train, validation, test
