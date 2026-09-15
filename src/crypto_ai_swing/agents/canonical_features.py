
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from crypto_ai_swing.agents.crypto_repo_signals import augment_canonical_model_features
from crypto_ai_swing.agents.temporal_patterns import augment_temporal_pattern_features
from crypto_ai_swing.agents.feature_denoising import (
    denoised_candidate_columns,
    select_stable_train_features,
)


# Technical-only model feature bridge. Historical L1/L2, news and macro are
# deliberately excluded because Round44 did not exist historically. Those
# sources enter through prospective PIT snapshots and the context challenger.
EXCLUDED_TOKENS = (
    "target_",
    "future_",
    "forward_",
    "label_",
    "outcome_",
    "pnl_",
)

PREFERRED_TOKENS = (
    "return",
    "relative",
    "beta",
    "alpha",
    "sma",
    "ema",
    "wma",
    "vwma",
    "dema",
    "tema",
    "hma",
    "kama",
    "supertrend",
    "donchian",
    "aroon",
    "rsi",
    "stoch",
    "macd",
    "ppo",
    "cci",
    "williams",
    "mfi",
    "vwap",
    "atr",
    "volatility",
    "bollinger",
    "keltner",
    "volume",
    "obv",
    "adl",
    "chaikin",
    "hammer",
    "doji",
    "engulf",
    "harami",
    "inside",
    "outside",
    "morning",
    "evening",
    "soldier",
    "crow",
    "pin_bar",
    "marubozu",
    "fractal",
    "hh_",
    "hl_",
    "lh_",
    "ll_",
    "bos",
    "choch",
    "breakout",
    "breakdown",
    "sweep",
    "fvg",
    "structure",
    "hurst",
    "entropy",
    "complexity",
    "drawdown",
)


PREFERRED_TOKENS = tuple(dict.fromkeys((*PREFERRED_TOKENS, "crypto_vwap_", "crypto_idx_", "crypto_strategy_", "strategy_family_")))


PREFERRED_TOKENS = tuple(dict.fromkeys((*PREFERRED_TOKENS, "pattern_", "motif_", "sequence_", "strategy_")))


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}

    for name in frame.columns:
        lower = str(name).lower()
        if any(token in lower for token in EXCLUDED_TOKENS):
            continue

        series = frame[name]

        if pd.api.types.is_bool_dtype(series):
            columns[str(name)] = series.astype(float)
            continue

        if pd.api.types.is_numeric_dtype(series):
            columns[str(name)] = pd.to_numeric(
                series,
                errors="coerce",
            )

    out = pd.DataFrame(
        columns,
        index=frame.index,
    )

    return out.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def canonical_model_frame(
    bridge: Any,
    frame: pd.DataFrame,
    *,
    market: str,
    timeframe: str | None = None,
    benchmark: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build causal canonical technical features for supervised/RL models.

    Only real closed-candle OHLCV is accepted. No macro/news/L2 backfill is
    attempted here. Benchmark data, when supplied, must be causal and aligned.
    """

    module = bridge.import_module("research.features")
    selected = frame.copy()
    tf = str(timeframe or selected.attrs.get("timeframe") or "1h")
    selected.attrs.update(
        {
            "market": str(market).upper(),
            "timeframe": "1W" if tf.lower() == "1w" else tf,
            "data_provenance": {
                "source_type": "REAL_PROVIDER_DATA",
                "synthetic_data_used": False,
                "historical_context_policy": "TECHNICAL_OHLCV_ONLY",
            },
        }
    )
    benchmark_selected = None
    if benchmark is not None and not benchmark.empty:
        benchmark_selected = benchmark.loc[benchmark.index <= selected.index.max()].copy()
        benchmark_selected.attrs.update(
            {
                "market": "BTC-EUR",
                "timeframe": "1W" if tf.lower() == "1w" else tf,
                "data_provenance": {
                    "source_type": "REAL_PROVIDER_DATA",
                    "synthetic_data_used": False,
                },
            }
        )
    features = module.FeaturePipeline(
        include_optional_garch=False,
        include_advanced_fractal_estimators=False,
    ).build(
        selected,
        market=str(market).upper(),
        benchmark=benchmark_selected,
    )
    features = augment_canonical_model_features(
        bridge,
        selected,
        features,
        market=str(market).upper(),
        timeframe=tf,
    )
    features = augment_temporal_pattern_features(features)
    out = _numeric_frame(features)
    out.attrs.update(
        {
            "canonical_feature_pipeline": True,
            "lookahead_safe": True,
            "closed_candles_only": True,
            "synthetic_data_used": False,
            "research_labels_excluded": list(
                features.attrs.get("research_labels_excluded") or []
            ),
        }
    )
    return out


def candidate_columns(
    frame: pd.DataFrame,
    *,
    minimum_coverage: float = 0.70,
    maximum_candidates: int = 180,
) -> tuple[str, ...]:
    return denoised_candidate_columns(
        frame,
        minimum_coverage=minimum_coverage,
        maximum_candidates=maximum_candidates,
    )


def select_train_only_features(
    train: pd.DataFrame,
    candidates: Iterable[str],
    *,
    target: pd.Series | None = None,
    maximum_features: int = 96,
    minimum_coverage: float = 0.70,
    maximum_abs_correlation: float = 0.95,
) -> tuple[str, ...]:
    return select_stable_train_features(
        train,
        candidates,
        target=target,
        maximum_features=maximum_features,
        minimum_coverage=minimum_coverage,
        maximum_abs_correlation=maximum_abs_correlation,
    )


__all__ = [
    "candidate_columns",
    "canonical_model_frame",
    "select_train_only_features",
]
