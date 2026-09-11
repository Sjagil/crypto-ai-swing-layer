
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


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


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for name in frame.columns:
        lower = str(name).lower()
        if any(token in lower for token in EXCLUDED_TOKENS):
            continue
        series = frame[name]
        if pd.api.types.is_bool_dtype(series):
            out[str(name)] = series.astype(float)
            continue
        if pd.api.types.is_numeric_dtype(series):
            out[str(name)] = pd.to_numeric(series, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


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
    rows: list[tuple[str, float, int]] = []
    for name in frame.columns:
        values = pd.to_numeric(frame[name], errors="coerce")
        coverage = float(values.notna().mean()) if len(values) else 0.0
        unique = int(values.nunique(dropna=True))
        if coverage < minimum_coverage or unique < 3:
            continue
        preferred = int(any(token in str(name).lower() for token in PREFERRED_TOKENS))
        rows.append((str(name), coverage, preferred))
    rows.sort(key=lambda item: (-item[2], -item[1], item[0]))
    return tuple(name for name, _, _ in rows[:maximum_candidates])


def select_train_only_features(
    train: pd.DataFrame,
    candidates: Iterable[str],
    *,
    target: pd.Series | None = None,
    maximum_features: int = 96,
    minimum_coverage: float = 0.70,
    maximum_abs_correlation: float = 0.95,
) -> tuple[str, ...]:
    """Feature selection using training rows only.

    Missingness, variance and redundancy are always filtered. If a target is
    supplied, rank association is used only to prioritize the already causal
    candidates. Validation and test rows never influence selection.
    """

    scored: list[tuple[str, float, float]] = []
    target_numeric = (
        pd.to_numeric(target, errors="coerce") if target is not None else None
    )
    for name in candidates:
        if name not in train.columns:
            continue
        values = pd.to_numeric(train[name], errors="coerce")
        coverage = float(values.notna().mean()) if len(values) else 0.0
        if coverage < minimum_coverage or values.nunique(dropna=True) < 3:
            continue
        association = 0.0
        if target_numeric is not None:
            valid = values.notna() & target_numeric.notna()
            if int(valid.sum()) >= 20:
                corr = values[valid].rank().corr(target_numeric[valid].rank())
                if pd.notna(corr):
                    association = abs(float(corr))
        scored.append((str(name), association, coverage))
    scored.sort(key=lambda item: (-item[1], -item[2], item[0]))
    ordered = [name for name, _, _ in scored]
    if not ordered:
        return ()
    numeric = train[ordered].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="spearman").abs()
    selected: list[str] = []
    for name in ordered:
        redundant = False
        for other in selected:
            try:
                value = corr.loc[name, other]
            except KeyError:
                continue
            if pd.notna(value) and float(value) >= maximum_abs_correlation:
                redundant = True
                break
        if redundant:
            continue
        selected.append(name)
        if len(selected) >= int(maximum_features):
            break
    return tuple(selected)


__all__ = [
    "candidate_columns",
    "canonical_model_frame",
    "select_train_only_features",
]
