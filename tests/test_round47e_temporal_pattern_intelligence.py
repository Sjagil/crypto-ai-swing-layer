from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_ai_swing.agents.temporal_patterns import (
    augment_temporal_pattern_features,
    pattern_feature_names,
)
from crypto_ai_swing.agents.crypto_repo_signals import (
    PROSPECTIVE_CONTEXT_FEATURES,
    prospective_context_feature_vector,
)


def _frame(rows=180):
    index = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    phase = np.linspace(0.0, 9.0 * np.pi, rows)
    trend = np.linspace(100.0, 130.0, rows)
    close = pd.Series(trend + 2.5 * np.sin(phase), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([open_, close], axis=1).min(axis=1) - 1.0
    volume = pd.Series(
        1000.0 + 150.0 * np.cos(phase * 0.7) + np.arange(rows),
        index=index,
    )

    frame = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
    frame["hammer"] = False
    frame["bullish_engulfing"] = close.diff().fillna(0.0) > 1.0
    frame["morning_star_proxy"] = False
    frame["shooting_star"] = False
    frame["bearish_engulfing"] = close.diff().fillna(0.0) < -1.0
    frame["bullish_bos"] = close > close.shift(1).rolling(12).max()
    frame["bearish_bos"] = close < close.shift(1).rolling(12).min()

    frame["crypto_idx_rsi_14"] = np.sin(phase) * 0.4
    frame["crypto_idx_mfi_14"] = np.cos(phase) * 0.3
    frame["crypto_idx_aroon_spread_25"] = np.sin(phase * 0.5)
    frame["crypto_idx_vortex_spread_14"] = np.cos(phase * 0.4) * 0.3
    frame["crypto_idx_macd_atr"] = np.sin(phase * 0.3)
    frame["crypto_idx_ppo"] = np.sin(phase * 0.2) * 0.01
    frame["crypto_idx_supertrend_direction"] = np.where(
        close.diff().fillna(0.0) >= 0.0, 1.0, -1.0
    )
    frame["crypto_idx_choppiness_14"] = 0.45
    frame["crypto_idx_trend_efficiency_20"] = 0.55
    frame["crypto_idx_adx_14"] = 0.35
    frame["crypto_vwap_distance_20"] = np.sin(phase) * 0.01
    frame["crypto_anchored_vwap_distance"] = np.sin(phase * 0.5) * 0.015
    frame["crypto_vwap_reclaim"] = (close.diff().fillna(0.0) > 0.0).astype(float)
    frame["crypto_anchored_vwap_reclaim"] = (
        close.diff().fillna(0.0) > 0.5
    ).astype(float)

    frame["crypto_strategy_entry_consensus"] = 0.55
    frame["crypto_strategy_exit_consensus"] = 0.15
    frame["crypto_strategy_family_trend_entry_share"] = 0.7
    frame["crypto_strategy_family_breakout_entry_share"] = 0.6
    frame["crypto_strategy_family_mean_reversion_entry_share"] = 0.2
    frame["crypto_strategy_family_relative_strength_entry_share"] = 0.5
    return frame


def test_round47e_temporal_price_action_and_strategy_patterns_exist():
    out = augment_temporal_pattern_features(_frame())
    names = set(pattern_feature_names(out))
    required = {
        "pattern_path_efficiency_8",
        "pattern_direction_balance_16",
        "pattern_return_autocorr1_32",
        "pattern_prior_breakout_distance_32",
        "pattern_filter_trend_8",
        "pattern_filter_reversal_16",
        "pattern_bullish_candle_score",
        "pattern_structure_balance",
        "pattern_strategy_net_bias",
        "pattern_family_breakout_consensus",
        "pattern_index_directional_confluence",
        "pattern_breakout_regime_fit",
        "pattern_price_action_confluence",
    }
    assert required.issubset(names)
    assert len(names) >= 70
    assert out.attrs["temporal_pattern_lookahead_safe"] is True
    assert out.attrs["orders_submitted"] == 0


def test_round47e_temporal_patterns_are_prefix_causal():
    frame = _frame()
    full = augment_temporal_pattern_features(frame)
    cut = 130
    prefix = augment_temporal_pattern_features(frame.iloc[:cut])
    names = (
        "pattern_path_efficiency_8",
        "pattern_direction_balance_16",
        "pattern_return_autocorr1_32",
        "pattern_prior_breakout_distance_32",
        "pattern_filter_trend_8",
        "pattern_filter_reversal_16",
        "pattern_bullish_candle_score",
        "pattern_strategy_net_bias",
        "pattern_price_action_confluence",
    )
    for name in names:
        left = full[name].iloc[cut - 1]
        right = prefix[name].iloc[-1]
        if pd.isna(left) and pd.isna(right):
            continue
        assert np.isclose(float(left), float(right), equal_nan=True)


def test_round47e_pattern_feature_names_carry_no_label_semantics():
    out = augment_temporal_pattern_features(_frame())
    forbidden = ("target_", "future_", "forward_", "label_", "outcome_", "pnl_")
    for name in pattern_feature_names(out):
        assert not any(token in name.lower() for token in forbidden)


def test_round47e_prospective_pattern_context_is_available():
    context = {
        "universe_screen": {
            "execution_adjusted_score": 0.8,
            "technical": {
                "breakout_state": "TREND_CONTINUATION",
                "vwap_reclaim": True,
                "anchored_vwap_reclaim": True,
                "vwap_distance": 0.005,
                "anchored_vwap_distance": 0.01,
            },
        },
        "mtf_challenger": {
            "strategy_family": "VWAP_RECLAIM",
            "alignment_score": 0.75,
            "execution_score": 0.70,
            "diagnostics": {
                "trend": 0.8,
                "setup": 0.7,
                "trigger": 0.9,
                "macro": 0.6,
                "h1_rsi": 58.0,
                "m15_rsi": 61.0,
            },
        },
    }
    row = prospective_context_feature_vector(context)
    required = {
        "context_pattern_trend",
        "context_pattern_breakout",
        "context_pattern_pullback",
        "context_pattern_reversal",
        "context_pattern_range",
        "context_pattern_vwap",
        "context_pattern_strategy_confidence",
        "context_pattern_confluence",
    }
    assert required.issubset(PROSPECTIVE_CONTEXT_FEATURES)
    assert required.issubset(row)
    for name in required:
        value = row[name]
        assert value is None or 0.0 <= float(value) <= 1.0
    assert row["context_pattern_vwap"] > 0.0
