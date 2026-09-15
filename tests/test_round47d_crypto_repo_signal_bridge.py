from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from crypto_ai_swing.agents.crypto_repo_signals import (
    PROSPECTIVE_CONTEXT_FEATURES,
    augment_canonical_model_features,
    prospective_context_feature_vector,
)


class _FakeStrategy:
    strategy_id = "fake_trend"
    family = "trend"
    uses_intelligence = False

    def generate(self, features):
        index = features.index
        close = features["close"]
        entry = close > close.rolling(3, min_periods=1).mean()
        exit_signal = close < close.rolling(3, min_periods=1).mean()
        return SimpleNamespace(
            entry=entry,
            exit=exit_signal,
            reduce=pd.Series(False, index=index),
        )


class _FakeStrategyModule:
    @staticmethod
    def strategy_registry():
        return {"fake_trend": _FakeStrategy()}


class _FakeTacticalSpec:
    family = "VWAP_RECLAIM"
    timeframe = "1h"


class _FakeTacticalModule:
    @staticmethod
    def tactical_strategy_specs():
        return (_FakeTacticalSpec(),)


class _Bridge:
    def import_module(self, name):
        if name == "research.strategies":
            return _FakeStrategyModule
        if name == "research.tactical_multitimeframe":
            return _FakeTacticalModule
        raise KeyError(name)


def _frame(rows=40):
    index = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    close = pd.Series(np.linspace(100.0, 120.0, rows), index=index)
    volume = pd.Series(np.linspace(10.0, 30.0, rows), index=index)
    raw = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    features = raw.copy()
    features["atr_14"] = 2.0
    features["vwap_20"] = close.rolling(5, min_periods=1).mean()
    features["anchored_vwap"] = (
        (close * volume).cumsum() / volume.cumsum()
    )
    features["vwap_reclaim"] = close > features["vwap_20"]
    features["anchored_vwap_reclaim"] = close > features["anchored_vwap"]
    features["rsi_14"] = np.linspace(40.0, 60.0, rows)
    features["mfi_14"] = np.linspace(35.0, 65.0, rows)
    features["williams_r_14"] = np.linspace(-80.0, -20.0, rows)
    features["cci_20"] = np.linspace(-150.0, 150.0, rows)
    features["adx_14"] = np.linspace(10.0, 40.0, rows)
    features["aroon_up_25"] = np.linspace(20.0, 80.0, rows)
    features["aroon_down_25"] = np.linspace(80.0, 20.0, rows)
    features["vortex_plus_14"] = np.linspace(0.8, 1.3, rows)
    features["vortex_minus_14"] = np.linspace(1.2, 0.7, rows)
    features["macd"] = np.linspace(-1.0, 1.0, rows)
    features["ppo"] = np.linspace(-2.0, 2.0, rows)
    features["bollinger_lower"] = close * 0.95
    features["bollinger_upper"] = close * 1.05
    features["choppiness_14"] = np.linspace(60.0, 30.0, rows)
    features["trend_efficiency_20"] = np.linspace(0.1, 0.8, rows)
    features["supertrend_direction"] = 1.0
    return raw, features


def test_round47d_vwap_indexes_and_strategy_signals_are_integrated():
    raw, features = _frame()
    out = augment_canonical_model_features(
        _Bridge(),
        raw,
        features,
        market="BTC-EUR",
        timeframe="1h",
    )

    required = {
        "crypto_vwap_distance_20",
        "crypto_anchored_vwap_distance",
        "crypto_idx_rsi_14",
        "crypto_idx_mfi_14",
        "crypto_idx_cci_20",
        "crypto_idx_macd_atr",
        "crypto_strategy_fake_trend_entry_density_12",
        "crypto_strategy_family_trend_entry_share",
        "crypto_strategy_entry_consensus",
    }
    assert required.issubset(out.columns)
    assert out.attrs["orders_submitted"] == 0
    assert out.attrs["execution_authority_changed"] is False


def test_round47d_vwap_derivatives_are_prefix_causal():
    raw, features = _frame()
    full = augment_canonical_model_features(
        _Bridge(),
        raw,
        features,
        market="BTC-EUR",
        timeframe="1h",
    )
    cut = 27
    prefix = augment_canonical_model_features(
        _Bridge(),
        raw.iloc[:cut],
        features.iloc[:cut],
        market="BTC-EUR",
        timeframe="1h",
    )
    names = [
        "crypto_vwap_distance_20",
        "crypto_anchored_vwap_distance",
        "crypto_vwap_slope_5",
        "crypto_strategy_fake_trend_entry_density_12",
        "crypto_strategy_fake_trend_entry_recency",
    ]
    for name in names:
        left = full[name].iloc[cut - 1]
        right = prefix[name].iloc[-1]
        if pd.isna(left) and pd.isna(right):
            continue
        assert np.isclose(float(left), float(right))


def test_round47d_prospective_macro_indices_and_strategy_family():
    context = {
        "macro": {
            "features": {
                "btc_dominance": 0.59,
                "eth_dominance": 0.12,
                "fear_greed": 56,
                "vix": 16.13,
                "nasdaq_100_return_5d": -0.001,
                "sp500_return_5d": -0.009,
                "stablecoin_liquidity_score": 0.7,
                "usdt_market_cap_change_1h": 0.001,
                "usdt_market_cap_change_24h": 0.002,
                "usdc_market_cap_change_1h": -0.001,
                "usdc_market_cap_change_24h": -0.002,
                "aggregate_funding_proxy": 0.00005,
                "perpetual_premium": -0.0004,
                "open_interest_proxy": 123456789.0,
                "total_crypto_market_cap": 2.3e12,
            }
        },
        "mtf_challenger": {
            "strategy_family": "VWAP_RECLAIM",
        },
    }
    row = prospective_context_feature_vector(context)
    assert set(PROSPECTIVE_CONTEXT_FEATURES).issubset(row)
    assert row["macro_btc_dominance"] == 0.59
    assert np.isclose(row["macro_fear_greed_scaled"], 0.56)
    assert row["strategy_family_vwap"] == 1.0
    assert row["strategy_family_trend"] == 0.0
