from __future__ import annotations

from datetime import UTC, datetime

from crypto_ai_swing.data.canonical_history import (
    RESIDUAL_MOMENTUM_MARKETS,
    CanonicalHistoryManager,
)


def test_residual_momentum_profile_is_deep_daily_only():
    profile = CanonicalHistoryManager.profile(
        "residual-momentum",
        now=datetime(2026, 9, 2, tzinfo=UTC),
    )
    assert profile.markets == RESIDUAL_MOMENTUM_MARKETS
    assert profile.timeframes == ("1d",)
    assert profile.start_by_timeframe["1d"].year == 2019


def test_swing_core_profile_matches_native_history_horizons():
    profile = CanonicalHistoryManager.profile(
        "swing-core",
        now=datetime(2026, 9, 2, tzinfo=UTC),
    )
    assert profile.timeframes == ("15m", "1h", "2h", "4h", "1d")
    assert profile.start_by_timeframe["15m"].year <= 2024
    assert profile.start_by_timeframe["4h"].year <= 2022
    assert profile.start_by_timeframe["1d"].year == 2018
