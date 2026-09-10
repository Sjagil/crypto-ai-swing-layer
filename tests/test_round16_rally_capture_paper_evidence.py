from __future__ import annotations

from types import SimpleNamespace

from crypto_ai_swing.orchestration.rally_capture import assess_rally, macro_override_allowed


def _cfg():
    return {
        "enabled": True,
        "minimum_ret_8": 0.015,
        "minimum_ret_24": 0.04,
        "minimum_trend_component": 0.45,
        "minimum_momentum_component": 0.65,
        "minimum_rally_score": 0.65,
        "maximum_rsi_for_macro_override": 84.0,
        "paper_macro_override_enabled": True,
        "macro_override_minimum_macro_score": -0.20,
        "macro_override_minimum_trend_score": 0.30,
        "macro_override_minimum_setup_score": 0.35,
        "macro_override_minimum_trigger_score": 0.15,
        "macro_override_minimum_execution_score": -0.10,
    }


def test_post_breakout_continuation_enters_rally_lane():
    row = {
        "ret_8": 0.035,
        "ret_24": 0.135,
        "trend_component": 0.999,
        "momentum_component": 0.966,
        "breakout_component": -0.746,
        "execution_quality": 0.583,
        "rsi_14": 62.8,
    }
    result = assess_rally(row, _cfg())
    assert result.deep_scan_candidate is True
    assert result.score >= 0.65


def test_macro_override_is_paper_only_and_soft_blocker_only():
    result = assess_rally(
        {
            "ret_8": 0.03,
            "ret_24": 0.149,
            "trend_component": 0.997,
            "momentum_component": 0.977,
            "execution_quality": 0.636,
            "rsi_14": 64.5,
        },
        _cfg(),
    )
    decision = SimpleNamespace(
        blockers=(
            "MTF_MACRO_LONG_PERMISSION_DENIED",
            "MTF_MACRO_AGGREGATE_NEGATIVE",
        ),
        macro_score=-0.090,
        trend_score=0.449,
        setup_score=0.452,
        trigger_score=0.344,
        execution_score=0.327,
    )
    assert macro_override_allowed(decision, result, mode="paper", config=_cfg())
    assert not macro_override_allowed(decision, result, mode="live", config=_cfg())

    decision.blockers = (
        "MTF_MACRO_AGGREGATE_NEGATIVE",
        "MTF_EXECUTION_SPREAD_TOO_WIDE",
    )
    assert not macro_override_allowed(decision, result, mode="paper", config=_cfg())


def test_runtime_universe_observes_overextended_asset_by_default(tmp_path):
    from types import SimpleNamespace

    from crypto_ai_swing.universe.runtime import UniverseManager

    class Exchange:
        def load_markets(self):
            return {
                f"Q{i:02d}/EUR": {
                    "symbol": f"Q{i:02d}/EUR",
                    "active": True,
                    "spot": True,
                }
                for i in range(27)
            }

        def fetch_tickers(self):
            rows = {}
            for i in range(27):
                mid = 100.0
                half = mid * 5.0 / 20_000.0
                rows[f"Q{i:02d}/EUR"] = {
                    "quoteVolume": 2_000_000 - i * 10_000,
                    "bid": mid - half,
                    "ask": mid + half,
                    "open": 100.0,
                    "last": 100.0,
                }
            rows["Q26/EUR"]["last"] = 150.0
            return rows

        def close(self):
            return None

    native = SimpleNamespace(
        shariah=SimpleNamespace(
            eligibility=lambda market: SimpleNamespace(
                status=SimpleNamespace(value="ALLOWED")
            )
        )
    )
    settings = SimpleNamespace(
        project_root=tmp_path,
        crypto_repo_root=tmp_path,
        universe={
            "runtime_selection": {
                "size": 27,
                "quote": "EUR",
                "minimum_24h_quote_volume_eur": 250000,
                "preferred_maximum_spread_bps": 15,
                "maximum_spread_bps": 35,
                "core_markets": [],
            },
            "anti_hype": {
                "maximum_24h_return": 0.35,
                "hard_exclude_above_threshold": False,
            },
        },
    )
    snapshot = UniverseManager(
        settings,
        exchange_factory=Exchange,
        crypto_settings=native,
    ).refresh().to_dict()
    row = next(
        x for x in snapshot["candidates"]
        if x["market"] == "Q26-EUR"
    )
    assert row["anti_hype_flag"] is True
    assert "ANTI_HYPE_24H_RETURN" not in snapshot["rejection_counts"]
    assert snapshot["policy"]["anti_hype_hard_exclude"] is False


def test_pre_v016_cached_universe_is_invalidated(tmp_path):
    import json
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    from crypto_ai_swing.universe.runtime import UniverseManager

    state = tmp_path / "universe.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": "crypto_ai_swing_runtime_universe_v3",
                "generated_at": datetime.now(UTC).isoformat(),
                "expires_at": (
                    datetime.now(UTC) + timedelta(hours=6)
                ).isoformat(),
                "selected_size": 1,
                "markets": ["OLD-EUR"],
                "policy": {"maximum_positive_return_24h": 0.35},
            }
        ),
        encoding="utf-8",
    )

    class Exchange:
        def load_markets(self):
            return {
                "NEW/EUR": {
                    "symbol": "NEW/EUR",
                    "active": True,
                    "spot": True,
                }
            }

        def fetch_tickers(self):
            return {
                "NEW/EUR": {
                    "quoteVolume": 1_000_000,
                    "bid": 99.99,
                    "ask": 100.01,
                    "open": 100.0,
                    "last": 100.0,
                }
            }

        def close(self):
            return None

    native = SimpleNamespace(
        shariah=SimpleNamespace(
            eligibility=lambda market: SimpleNamespace(
                status=SimpleNamespace(value="ALLOWED")
            )
        )
    )
    settings = SimpleNamespace(
        project_root=tmp_path,
        crypto_repo_root=tmp_path,
        universe={
            "runtime_selection": {
                "size": 1,
                "quote": "EUR",
                "minimum_24h_quote_volume_eur": 250000,
                "preferred_maximum_spread_bps": 15,
                "maximum_spread_bps": 35,
                "core_markets": [],
                "state_path": "universe.json",
            },
            "anti_hype": {
                "maximum_24h_return": 0.35,
                "hard_exclude_above_threshold": False,
            },
        },
    )
    snapshot = UniverseManager(
        settings,
        exchange_factory=Exchange,
        crypto_settings=native,
    ).current()
    assert snapshot["markets"] == ["NEW-EUR"]
    assert snapshot["policy"]["anti_hype_hard_exclude"] is False
