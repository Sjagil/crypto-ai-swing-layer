from types import SimpleNamespace

from crypto_ai_swing.universe.runtime import UniverseManager


class FakeExchange:
    def load_markets(self):
        rows = {}
        for i in range(34):
            symbol = f"Q{i:02d}/EUR"
            rows[symbol] = {"symbol": symbol, "active": True, "spot": True}
        return rows

    def fetch_tickers(self):
        rows = {}
        for i in range(34):
            spread_bps = 5.0 if i < 27 else 20.0 if i < 32 else 50.0
            mid = 100.0
            half = mid * spread_bps / 20_000.0
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
        pass


class NativeSettings:
    shariah = SimpleNamespace(
        eligibility=lambda market: SimpleNamespace(
            status=SimpleNamespace(value="ALLOWED")
        )
    )


def test_universe_prefers_tight_spreads_and_rejects_hype(tmp_path):
    settings = SimpleNamespace(
        project_root=tmp_path,
        crypto_repo_root=tmp_path,
        universe={
            "runtime_selection": {
                "size": 25,
                "quote": "EUR",
                "minimum_24h_quote_volume_eur": 250000,
                "preferred_maximum_spread_bps": 15,
                "maximum_spread_bps": 35,
                "core_markets": [],
            },
            "anti_hype": {"maximum_24h_return": 0.35},
        },
    )
    snapshot = UniverseManager(
        settings,
        exchange_factory=FakeExchange,
        crypto_settings=NativeSettings(),
    ).refresh().to_dict()
    assert snapshot["schema_version"] == "crypto_ai_swing_runtime_universe_v3"
    assert snapshot["selected_size"] == 25
    assert snapshot["fallback_liquidity_count"] == 0
    assert "Q26-EUR" not in snapshot["markets"]
    assert snapshot["rejection_counts"]["ANTI_HYPE_24H_RETURN"] == 1
    assert snapshot["rejection_counts"]["SPREAD_TOO_WIDE"] == 2
    assert max(float(row["spread_bps"]) for row in snapshot["candidates"]) <= 15.0
