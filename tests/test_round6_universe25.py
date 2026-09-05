from __future__ import annotations

from types import SimpleNamespace

from crypto_ai_swing.universe.runtime import UniverseManager


class FakeExchange:
    def load_markets(self):
        rows = {}
        for i in range(32):
            base = f"A{i:02d}"
            symbol = f"{base}/EUR"
            rows[symbol] = {"symbol": symbol, "active": True, "spot": True}
        rows["USDT/EUR"] = {"symbol": "USDT/EUR", "active": True, "spot": True}
        return rows

    def fetch_tickers(self):
        rows = {}
        for i in range(32):
            base = f"A{i:02d}"
            rows[f"{base}/EUR"] = {
                "quoteVolume": 10_000_000 - i * 100_000,
                "bid": 100,
                "ask": 100.01,
            }
        rows["USDT/EUR"] = {"quoteVolume": 999_000_000, "bid": 1, "ask": 1.0001}
        return rows

    def close(self):
        pass


class NativeSettings:
    shariah = SimpleNamespace(
        eligibility=lambda market: SimpleNamespace(status=SimpleNamespace(value="ALLOWED"))
    )


def test_runtime_universe_selects_exactly_25_and_excludes_stablecoin(tmp_path):
    settings = SimpleNamespace(
        project_root=tmp_path,
        crypto_repo_root=tmp_path,
        universe={
            "runtime_selection": {
                "size": 25,
                "quote": "EUR",
                "refresh_seconds": 21600,
                "minimum_24h_quote_volume_eur": 250000,
                "maximum_spread_bps": 80,
                "core_markets": [],
            }
        },
    )
    manager = UniverseManager(
        settings,
        exchange_factory=FakeExchange,
        crypto_settings=NativeSettings(),
    )
    snapshot = manager.refresh().to_dict()
    assert snapshot["selected_size"] == 25
    assert len(snapshot["markets"]) == 25
    assert "USDT-EUR" not in snapshot["markets"]
