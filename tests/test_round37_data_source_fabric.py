from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

from crypto_ai_swing.bridge.data_source_fabric import (
    CANONICAL_CONTRACTS,
    CanonicalDataSourceFabric,
    scan_env_duplicates,
)
from crypto_ai_swing.bridge.environment import hydrate_canonical_environment


class _ProviderSettings:
    model_fields = {
        "coinmarketcap_api_key": object(),
        "eodhd_api_key": object(),
        "fred_api_key": object(),
        "sec_user_agent": object(),
        "polygon_api_key": object(),
    }


class _Bridge:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._settings = SimpleNamespace(
            providers=_ProviderSettings(),
            paths=SimpleNamespace(
                output_dir=root / "output",
                intelligence_dir=root / "data_store" / "intelligence",
            ),
        )

    def settings(self):
        return self._settings

    def import_module(self, name: str):
        if name == "config.settings":
            return SimpleNamespace(
                Settings=object,
                SUPPORTED_PROVIDERS={
                    "bitvavo",
                    "kraken",
                    "mexc",
                    "coinmarketcap",
                    "eodhd",
                    "fred",
                    "sec",
                    "polygon",
                },
            )
        if name == "scrapers.rss":
            return SimpleNamespace(
                collect_registered_feeds=lambda: None,
                DEFAULT_FEEDS=(
                    SimpleNamespace(
                        feed_id="UNIT_RSS",
                        publisher="Unit",
                        categories=("crypto_news",),
                        crypto_native=True,
                    ),
                ),
            )
        if name == "scrapers.intelligence":
            return SimpleNamespace(
                run_intelligence_pipeline=lambda settings: None,
                DEFAULT_SOURCES=(
                    SimpleNamespace(
                        source_id="UNIT_WEB",
                        publisher="Unit",
                        categories=("macro",),
                        crypto_native=False,
                    ),
                ),
            )
        module_name_to_contract = {
            module: attribute for module, attribute in CANONICAL_CONTRACTS.values()
        }
        attribute = module_name_to_contract.get(name)
        if attribute:
            return SimpleNamespace(**{attribute: lambda *args, **kwargs: None})
        raise ImportError(name)


def test_scan_env_duplicates_reports_names_only(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "API_KEY=first-secret\n"
        "OTHER=1\n"
        "API_KEY=second-secret\n"
        "OTHER=1\n",
        encoding="utf-8",
    )
    result = scan_env_duplicates(path)
    assert result["duplicate_keys"] == ["API_KEY", "OTHER"]
    assert result["conflicting_duplicate_keys"] == ["API_KEY"]
    rendered = json.dumps(result)
    assert "first-secret" not in rendered
    assert "second-secret" not in rendered


def test_canonical_environment_is_authoritative_for_engine_keys(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / ".env").write_text(
        "BITVAVO_TRADE_API_KEY=canonical-secret\n"
        "LIVE_TRADING_ALLOWED=false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BITVAVO_TRADE_API_KEY", "stale-secret")
    monkeypatch.setenv("LIVE_TRADING_ALLOWED", "true")
    monkeypatch.delenv("CRYPTO_SWING_CANONICAL_ENV_PRECEDENCE", raising=False)

    result = hydrate_canonical_environment(tmp_path)

    assert result["status"] == "LOADED"
    assert "BITVAVO_TRADE_API_KEY" in result["overridden_keys"]
    assert "LIVE_TRADING_ALLOWED" in result["overridden_keys"]
    assert "LIVE_TRADING_ALLOWED" in result["conflict_keys"]
    assert "canonical-secret" not in json.dumps(result)
    assert "stale-secret" not in json.dumps(result)
    assert __import__("os").environ["BITVAVO_TRADE_API_KEY"] == "canonical-secret"
    assert __import__("os").environ["LIVE_TRADING_ALLOWED"] == "false"


def test_fabric_audit_is_secret_safe_and_uses_canonical_contracts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("CMC_API_KEY", raising=False)
    monkeypatch.setenv("COINMARKETCAP_API_KEY", "never-print-this")
    bridge = _Bridge(tmp_path)
    fabric = CanonicalDataSourceFabric(bridge, swing_root=tmp_path)

    result = fabric.audit()

    assert result["code_ready"] is True
    assert result["status"] == "READY"
    assert result["orders_generated"] == 0
    assert result["private_exchange_requests"] == 0
    cmc = next(row for row in result["providers"] if row["source_id"] == "coinmarketcap")
    assert cmc["configured"] is True
    assert cmc["configured_aliases"] == ["COINMARKETCAP_API_KEY"]
    assert "never-print-this" not in json.dumps(result, default=str)
    assert result["source_catalog"]["rss"][0]["source_id"] == "UNIT_RSS"
    assert result["source_catalog"]["web"][0]["source_id"] == "UNIT_WEB"


def test_swing_env_loader_does_not_import_canonical_authority_keys(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crypto_ai_swing.settings import _load_swing_environment

    (tmp_path / ".env").write_text(
        "CRYPTO_REPO_PATH=../crypto\n"
        "CRYPTO_SWING_MARKETS=BTC-EUR,ETH-EUR\n"
        "LIVE_TRADING_ALLOWED=true\n"
        "BITVAVO_TRADE_API_KEY=must-not-load\n",
        encoding="utf-8",
    )
    for name in (
        "CRYPTO_REPO_PATH",
        "CRYPTO_SWING_MARKETS",
        "LIVE_TRADING_ALLOWED",
        "BITVAVO_TRADE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    _load_swing_environment(tmp_path)

    env = __import__("os").environ
    assert env["CRYPTO_REPO_PATH"] == "../crypto"
    assert env["CRYPTO_SWING_MARKETS"] == "BTC-EUR,ETH-EUR"
    assert "LIVE_TRADING_ALLOWED" not in env
    assert "BITVAVO_TRADE_API_KEY" not in env
