from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping
import asyncio
import json
import os

if TYPE_CHECKING:
    from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge

try:
    from dotenv import dotenv_values
except Exception:  # pragma: no cover - python-dotenv is a project dependency
    dotenv_values = None


@dataclass(frozen=True)
class DataSourceSpec:
    source_id: str
    family: str
    env_aliases: tuple[str, ...] = ()
    keyless: bool = False
    note: str = ""


# This is an integration registry, not a second provider implementation. Every
# network call remains owned by Sjagil/crypto. Aliases mirror its validated
# settings and .env.example so this repository can prove wiring without ever
# printing credential values.
SOURCE_SPECS: tuple[DataSourceSpec, ...] = (
    DataSourceSpec("bitvavo", "exchange_market_data", keyless=True),
    DataSourceSpec("kraken", "exchange_market_data", keyless=True),
    DataSourceSpec("mexc", "exchange_market_data", keyless=True),
    DataSourceSpec(
        "coinmarketcap",
        "market_context",
        ("COINMARKETCAP_API_KEY", "CMC_API_KEY"),
    ),
    DataSourceSpec(
        "eodhd",
        "market_context",
        ("EODHD_API_KEY", "EOD_API_KEY", "EODHISTORICALDATA_API_KEY"),
    ),
    DataSourceSpec("fred", "macro", ("FRED_API_KEY",)),
    DataSourceSpec("sec", "fundamentals", ("SEC_USER_AGENT",)),
    DataSourceSpec("alternative_me", "sentiment", keyless=True),
    DataSourceSpec("defillama", "onchain", keyless=True),
    DataSourceSpec("deribit", "derivatives_context", keyless=True),
    DataSourceSpec("coingecko", "market_context", keyless=True),
    DataSourceSpec("polygon", "market_context", ("POLYGON_API_KEY",)),
    DataSourceSpec("coinglass", "derivatives_context", ("COINGLASS_API_KEY",)),
    DataSourceSpec("glassnode", "onchain", ("GLASSNODE_API_KEY",)),
    DataSourceSpec("cryptoquant", "onchain", ("CRYPTOQUANT_API_KEY",)),
    DataSourceSpec("dune", "supplementary", ("DUNE_API_KEY",)),
    DataSourceSpec("coinpaprika", "supplementary", ("COINPAPRIKA_API_KEY",), keyless=True),
    DataSourceSpec("fmp", "supplementary", ("FMP_API_KEY",)),
    DataSourceSpec("twelvedata", "supplementary", ("TWELVEDATA_API_KEY",)),
    DataSourceSpec("alphavantage", "news_sentiment", ("ALPHAVANTAGE_API_KEY",)),
    DataSourceSpec("finnhub", "supplementary", ("FINNHUB_API_KEY",)),
    DataSourceSpec("marketstack", "supplementary", ("MARKETSTACK_API_KEY",)),
    DataSourceSpec("theta_data", "supplementary", ("THETA_DATA_API_KEY",)),
    DataSourceSpec("fiscal_data", "macro", ("FISCAL_API_KEY",)),
    DataSourceSpec("currents", "news", ("CURRENT_NEWS_API_KEY",)),
    DataSourceSpec("marketaux", "news", ("MARKETAUX_API_TOKEN",)),
    DataSourceSpec("openfigi", "symbol_mapping", ("OPENFIGI_API_KEY",)),
    DataSourceSpec(
        "openexchangerates",
        "fx",
        ("OPENEXCHANGERATES_APP_ID", "OPENEXCHANGE_API_KEY"),
    ),
)

CANONICAL_CONTRACTS: Mapping[str, tuple[str, str]] = {
    "settings": ("config.settings", "Settings"),
    "data_loader": ("data.data_loader", "DataLoader"),
    "multi_source_collector": (
        "data.multi_source_runtime",
        "run_multi_source_collector",
    ),
    "collector_health": ("data.collector_health", "collector_health_report"),
    "rss": ("scrapers.rss", "collect_registered_feeds"),
    "web_intelligence": ("scrapers.intelligence", "run_intelligence_pipeline"),
    "market_intelligence": ("core.market_intelligence", "build_coin_ranking"),
    "prospective_context": ("data.prospective_context", "ProspectiveContextCollector"),
}


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def _is_set(name: str) -> bool:
    return bool(str(os.environ.get(name, "")).strip())


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > 20_000_000:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _model_fields(value: Any) -> set[str]:
    cls = type(value)
    fields = getattr(cls, "model_fields", {})
    return set(fields) if isinstance(fields, Mapping) else set()


def _public_default_sources(module: Any, attr: str) -> list[dict[str, Any]]:
    rows = []
    for item in getattr(module, attr, ()) or ():
        rows.append(
            {
                "source_id": str(
                    getattr(item, "source_id", None)
                    or getattr(item, "feed_id", None)
                    or "UNKNOWN"
                ),
                "publisher": str(getattr(item, "publisher", "") or ""),
                "categories": list(getattr(item, "categories", ()) or ()),
                "crypto_native": bool(getattr(item, "crypto_native", False)),
            }
        )
    return rows


def scan_env_duplicates(path: Path) -> dict[str, Any]:
    """Return duplicate/conflicting key names without returning any values."""

    seen: dict[str, str] = {}
    duplicates: set[str] = set()
    conflicts: set[str] = set()
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "duplicate_keys": [],
            "conflicting_duplicate_keys": [],
        }
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        lines = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        normalized_value = value.strip()
        if key in seen:
            duplicates.add(key)
            if seen[key] != normalized_value:
                conflicts.add(key)
        else:
            seen[key] = normalized_value
    return {
        "path": str(path),
        "exists": True,
        "duplicate_keys": sorted(duplicates),
        "conflicting_duplicate_keys": sorted(conflicts),
    }


class CanonicalDataSourceFabric:
    """Orchestrate the canonical Sjagil/crypto data stack without duplicating it."""

    def __init__(self, bridge: CryptoLibraryBridge, *, swing_root: Path | None = None) -> None:
        self.bridge = bridge
        self.swing_root = Path(swing_root).resolve() if swing_root else None

    def _canonical_settings(self) -> Any:
        return self.bridge.settings()

    def _contracts(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name, (module_name, attribute) in CANONICAL_CONTRACTS.items():
            try:
                module = self.bridge.import_module(module_name)
                value = getattr(module, attribute, None)
                result[name] = {
                    "module": module_name,
                    "attribute": attribute,
                    "ready": value is not None,
                    "callable": callable(value),
                }
            except Exception as exc:
                result[name] = {
                    "module": module_name,
                    "attribute": attribute,
                    "ready": False,
                    "callable": False,
                    "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                }
        return result

    def _provider_inventory(self, settings: Any) -> list[dict[str, Any]]:
        settings_module = self.bridge.import_module("config.settings")
        supported = set(getattr(settings_module, "SUPPORTED_PROVIDERS", ()) or ())
        provider_settings = getattr(settings, "providers", None)
        fields = _model_fields(provider_settings) if provider_settings is not None else set()
        rows: list[dict[str, Any]] = []
        for spec in SOURCE_SPECS:
            aliases = tuple(spec.env_aliases)
            configured_aliases = [name for name in aliases if _is_set(name)]
            field_name = {
                "sec": "sec_user_agent",
                "twelvedata": "twelve_data_api_key",
                "theta_data": "theta_data_api_key",
                "fiscal_data": "fiscal_api_key",
                "currents": "current_news_api_key",
                "marketaux": "marketaux_api_token",
                "alphavantage": "alpha_vantage_api_key",
                "openexchangerates": "open_exchange_rates_app_id",
            }.get(spec.source_id, f"{spec.source_id}_api_key")
            settings_bound = field_name in fields or spec.keyless
            rows.append(
                {
                    **asdict(spec),
                    "env_aliases": list(aliases),
                    "configured": bool(configured_aliases) or spec.keyless,
                    "configured_aliases": configured_aliases,
                    "canonical_supported_provider": spec.source_id in supported,
                    "canonical_settings_bound": settings_bound,
                }
            )
        return rows

    def _canonical_artifacts(self, settings: Any) -> dict[str, Any]:
        paths = settings.paths
        multi_root = Path(paths.output_dir) / "multi_source"
        intelligence_root = Path(paths.intelligence_dir)
        multi = _safe_json(multi_root / "status.json")
        heartbeat = _safe_json(multi_root / "heartbeat.json")
        scraper = _safe_json(intelligence_root / "scraper_status.json")
        source_status = dict(multi.get("source_status") or {})
        return {
            "multi_source_status_path": str(multi_root / "status.json"),
            "multi_source_heartbeat_path": str(multi_root / "heartbeat.json"),
            "intelligence_status_path": str(intelligence_root / "scraper_status.json"),
            "multi_source_status": multi.get("runtime_status") or multi.get("status"),
            "multi_source_mode": multi.get("mode"),
            "multi_source_observed_at": multi.get("observed_at"),
            "heartbeat_status": heartbeat.get("status"),
            "heartbeat_observed_at": heartbeat.get("observed_at"),
            "source_status": source_status,
            "intelligence_status": scraper.get("status"),
            "intelligence_observed_at": scraper.get("observed_at"),
            "intelligence_sources": list(scraper.get("sources") or ()),
        }

    def _source_catalog(self) -> dict[str, Any]:
        rss_module = self.bridge.import_module("scrapers.rss")
        intelligence_module = self.bridge.import_module("scrapers.intelligence")
        return {
            "rss": _public_default_sources(rss_module, "DEFAULT_FEEDS"),
            "web": _public_default_sources(intelligence_module, "DEFAULT_SOURCES"),
        }

    def _environment_audit(self) -> dict[str, Any]:
        canonical_env = self.bridge.root / ".env"
        swing_env = self.swing_root / ".env" if self.swing_root else None
        canonical_duplicates = scan_env_duplicates(canonical_env)
        swing_duplicates = (
            scan_env_duplicates(swing_env)
            if swing_env is not None
            else {
                "path": None,
                "exists": False,
                "duplicate_keys": [],
                "conflicting_duplicate_keys": [],
            }
        )
        canonical_conflicts: list[str] = []
        if dotenv_values is not None and canonical_env.is_file():
            values = dotenv_values(canonical_env)
            for key, value in values.items():
                if value is None or not str(value).strip() or key not in os.environ:
                    continue
                if str(os.environ.get(key)) != str(value):
                    canonical_conflicts.append(str(key))
        whitelist_assertions = {
            name: _truthy(os.environ.get(name))
            for name in (
                "BITVAVO_IP_WHITELIST_CONFIRMED",
                "EXCHANGE_IP_WHITELIST_CONFIRMED",
            )
            if name in os.environ
        }
        return {
            "canonical_env": canonical_duplicates,
            "swing_env": swing_duplicates,
            "canonical_process_conflict_keys": sorted(set(canonical_conflicts)),
            "bitvavo_whitelist_assertions": whitelist_assertions,
            "secrets_exposed": False,
        }

    def audit(self) -> dict[str, Any]:
        settings = self._canonical_settings()
        contracts = self._contracts()
        providers = self._provider_inventory(settings)
        environment = self._environment_audit()
        required_contracts = (
            "settings",
            "data_loader",
            "multi_source_collector",
            "rss",
            "web_intelligence",
        )
        contract_ready = all(contracts[name]["ready"] for name in required_contracts)
        duplicate_conflicts = list(
            environment["canonical_env"]["conflicting_duplicate_keys"]
        ) + list(environment["swing_env"]["conflicting_duplicate_keys"])
        return {
            "schema_version": "round37_canonical_data_source_fabric_v1",
            "status": "READY" if contract_ready and not duplicate_conflicts else "ACTION_REQUIRED",
            "code_ready": contract_ready,
            "crypto_root": str(self.bridge.root),
            "contracts": contracts,
            "providers": providers,
            "source_catalog": self._source_catalog(),
            "artifacts": self._canonical_artifacts(settings),
            "environment": environment,
            "blockers": (
                []
                if contract_ready and not duplicate_conflicts
                else (
                    ([] if contract_ready else ["CANONICAL_DATA_CONTRACT_INCOMPLETE"])
                    + (["CONFLICTING_DUPLICATE_ENV_KEYS"] if duplicate_conflicts else [])
                )
            ),
            "orders_generated": 0,
            "orders_submitted": 0,
            "private_exchange_requests": 0,
        }

    async def collect_public_once(
        self,
        *,
        duration_seconds: float = 20.0,
        include_intelligence: bool = True,
    ) -> dict[str, Any]:
        """Run only canonical public/read-only collectors for a bounded interval."""

        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        settings = self._canonical_settings()
        runtime = self.bridge.import_module("data.multi_source_runtime")
        runner = getattr(runtime, "run_multi_source_collector", None)
        if not callable(runner):
            raise RuntimeError("run_multi_source_collector unavailable")
        collection = await runner(settings, duration_seconds=float(duration_seconds))
        intelligence_result: dict[str, Any] | None = None
        if include_intelligence:
            intelligence = self.bridge.import_module("scrapers.intelligence")
            run = getattr(intelligence, "run_intelligence_pipeline", None)
            if not callable(run):
                raise RuntimeError("run_intelligence_pipeline unavailable")
            result = await run(settings)
            intelligence_result = {
                "status": str(getattr(result, "status", "UNKNOWN")),
                "record_count": len(getattr(result, "records", ()) or ()),
                "observed_at": str(getattr(result, "observed_at", "")),
                "output_path": str(getattr(result, "output_path", "") or "") or None,
            }
        return {
            "schema_version": "round37_canonical_data_collection_v1",
            "status": "COMPLETED",
            "duration_seconds": float(duration_seconds),
            "multi_source": {
                "runtime_status": collection.get("runtime_status"),
                "mode": collection.get("mode"),
                "source_status": collection.get("source_status"),
                "context_counts": collection.get("context_counts"),
                "observed_at": collection.get("observed_at"),
            },
            "intelligence": intelligence_result,
            "orders_generated": 0,
            "orders_submitted": 0,
            "private_exchange_requests": 0,
        }

    def collect_public_once_sync(
        self,
        *,
        duration_seconds: float = 20.0,
        include_intelligence: bool = True,
    ) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.collect_public_once(
                    duration_seconds=duration_seconds,
                    include_intelligence=include_intelligence,
                )
            )
        raise RuntimeError("sync collection cannot run inside an active asyncio loop")


def configured_source_ids(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(row.get("source_id"))
            for row in rows
            if row.get("configured") and row.get("source_id")
        )
    )


__all__ = [
    "CANONICAL_CONTRACTS",
    "SOURCE_SPECS",
    "CanonicalDataSourceFabric",
    "DataSourceSpec",
    "configured_source_ids",
    "scan_env_duplicates",
]
