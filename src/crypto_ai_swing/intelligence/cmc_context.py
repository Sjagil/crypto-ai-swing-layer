from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _secret(value: Any) -> str | None:
    if value is None:
        return None
    getter = getattr(value, "get_secret_value", None)
    raw = getter() if callable(getter) else value
    text = str(raw).strip()
    return text or None


def _values(record: Any) -> dict[str, Any]:
    value = getattr(record, "values", None)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(record, Mapping):
        nested = record.get("values")
        return dict(nested) if isinstance(nested, Mapping) else dict(record)
    return {}


def _market(record: Any) -> str:
    value = getattr(record, "canonical_market", None)
    if value is None and isinstance(record, Mapping):
        value = record.get("canonical_market") or record.get("market")
    return str(value or "").upper()


class CMCContextCollector:
    # Existing Sjagil/crypto DataLoader remains canonical for CMC rankings
    # and global metrics. New context-only CMC endpoints are cached here.
    def __init__(self, settings) -> None:
        self.settings = settings
        self.crypto = CryptoLibraryBridge(settings.crypto_repo_root)
        self.root = settings.project_root / "output/crypto_ai_swing/intelligence/cmc"
        self.latest = self.root / "latest.json"
        self.cache = self.root / "cache"
        self.refresh_seconds = 300.0

    def _key(self) -> str | None:
        native = self.crypto.settings()
        providers = getattr(native, "providers", None)
        return _secret(
            getattr(providers, "coinmarketcap_api_key", None)
            if providers is not None
            else None
        )

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return dict(value) if isinstance(value, dict) else {}

    def _request(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        key_required: bool = False,
    ) -> dict[str, Any]:
        key = self._key()
        if key_required and not key:
            raise PermissionError("CMC_API_KEY_REQUIRED")

        keyless = {
            "/v3/fear-and-greed/latest",
            "/v1/altcoin-season-index/latest",
            "/v3/index/cmc100-latest",
            "/v3/index/cmc20-latest",
        }
        base = "https://pro-api.coinmarketcap.com"
        if not key and path in keyless:
            base += "/public-api"

        query = urllib.parse.urlencode(
            {k: v for k, v in dict(params or {}).items() if v is not None}
        )
        url = f"{base}{path}" + (f"?{query}" if query else "")
        headers = {
            "Accept": "application/json",
            "User-Agent": "crypto-ai-swing-layer/0.21",
        }
        if key:
            headers["X-CMC_PRO_API_KEY"] = key

        request = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("CMC_NON_OBJECT_RESPONSE")
        status = payload.get("status") or {}
        if str(status.get("error_code", 0)) not in {"0", "None"}:
            raise RuntimeError(
                f"CMC_{status.get('error_code')}:{status.get('error_message')}"
            )
        return payload

    def _cached(
        self,
        name: str,
        path: str,
        *,
        ttl: float,
        params: Mapping[str, Any] | None = None,
        key_required: bool = False,
    ) -> dict[str, Any]:
        target = self.cache / f"{name}.json"
        if target.is_file() and time.time() - target.stat().st_mtime <= ttl:
            cached = self._read(target)
            if cached:
                return cached
        try:
            raw = self._request(path, params, key_required=key_required)
            result = {
                "fetched_at": datetime.now(UTC).isoformat(),
                "payload": raw,
                "stale": False,
            }
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result
        except Exception as exc:
            cached = self._read(target) if target.is_file() else {}
            if cached:
                cached["stale"] = True
                cached["refresh_error"] = f"{type(exc).__name__}:{str(exc)[:220]}"
                return cached
            return {
                "fetched_at": None,
                "payload": {},
                "stale": False,
                "refresh_error": f"{type(exc).__name__}:{str(exc)[:220]}",
            }

    @staticmethod
    def _data(item: Mapping[str, Any]) -> Any:
        payload = item.get("payload") or {}
        return payload.get("data") if isinstance(payload, Mapping) else None

    @staticmethod
    def _breadth(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, label in (
            ("percent_change_1h", "1h"),
            ("percent_change_24h", "24h"),
            ("percent_change_7d", "7d"),
        ):
            values = [x for row in rows if (x := _num(row.get(key))) is not None]
            out[f"positive_fraction_{label}"] = (
                float(np.mean(np.asarray(values) > 0.0)) if values else None
            )
            out[f"median_change_{label}"] = (
                float(np.median(values)) if values else None
            )
        return out

    @staticmethod
    def _trend_rows(data: Any) -> list[dict[str, Any]]:
        if not isinstance(data, list):
            return []
        result = []
        for row in data[:20]:
            if not isinstance(row, Mapping):
                continue
            quote_map = row.get("quote") or {}
            quote = quote_map.get("EUR") or quote_map.get("USD") or {}
            result.append(
                {
                    "symbol": row.get("symbol"),
                    "name": row.get("name"),
                    "cmc_rank": row.get("cmc_rank"),
                    "price": quote.get("price"),
                    "market_cap": quote.get("market_cap"),
                    "volume_24h": quote.get("volume_24h"),
                    "percent_change_1h": quote.get("percent_change_1h"),
                    "percent_change_24h": quote.get("percent_change_24h"),
                    "percent_change_7d": quote.get("percent_change_7d"),
                }
            )
        return result

    def current(
        self,
        markets: list[str] | tuple[str, ...],
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        if not force and self.latest.is_file():
            if time.time() - self.latest.stat().st_mtime <= self.refresh_seconds:
                cached = self._read(self.latest)
                if cached:
                    return cached

        failures: list[str] = []
        global_metrics: dict[str, Any] = {}
        rows: list[dict[str, Any]] = []
        alternative_fear: dict[str, Any] = {}

        try:
            loader = self.crypto.data_loader()
            records = self.crypto._run(
                loader.download_macro_series(
                    provider="coinmarketcap",
                    series="GLOBAL",
                    persist=False,
                )
            )
            if records:
                global_metrics = _values(records[-1])
        except Exception as exc:
            failures.append(f"CMC_GLOBAL:{type(exc).__name__}:{str(exc)[:180]}")

        try:
            loader = self.crypto.data_loader()
            records = self.crypto._run(
                loader.download_cmc_rankings(
                    limit=250,
                    convert="EUR",
                    persist=False,
                )
            )
            for record in records:
                values = _values(record)
                market = _market(record)
                rows.append({"market": market, **values})
        except Exception as exc:
            failures.append(f"CMC_RANKINGS:{type(exc).__name__}:{str(exc)[:180]}")

        try:
            loader = self.crypto.data_loader()
            records = self.crypto._run(
                loader.download_macro_series(
                    provider="alternative_me",
                    series="fear_and_greed",
                    persist=False,
                )
            )
            if records:
                alternative_fear = _values(records[-1])
        except Exception as exc:
            failures.append(f"ALT_FEAR:{type(exc).__name__}:{str(exc)[:180]}")

        fear = self._cached(
            "fear_greed",
            "/v3/fear-and-greed/latest",
            ttl=300,
        )
        altseason = self._cached(
            "altcoin_season",
            "/v1/altcoin-season-index/latest",
            ttl=900,
        )
        cmc100 = self._cached(
            "cmc100",
            "/v3/index/cmc100-latest",
            ttl=900,
        )
        cmc20 = self._cached(
            "cmc20",
            "/v3/index/cmc20-latest",
            ttl=900,
        )
        trending = self._cached(
            "trending",
            "/v1/cryptocurrency/trending/latest",
            ttl=600,
            params={"limit": 30, "convert": "EUR"},
            key_required=True,
        )
        gainers = self._cached(
            "gainers",
            "/v1/cryptocurrency/trending/gainers-losers",
            ttl=600,
            params={
                "limit": 30,
                "sort": "percent_change_24h",
                "sort_dir": "desc",
                "convert": "EUR",
            },
            key_required=True,
        )
        losers = self._cached(
            "losers",
            "/v1/cryptocurrency/trending/gainers-losers",
            ttl=600,
            params={
                "limit": 30,
                "sort": "percent_change_24h",
                "sort_dir": "asc",
                "convert": "EUR",
            },
            key_required=True,
        )
        visited = self._cached(
            "most_visited",
            "/v1/cryptocurrency/trending/most-visited",
            ttl=21600,
            params={"limit": 30, "convert": "EUR"},
            key_required=True,
        )

        requested = {str(x).upper() for x in markets}
        assets = {
            str(row.get("market")).upper(): row
            for row in rows
            if str(row.get("market")).upper() in requested
        }

        output = {
            "schema_version": "crypto_ai_swing_cmc_context_v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "READY" if not failures else "DEGRADED",
            "global_metrics": global_metrics,
            "fear_and_greed": self._data(fear),
            "fear_and_greed_alternative_me": alternative_fear,
            "altcoin_season": self._data(altseason),
            "cmc100": self._data(cmc100),
            "cmc20": self._data(cmc20),
            "breadth_top250": self._breadth(rows),
            "runtime_assets": assets,
            "ranking_count": len(rows),
            "trending": self._trend_rows(self._data(trending)),
            "gainers": self._trend_rows(self._data(gainers)),
            "losers": self._trend_rows(self._data(losers)),
            "most_visited": self._trend_rows(self._data(visited)),
            "source_health": {
                "fear": {
                    "stale": fear.get("stale"),
                    "error": fear.get("refresh_error"),
                },
                "altseason": {
                    "stale": altseason.get("stale"),
                    "error": altseason.get("refresh_error"),
                },
                "trending": {
                    "stale": trending.get("stale"),
                    "error": trending.get("refresh_error"),
                },
            },
            "failures": failures,
            "authority": "CONTEXT_ONLY",
            "live_decision_influence": False,
            "orders_submitted": 0,
            "secrets_serialized": False,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest.write_text(
            json.dumps(output, indent=2, default=str),
            encoding="utf-8",
        )
        return output
