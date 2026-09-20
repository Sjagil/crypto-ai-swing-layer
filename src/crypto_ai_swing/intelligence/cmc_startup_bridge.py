"""Bridge to canonical CoinMarketCap Startup intelligence."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge

GLOBAL_PREFIXES = (
    "global_",
    "fear_greed",
    "altcoin_season",
    "cmc100",
    "cmc20",
    "trending_",
    "exchange_",
    "derivatives_exchanges",
    "liquidations_total",
    "liquidations_by_exchange",
    "rwa_",
    "cmc_ai_",
    "community_",
    "content_",
    "dex_spot_pairs",
    "dex_trending",
    "dex_new",
    "dex_meme",
    "dex_gainer_loser",
    "dex_liquidity_change",
    "dex_platform",
    "listings_",
    "categories",
)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _normalize_crypto_ids(value: Any) -> set[int]:
    """Normalize CMC push crypto_ids from scalar or collection form."""
    if value is None:
        return set()

    if isinstance(value, (list, tuple, set, frozenset)):
        values = value
    else:
        values = (value,)

    result: set[int] = set()

    for raw in values:
        try:
            result.add(int(raw))
        except (TypeError, ValueError):
            continue

    return result


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    return value if math.isfinite(value) else None


def _skip_key(value: str) -> bool:
    key = str(value).lower()

    if key in {
        "id",
        "cmc_id",
        "exchange_id",
        "platform_id",
        "token_id",
        "pool_id",
        "pair_id",
    }:
        return True

    if key.endswith("_id"):
        return True

    return any(
        token in key
        for token in (
            "timestamp",
            "time_open",
            "time_close",
            "address",
            "hash",
            "url",
        )
    )


def _collect(
    value: Any,
    prefix: str = "",
    output: dict[str, list[float]] | None = None,
    depth: int = 0,
) -> dict[str, list[float]]:
    result = output if output is not None else defaultdict(list)

    if depth > 7:
        return result

    if isinstance(value, Mapping):
        for key, item in value.items():
            if _skip_key(str(key)):
                continue

            name = f"{prefix}.{key}" if prefix else str(key)

            _collect(
                item,
                name,
                result,
                depth + 1,
            )

        return result

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for item in value:
            _collect(
                item,
                prefix,
                result,
                depth + 1,
            )

        return result

    number = _finite(value)

    if number is not None and prefix:
        result[prefix].append(number)

    return result


def _summarize(
    value: Any,
    *,
    maximum_features: int,
) -> dict[str, float]:
    buckets = _collect(value)
    result: dict[str, float] = {}

    for name in sorted(buckets):
        values = [value for value in buckets[name] if math.isfinite(value)]

        if not values:
            continue

        if len(values) == 1:
            result[name] = values[0]
        else:
            mean = sum(values) / len(values)

            result[f"{name}.mean"] = mean
            result[f"{name}.median"] = median(values)
            result[f"{name}.min"] = min(values)
            result[f"{name}.max"] = max(values)

            variance = sum((value - mean) ** 2 for value in values) / len(values)

            result[f"{name}.std"] = variance**0.5
            result[f"{name}.count"] = float(len(values))

        if len(result) >= maximum_features:
            break

    return dict(list(sorted(result.items()))[:maximum_features])


def _identity(
    value: Mapping[str, Any],
) -> tuple[int | None, str | None]:
    cmc_id = None

    for key in (
        "cmc_id",
        "id",
        "cryptocurrency_id",
        "crypto_id",
        "token_id",
    ):
        try:
            cmc_id = int(value[key])
            break
        except (KeyError, TypeError, ValueError):
            continue

    symbol = value.get("symbol")

    return (
        cmc_id,
        str(symbol).upper() if symbol else None,
    )


def _asset_payload(
    value: Any,
    *,
    cmc_id: int,
    symbol: str,
) -> Any | None:
    if isinstance(value, Mapping):
        if str(cmc_id) in value:
            return value[str(cmc_id)]

        own_id, own_symbol = _identity(value)

        if own_id == cmc_id or own_symbol == symbol:
            return value

        selected = {}

        for key, item in value.items():
            if not isinstance(item, Mapping):
                continue

            item_id, item_symbol = _identity(item)

            if item_id == cmc_id or item_symbol == symbol:
                selected[str(key)] = item

        return selected or None

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        selected = []

        for item in value:
            if not isinstance(item, Mapping):
                continue

            item_id, item_symbol = _identity(item)

            if item_id == cmc_id or item_symbol == symbol:
                selected.append(item)

        return selected or None

    return None


class CMCStartupBridge:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.crypto = CryptoLibraryBridge(settings.crypto_repo_root)

        self.module = self.crypto.import_module("data.cmc_startup_intelligence")

        self.service = self.module.CMCStartupIntelligence(self.crypto.settings())

    def status(self) -> dict[str, Any]:
        payload = dict(self.service.status())

        payload.update(
            {
                "authority": ("CONTEXT_RESEARCH_ONLY"),
                "execution_authority": False,
                "capital_authority": False,
            }
        )

        return payload

    def coverage(
        self,
        markets: Sequence[str],
    ) -> dict[str, Any]:
        return dict(self.service.coverage_report(list(markets)))

    def refresh(
        self,
        markets: Sequence[str],
    ) -> dict[str, Any]:
        return dict(self.service.incremental_refresh(list(markets)))

    def bootstrap(
        self,
        markets: Sequence[str],
    ) -> dict[str, Any]:
        return dict(self.service.bootstrap(list(markets)))

    @property
    def feature_path(self) -> Path:
        return Path(self.service.feature_root) / "pit_features.parquet"

    def forward_features(
        self,
        markets: Sequence[str],
    ) -> dict[str, Any]:
        latest = _load(Path(self.service.state_root) / "latest_context.json")

        websocket = _load(Path(self.service.state_root) / "websocket_latest.json")

        assets = self.service.resolve_assets(list(markets))

        identities = {
            f"{str(row.symbol).upper()}-EUR": (
                int(row.cmc_id),
                str(row.symbol).upper(),
            )
            for row in assets.itertuples(index=False)
        }

        sources = dict(latest.get("sources") or {})

        global_features: dict[str, float] = {}

        for name, raw in sorted(sources.items()):
            if not str(name).startswith(GLOBAL_PREFIXES):
                continue

            data = dict(raw or {}).get("data")

            summary = _summarize(
                data,
                maximum_features=40,
            )

            for key, value in summary.items():
                global_features[f"{name}.{key}"] = value

                if len(global_features) >= 180:
                    break

            if len(global_features) >= 180:
                break

        contracts = self.service._contract_records(  # noqa: SLF001
            {identity[0] for identity in identities.values()}
        )

        by_id: dict[
            int,
            set[tuple[str, str]],
        ] = defaultdict(set)

        for row in contracts:
            try:
                platform_id = str(row.get("platform_id") or row.get("platform"))

                address = str(row.get("address")).lower()

                by_id[int(row["cmc_id"])].add(
                    (
                        platform_id,
                        address,
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue

        ws_events = dict(websocket.get("events") or {})

        market_rows: dict[str, Any] = {}

        for market in [str(value).upper() for value in markets]:
            if market not in identities:
                market_rows[market] = {
                    "status": "UNMAPPED",
                    "features": {},
                }
                continue

            cmc_id, symbol = identities[market]
            features: dict[str, float] = {}

            for name, raw in sorted(sources.items()):
                data = dict(raw or {}).get("data")

                selected = _asset_payload(
                    data,
                    cmc_id=cmc_id,
                    symbol=symbol,
                )

                if selected is None and (
                    str(name).startswith(f"derivatives_crypto_{cmc_id}")
                    or f":{cmc_id}:" in str(name)
                ):
                    selected = data

                if selected is None:
                    continue

                summary = _summarize(
                    selected,
                    maximum_features=48,
                )

                for key, value in summary.items():
                    features[f"rest.{name}.{key}"] = value

                    if len(features) >= 220:
                        break

                if len(features) >= 220:
                    break

            for event in ws_events.values():
                if not isinstance(event, Mapping):
                    continue

                params = dict(event.get("params") or {})

                event_market = str(
                    event.get("market") or ""
                ).upper()

                match = event_market == market

                if not match:
                    match = cmc_id in _normalize_crypto_ids(
                        params.get("crypto_ids")
                    )

                if not match:
                    platform = str(params.get("platform_id") or params.get("platform") or "")

                    address = str(params.get("address") or "").lower()

                    match = (
                        platform,
                        address,
                    ) in by_id.get(
                        cmc_id,
                        set(),
                    )

                if not match:
                    continue

                channel = str(event.get("channel") or "unknown")

                summary = _summarize(
                    event.get("data"),
                    maximum_features=48,
                )

                for key, value in summary.items():
                    features[f"ws.{channel}.{key}"] = value

                    if len(features) >= 300:
                        break

                if len(features) >= 300:
                    break

            market_rows[market] = {
                "status": ("READY" if features else "NO_FORWARD_FEATURES"),
                "cmc_id": cmc_id,
                "source_feature_count": (len(features)),
                "features": dict(sorted(features.items())),
            }

        return {
            "status": "READY",
            "generated_at": latest.get("generated_at"),
            "global_features": dict(sorted(global_features.items())),
            "global_feature_count": (len(global_features)),
            "markets": market_rows,
            "forward_only": True,
            "point_in_time": True,
            "automatic_live_promotion": False,
            "execution_authority": False,
        }


__all__ = [
    "CMCStartupBridge",
]
