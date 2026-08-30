from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import json


DEFAULT_CANDIDATES: dict[str, tuple[str, ...]] = {
    # Bitvavo public/realtime market data.
    "bitvavo_ws": (
        "data/websocket_manager.py",
        "data/orderflow_recorder.py",
    ),

    # Bitvavo orderbook/L2 infrastructure.
    "bitvavo_market_data_pro_ws": (
        "data/orderbook_l2.py",
        "data/bitvavo_l2_reconstruction_v2.py",
        "data/orderflow_recorder.py",
    ),

    # Authenticated Bitvavo stream.
    "bitvavo_private_stream": (
        "data/bitvavo_private_stream.py",
    ),

    # CoinMarketCap integration.
    "coinmarketcap_client": (
        # Current crypto repository.
        "legacy/data_downloaders/crypto_data_downloader.py",
        "legacy/data_downloaders/multi_venue_crypto_data_downloader.py",

        # Backwards-compatible v7 layout.
        "src/data/providers/coinmarketcap_client.py",
    ),

    # Provider configuration and multi-source capability layer.
    "provider_capabilities": (
        "config/settings.py",
        "data/multi_source_platform.py",
        "data/multi_source_maturation.py",
    ),

    # Live/realtime ingestion pipeline.
    "market_data_ingestion": (
        "data/websocket_manager.py",
        "data/orderflow_recorder.py",
        "data/multi_source_runtime.py",
        "data/market_data.py",
    ),

    # Current market state / opportunity state.
    "realtime_market_state": (
        "data/orderflow_recorder.py",
        "data/realtime_candle_builder.py",
        "core/opportunity_intelligence.py",
        "core/market_intelligence.py",
    ),

    # State construction.
    "realtime_market_state_builder": (
        "data/realtime_candle_builder.py",
        "data/orderflow_recorder.py",
        "data/market_structure.py",
    ),

    # Historical-data infrastructure.
    "historical_data_lake": (
        "data/downloader.py",
        "data/data_loader.py",
        "data/database.py",
        "data/feature_store.py",
        "data/multi_source_platform.py",
    ),

    # Existing data-health / validation components.
    "historical_data_validator": (
        "data/collector_health.py",
        "data/multi_source_maturation.py",
        "reporting/multi_source_data_platform.py",
        "reporting/multi_source_maturation_platform.py",
    ),

    # Existing feature / research dataset construction.
    "training_dataset_builder": (
        "data/feature_store.py",
        "research/features.py",
        "research/research_factory.py",
    ),

    # Order-flow engine.
    "orderflow": (
        "data/orderflow_recorder.py",
        "data/orderbook_l2.py",
    ),

    # Multi-source runtime.
    "multi_source_runtime": (
        "data/multi_source_runtime.py",
        "data/multi_source_platform.py",
    ),

    # Realtime candles.
    "realtime_candles": (
        "data/realtime_candle_builder.py",
    ),

    # Existing research/intelligence inputs.
    "rss": (
        "scrapers/rss.py",
    ),
    "intelligence": (
        "scrapers/intelligence.py",
    ),

    # Execution authority. Discovery only, never called directly by
    # strategies or AI models.
    "execution_authority": (
        "core/execution_authority.py",
        "core/practical_governance.py",
    ),

    # Existing autonomous/live orchestration.
    "launcher": (
        "core/autonomous_live.py",
        "core/cli.py",
    ),
}


@dataclass(frozen=True)
class DiscoveryItem:
    capability: str
    found: bool
    paths: tuple[str, ...]


def discover(root: Path, candidates: dict[str, Iterable[str]] | None = None) -> list[DiscoveryItem]:
    root = root.resolve()
    mapping = candidates or DEFAULT_CANDIDATES
    result: list[DiscoveryItem] = []
    for capability, rels in mapping.items():
        rel_tuple = tuple(str(x) for x in rels)
        found_paths = tuple(rel for rel in rel_tuple if (root / rel).exists())
        result.append(DiscoveryItem(capability, bool(found_paths), found_paths))
    return result


def write_report(root: Path, report_path: Path) -> dict:
    items = discover(root)
    report = {
        "crypto_repo_root": str(root.resolve()),
        "repo_exists": root.exists(),
        "found_count": sum(item.found for item in items),
        "total_candidates": len(items),
        "items": [asdict(item) for item in items],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
