from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge

TIMEFRAME_SECONDS = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
    "1W": 604800,
}

RESIDUAL_MOMENTUM_MARKETS = (
    "BTC-EUR",
    "ETH-EUR",
    "SOL-EUR",
    "LINK-EUR",
)


def _utc(value: datetime | pd.Timestamp | str) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _native_timeframe(timeframe: str) -> str:
    value = str(timeframe).strip()
    return "1W" if value.lower() == "1w" else value


def _time_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "rows": 0, "start": None, "end": None}
    parquet = pq.ParquetFile(path)
    rows = int(parquet.metadata.num_rows)
    if rows <= 0 or "timestamp" not in parquet.schema_arrow.names:
        return {"exists": True, "rows": rows, "start": None, "end": None}
    first = parquet.read_row_group(0, columns=["timestamp"])["timestamp"]
    last = parquet.read_row_group(
        parquet.metadata.num_row_groups - 1,
        columns=["timestamp"],
    )["timestamp"]
    first_values = pd.to_datetime(first.to_pylist(), utc=True, errors="coerce")
    last_values = pd.to_datetime(last.to_pylist(), utc=True, errors="coerce")
    return {
        "exists": True,
        "rows": rows,
        "start": (
            first_values.min().isoformat()
            if len(first_values) and not pd.isna(first_values.min())
            else None
        ),
        "end": (
            last_values.max().isoformat()
            if len(last_values) and not pd.isna(last_values.max())
            else None
        ),
    }


@dataclass(frozen=True)
class HistoryProfile:
    name: str
    markets: tuple[str, ...]
    timeframes: tuple[str, ...]
    start_by_timeframe: dict[str, datetime]


class CanonicalHistoryManager:
    """Orchestrate Sjagil/crypto canonical history APIs without reimplementing them."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self.crypto = CryptoLibraryBridge(settings.crypto_repo_root)
        self.native_settings = self.crypto.settings()
        self.loader = self.crypto.data_loader()

    @property
    def processed_root(self) -> Path:
        return Path(self.native_settings.paths.processed_data_dir)

    def provider_path(
        self,
        provider: str,
        market: str,
        timeframe: str,
    ) -> Path:
        return (
            self.processed_root
            / provider.casefold()
            / str(market).upper()
            / f"{_native_timeframe(timeframe)}.parquet"
        )

    def flat_path(self, market: str, timeframe: str) -> Path:
        return (
            self.processed_root
            / f"{str(market).upper()}_{_native_timeframe(timeframe)}.parquet"
        )

    def storage_status(
        self,
        *,
        provider: str,
        markets: Iterable[str],
        timeframes: Iterable[str],
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for market in markets:
            for timeframe in timeframes:
                provider_path = self.provider_path(provider, market, timeframe)
                flat_path = self.flat_path(market, timeframe)
                rows.append(
                    {
                        "market": str(market).upper(),
                        "timeframe": _native_timeframe(timeframe),
                        "provider_path": str(provider_path),
                        "provider_storage": _time_summary(provider_path),
                        "flat_path": str(flat_path),
                        "flat_storage": _time_summary(flat_path),
                        "canonical_flat_ready": flat_path.is_file(),
                    }
                )
        return {
            "schema_version": "crypto_ai_swing_storage_contract_v1",
            "provider": provider.casefold(),
            "processed_root": str(self.processed_root),
            "series": rows,
            "provider_ready_series": sum(
                bool(row["provider_storage"]["exists"]) for row in rows
            ),
            "flat_ready_series": sum(
                bool(row["flat_storage"]["exists"]) for row in rows
            ),
            "total_series": len(rows),
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    @staticmethod
    def profile(
        name: str,
        *,
        now: datetime | None = None,
    ) -> HistoryProfile:
        selected = str(name).strip().lower()
        current = _utc(now or datetime.now(UTC))
        if selected == "residual-momentum":
            return HistoryProfile(
                name="residual-momentum",
                markets=RESIDUAL_MOMENTUM_MARKETS,
                timeframes=("1d",),
                start_by_timeframe={
                    "1d": datetime(2019, 1, 1, tzinfo=UTC),
                },
            )
        if selected == "swing-core":
            return HistoryProfile(
                name="swing-core",
                markets=RESIDUAL_MOMENTUM_MARKETS,
                timeframes=("15m", "1h", "2h", "4h", "1d"),
                start_by_timeframe={
                    "15m": current - timedelta(days=365 * 2),
                    "1h": current - timedelta(days=365 * 3),
                    "2h": current - timedelta(days=365 * 4),
                    "4h": current - timedelta(days=365 * 4),
                    "1d": datetime(2018, 1, 1, tzinfo=UTC),
                },
            )
        raise ValueError("profile must be residual-momentum or swing-core")

    def sync_series(
        self,
        *,
        provider: str,
        market: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> dict[str, Any]:
        selected_market = str(market).upper()
        selected_tf = _native_timeframe(timeframe)
        native_summary = self.crypto._run(
            self.loader.sync_canonical_ohlcv_compact(
                provider=provider.casefold(),
                market=selected_market,
                timeframe=selected_tf,
                start=_utc(start),
                end=_utc(end),
                resume=True,
            )
        )
        provider_path = self.provider_path(
            provider,
            selected_market,
            selected_tf,
        )
        flat_path = self.flat_path(selected_market, selected_tf)
        if not provider_path.is_file():
            return {
                "market": selected_market,
                "timeframe": selected_tf,
                "status": "BLOCKED_PROVIDER_CANONICAL_MISSING",
                "native_sync": native_summary,
                "provider_path": str(provider_path),
                "flat_path": str(flat_path),
                "orders_submitted": 0,
            }

        seconds = TIMEFRAME_SECONDS.get(selected_tf, 86400)
        materialized = self.loader.materialize_provider_ohlcv_compact(
            provider_path,
            flat_path,
            provider=provider.casefold(),
            market=selected_market,
            timeframe=selected_tf,
            maximum_staleness=timedelta(
                seconds=max(seconds * 3, 3600)
            ),
        )
        return {
            "market": selected_market,
            "timeframe": selected_tf,
            "status": "CANONICAL_FLAT_READY",
            "native_sync": native_summary,
            "materialization": materialized,
            "provider_path": str(provider_path),
            "flat_path": str(flat_path),
            "provider_storage": _time_summary(provider_path),
            "flat_storage": _time_summary(flat_path),
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    def sync_profile(self, name: str) -> dict[str, Any]:
        profile = self.profile(name)
        end = datetime.now(UTC)
        rows: list[dict[str, Any]] = []
        for timeframe in profile.timeframes:
            start = profile.start_by_timeframe[timeframe]
            for market in profile.markets:
                try:
                    row = self.sync_series(
                        provider="bitvavo",
                        market=market,
                        timeframe=timeframe,
                        start=start,
                        end=end,
                    )
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "market": market,
                        "timeframe": timeframe,
                        "status": "BLOCKED",
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                        "orders_submitted": 0,
                    }
                rows.append(row)
        return {
            "schema_version": "crypto_ai_swing_canonical_history_sync_v1",
            "profile": profile.name,
            "markets": list(profile.markets),
            "timeframes": list(profile.timeframes),
            "series": rows,
            "ready_series": sum(
                row.get("status") == "CANONICAL_FLAT_READY"
                for row in rows
            ),
            "blocked_series": sum(
                row.get("status") != "CANONICAL_FLAT_READY"
                for row in rows
            ),
            "total_series": len(rows),
            "source": (
                "Sjagil/crypto:DataLoader.sync_canonical_ohlcv_compact+"
                "materialize_provider_ohlcv_compact"
            ),
            "automatic_live_promotion": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    def residual_momentum_preflight(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for market in RESIDUAL_MOMENTUM_MARKETS:
            path = self.flat_path(market, "1d")
            rows.append(
                {
                    "market": market,
                    "path": str(path),
                    **_time_summary(path),
                }
            )
        files_ready = all(bool(row["exists"]) for row in rows)
        confirmation_end = pd.Timestamp("2026-07-24T00:00:00Z")
        confirmation_covered = all(
            row["end"] is not None
            and pd.Timestamp(row["end"]) >= confirmation_end
            for row in rows
        )
        minimum_rows_ready = all(int(row["rows"]) >= 365 for row in rows)
        blockers: list[str] = []
        if not files_ready:
            blockers.append("CANONICAL_FLAT_DATASETS_MISSING")
        if files_ready and not minimum_rows_ready:
            blockers.append("INSUFFICIENT_DAILY_HISTORY_ROWS")
        if files_ready and not confirmation_covered:
            blockers.append("CONFIRMATION_PERIOD_NOT_COVERED")
        return {
            "schema_version": "crypto_ai_swing_residual_momentum_preflight_v1",
            "campaign": "RESIDUAL_MOMENTUM_V1",
            "ready_to_invoke_native_campaign": not blockers,
            "series": rows,
            "blockers": blockers,
            "note": (
                "Native campaign validation remains authoritative for "
                "listing-aware historical sufficiency and promotion."
            ),
            "authority": "RESEARCH_ONLY",
            "automatic_live_promotion": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
