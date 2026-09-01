from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
import asyncio
import importlib
import inspect
import json
import sys
import time

import pandas as pd


class CryptoLibraryError(RuntimeError):
    pass


@dataclass(frozen=True)
class CryptoModuleStatus:
    module: str
    imported: bool
    file: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class CryptoMarketBundle:
    market: str
    timeframe: str
    frame: pd.DataFrame
    ticker: dict[str, Any]
    trades: list[dict[str, Any]]
    orderbook: dict[str, Any]
    microstructure: dict[str, float]


REQUIRED_CRYPTO_MODULES: tuple[str, ...] = (
    "config.settings",
    "data.data_loader",
    "data.database",
    "data.websocket_manager",
    "data.orderflow_recorder",
    "data.orderbook_l2",
    "data.bitvavo_l2_reconstruction_v2",
    "data.realtime_candle_builder",
    "data.multi_source_runtime",
    "data.multi_source_platform",
    "data.multi_source_maturation",
    "data.collector_health",
    "data.feature_store",
    "data.prospective_context",
    "core.market_intelligence",
    "core.opportunity_intelligence",
    "core.practical_governance",
    "core.execution_authority",
    "core.account_inventory",
    "core.live_asset_preflight",
    "core.autonomous_live",
    "research.features",
    "research.research_factory",
    "scrapers.rss",
    "scrapers.intelligence",
)


class CryptoLibraryBridge:
    """Use the existing Sjagil/crypto checkout as the runtime library.

    The crypto repository remains authoritative for provider settings,
    Bitvavo market data, CMC/prospective context, order flow, data health,
    account/inventory governance and execution authority.
    """

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise CryptoLibraryError(
                f"Crypto repository does not exist: {self.root}"
            )
        self._activate()
        self._frame_cache: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
        self._prospective_cache: tuple[float, dict[str, Any]] | None = None

    def _activate(self) -> None:
        root = str(self.root)
        if root not in sys.path:
            sys.path.insert(0, root)

    def import_module(self, name: str):
        self._activate()
        module = importlib.import_module(name)
        file = Path(str(getattr(module, "__file__", "") or "")).resolve()
        if file and self.root not in file.parents and file != self.root:
            raise CryptoLibraryError(
                f"Imported {name} from outside crypto repo: {file}"
            )
        return module

    def integration_status(
        self,
        modules: Iterable[str] = REQUIRED_CRYPTO_MODULES,
    ) -> dict[str, Any]:
        statuses: list[CryptoModuleStatus] = []
        for name in modules:
            try:
                module = self.import_module(name)
                statuses.append(
                    CryptoModuleStatus(
                        module=name,
                        imported=True,
                        file=str(getattr(module, "__file__", "") or "") or None,
                    )
                )
            except Exception as exc:
                statuses.append(
                    CryptoModuleStatus(
                        module=name,
                        imported=False,
                        error=f"{type(exc).__name__}: {str(exc)[:300]}",
                    )
                )
        return {
            "crypto_repo_root": str(self.root),
            "required_modules": len(statuses),
            "imported_modules": sum(x.imported for x in statuses),
            "ready": bool(statuses) and all(x.imported for x in statuses),
            "modules": [asdict(x) for x in statuses],
        }

    def public_interfaces(self, module_name: str) -> dict[str, str]:
        module = self.import_module(module_name)
        result: dict[str, str] = {}
        for name, value in vars(module).items():
            if name.startswith("_"):
                continue
            if inspect.isfunction(value) or inspect.isclass(value):
                try:
                    result[name] = str(inspect.signature(value))
                except Exception:
                    result[name] = "<signature unavailable>"
        return result

    def settings(self):
        module = self.import_module("config.settings")
        get_settings = getattr(module, "get_settings", None)
        if not callable(get_settings):
            raise CryptoLibraryError(
                "config.settings.get_settings is unavailable"
            )
        return get_settings()

    def data_loader(self):
        module = self.import_module("data.data_loader")
        cls = getattr(module, "DataLoader", None)
        if cls is None:
            raise CryptoLibraryError(
                "data.data_loader.DataLoader is unavailable"
            )
        return cls(self.settings())

    @staticmethod
    def _run(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise CryptoLibraryError(
            "Synchronous bridge API cannot run inside an active asyncio loop"
        )

    @staticmethod
    def _record_values(record: Any) -> dict[str, Any]:
        values = getattr(record, "values", None)
        if isinstance(values, Mapping):
            return dict(values)
        if isinstance(record, Mapping):
            return dict(record)
        dump = getattr(record, "model_dump", None)
        if callable(dump):
            raw = dump(mode="python")
            if isinstance(raw, Mapping):
                nested = raw.get("values")
                return dict(nested) if isinstance(nested, Mapping) else dict(raw)
        return {}

    @staticmethod
    def _record_timestamp(record: Any) -> pd.Timestamp | None:
        value = getattr(record, "timestamp", None)
        if value is None and isinstance(record, Mapping):
            value = record.get("timestamp") or record.get("observed_at")
        if value is None:
            return None
        try:
            ts = pd.Timestamp(value)
            return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        except Exception:
            return None

    def _ohlcv_frame(self, records: Iterable[Any]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for record in records:
            if getattr(record, "closed", True) is False:
                continue
            ts = self._record_timestamp(record)
            values = self._record_values(record)
            if ts is None:
                continue
            try:
                rows.append(
                    {
                        "timestamp": ts,
                        "open": float(values["open"]),
                        "high": float(values["high"]),
                        "low": float(values["low"]),
                        "close": float(values["close"]),
                        "volume": float(values.get("volume", 0.0)),
                    }
                )
            except Exception:
                continue
        if not rows:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )
        return (
            pd.DataFrame(rows)
            .drop_duplicates("timestamp", keep="last")
            .set_index("timestamp")
            .sort_index()
        )

    @staticmethod
    def _book_levels(
        values: Mapping[str, Any], side: str
    ) -> list[tuple[float, float]]:
        levels = values.get(side) or values.get(f"book_{side}") or []
        result: list[tuple[float, float]] = []
        for item in levels:
            if isinstance(item, Mapping):
                price = item.get("price")
                amount = (
                    item.get("amount")
                    or item.get("size")
                    or item.get("quantity")
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                price, amount = item[0], item[1]
            else:
                continue
            try:
                result.append((float(price), float(amount)))
            except Exception:
                continue
        return result

    @classmethod
    def microstructure_features(
        cls,
        ticker: Mapping[str, Any],
        trades: Iterable[Mapping[str, Any]],
        orderbook: Mapping[str, Any],
    ) -> dict[str, float]:
        bids = cls._book_levels(orderbook, "bids")
        asks = cls._book_levels(orderbook, "asks")
        best_bid = (
            bids[0][0]
            if bids
            else float(ticker.get("bid") or ticker.get("best_bid") or 0.0)
        )
        best_ask = (
            asks[0][0]
            if asks
            else float(ticker.get("ask") or ticker.get("best_ask") or 0.0)
        )
        mid = (
            (best_bid + best_ask) / 2.0
            if best_bid > 0 and best_ask > 0
            else 0.0
        )
        spread_bps = (
            (best_ask - best_bid) / mid * 10000.0 if mid > 0 else 999.0
        )

        bid_depth = sum(amount for _, amount in bids[:10])
        ask_depth = sum(amount for _, amount in asks[:10])
        total_depth = bid_depth + ask_depth
        book_imbalance = (
            (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0
        )

        microprice = mid
        if bids and asks and bids[0][1] + asks[0][1] > 0:
            microprice = (
                best_ask * bids[0][1] + best_bid * asks[0][1]
            ) / (bids[0][1] + asks[0][1])
        microprice_edge_bps = (
            (microprice - mid) / mid * 10000.0 if mid > 0 else 0.0
        )

        buy_volume = 0.0
        sell_volume = 0.0
        buy_notional = 0.0
        sell_notional = 0.0
        trade_count = 0
        for trade in trades:
            trade_count += 1
            side = str(
                trade.get("side")
                or trade.get("aggressor_side")
                or trade.get("taker_side")
                or ""
            ).lower()
            amount = (
                trade.get("quantity")
                or trade.get("amount")
                or trade.get("volume")
                or trade.get("size")
                or 0.0
            )
            try:
                qty = float(amount)
            except Exception:
                qty = 0.0
            try:
                trade_price = float(trade.get("price") or 0.0)
            except Exception:
                trade_price = 0.0
            if side in {"buy", "bid", "buyer", "b"}:
                buy_volume += qty
                buy_notional += qty * trade_price
            elif side in {"sell", "ask", "seller", "s"}:
                sell_volume += qty
                sell_notional += qty * trade_price
        classified = buy_volume + sell_volume
        classified_trade_count = 0
        for trade in trades:
            side = str(
                trade.get("side")
                or trade.get("aggressor_side")
                or trade.get("taker_side")
                or ""
            ).lower()
            if side in {"buy", "bid", "buyer", "b", "sell", "ask", "seller", "s"}:
                classified_trade_count += 1
        cvd_ratio = (
            (buy_volume - sell_volume) / classified
            if classified > 0
            else 0.0
        )
        classified_notional = buy_notional + sell_notional
        cvd_notional_ratio = (
            (buy_notional - sell_notional) / classified_notional
            if classified_notional > 0
            else 0.0
        )

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread_bps": spread_bps,
            "book_imbalance": book_imbalance,
            "microprice_edge_bps": microprice_edge_bps,
            "trade_count": float(trade_count),
            "classified_trade_count": float(classified_trade_count),
            "unclassified_trade_count": float(max(0, trade_count - classified_trade_count)),
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_notional": buy_notional,
            "sell_notional": sell_notional,
            "cvd_ratio": cvd_ratio,
            "cvd_notional_ratio": cvd_notional_ratio,
        }

    @staticmethod
    def timeframe_seconds(timeframe: str) -> int:
        mapping = {
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "1d": 86400,
            "1w": 604800,
            "1W": 604800,
        }
        if timeframe not in mapping:
            raise CryptoLibraryError(f"Unsupported timeframe: {timeframe}")
        return mapping[timeframe]

    @classmethod
    def causal_frame(
        cls,
        frame: pd.DataFrame,
        timeframe: str,
        decision_at: pd.Timestamp | datetime,
    ) -> pd.DataFrame:
        """Keep only bars fully closed by decision_at. Index is candle-open time."""
        if frame is None or frame.empty:
            return frame.copy() if frame is not None else pd.DataFrame()
        decision = pd.Timestamp(decision_at)
        decision = decision.tz_convert("UTC") if decision.tzinfo else decision.tz_localize("UTC")
        close_times = frame.index + pd.to_timedelta(cls.timeframe_seconds(timeframe), unit="s")
        return frame.loc[close_times <= decision].copy()

    @staticmethod
    def quote_volume_24h(
        ticker: Mapping[str, Any],
        microstructure: Mapping[str, float] | None = None,
    ) -> float:
        for key in (
            "volumeQuote",
            "quoteVolume",
            "quote_volume",
            "volume_quote",
            "quote_volume_24h",
            "volumeQuote24h",
        ):
            value = ticker.get(key)
            try:
                selected = float(value)
                if selected > 0:
                    return selected
            except Exception:
                pass
        base_volume = None
        for key in ("volume", "baseVolume", "base_volume", "volume_24h"):
            try:
                selected = float(ticker.get(key) or 0.0)
                if selected > 0:
                    base_volume = selected
                    break
            except Exception:
                pass
        price = 0.0
        for key in ("price", "last", "lastPrice", "close"):
            try:
                selected = float(ticker.get(key) or 0.0)
                if selected > 0:
                    price = selected
                    break
            except Exception:
                pass
        if price <= 0 and microstructure:
            bid = float(microstructure.get("best_bid", 0.0) or 0.0)
            ask = float(microstructure.get("best_ask", 0.0) or 0.0)
            if bid > 0 and ask > 0:
                price = (bid + ask) / 2.0
        return float(base_volume * price) if base_volume and price > 0 else 0.0

    @staticmethod
    def market_bundle_audit(bundle: CryptoMarketBundle) -> dict[str, Any]:
        side_counts: dict[str, int] = {}
        trade_keys: set[str] = set()
        for trade in bundle.trades:
            trade_keys.update(str(k) for k in trade.keys())
            side = str(
                trade.get("side")
                or trade.get("aggressor_side")
                or trade.get("taker_side")
                or "<missing>"
            ).lower()
            side_counts[side] = side_counts.get(side, 0) + 1
        return {
            "market": bundle.market,
            "timeframe": bundle.timeframe,
            "rows": len(bundle.frame),
            "ticker_keys": sorted(str(k) for k in bundle.ticker.keys()),
            "ticker_public": {
                str(k): v
                for k, v in bundle.ticker.items()
                if str(k).lower() not in {"apikey", "api_key", "secret", "signature"}
            },
            "quote_volume_24h": CryptoLibraryBridge.quote_volume_24h(
                bundle.ticker, bundle.microstructure
            ),
            "trade_count": len(bundle.trades),
            "trade_keys": sorted(trade_keys),
            "trade_side_counts": dict(sorted(side_counts.items())),
            "orderbook_keys": sorted(str(k) for k in bundle.orderbook.keys()),
            "bid_levels": len(CryptoLibraryBridge._book_levels(bundle.orderbook, "bids")),
            "ask_levels": len(CryptoLibraryBridge._book_levels(bundle.orderbook, "asks")),
            "microstructure": dict(bundle.microstructure),
        }

    @staticmethod
    def _lookback_hours(timeframe: str) -> int:
        return {
            "15m": 24 * 8,
            "1h": 24 * 45,
            "2h": 24 * 90,
            "4h": 24 * 180,
            "1d": 24 * 280,
            "1w": 24 * 7 * 220,
        }.get(timeframe, 24 * 45)

    @staticmethod
    def _cache_ttl(timeframe: str) -> float:
        return {
            "15m": 45.0,
            "1h": 180.0,
            "2h": 300.0,
            "4h": 600.0,
            "1d": 1800.0,
            "1w": 7200.0,
        }.get(timeframe, 180.0)

    async def _ohlcv_async(
        self,
        market: str,
        timeframe: str,
        *,
        persist: bool,
    ) -> pd.DataFrame:
        key = (market, timeframe)
        cached = self._frame_cache.get(key)
        now = time.time()
        if cached and now - cached[0] <= self._cache_ttl(timeframe):
            return cached[1].copy()
        loader = self.data_loader()
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=self._lookback_hours(timeframe))
        records = await loader.download_ohlcv(
            provider="bitvavo",
            market=market,
            timeframe=timeframe,
            start=start,
            end=end,
            resume=True,
            persist=persist,
        )
        frame = self._ohlcv_frame(records)
        self._frame_cache[key] = (now, frame.copy())
        return frame

    def ohlcv(
        self,
        market: str,
        timeframe: str,
        *,
        persist: bool = False,
    ) -> pd.DataFrame:
        return self._run(
            self._ohlcv_async(market, timeframe, persist=persist)
        )

    async def _ohlcv_many_async(
        self,
        markets: Iterable[str],
        timeframe: str,
        *,
        persist: bool,
        concurrency: int,
    ) -> dict[str, pd.DataFrame]:
        semaphore = asyncio.Semaphore(max(1, int(concurrency)))

        async def one(market: str):
            async with semaphore:
                try:
                    frame = await self._ohlcv_async(
                        market, timeframe, persist=persist
                    )
                    return market, frame, None
                except Exception as exc:
                    return market, pd.DataFrame(), exc

        rows = await asyncio.gather(
            *(one(str(market).upper()) for market in markets)
        )
        result: dict[str, pd.DataFrame] = {}
        for market, frame, _error in rows:
            result[market] = frame
        return result

    def ohlcv_many(
        self,
        markets: Iterable[str],
        timeframe: str,
        *,
        persist: bool = False,
        concurrency: int = 4,
    ) -> dict[str, pd.DataFrame]:
        return self._run(
            self._ohlcv_many_async(
                markets, timeframe, persist=persist, concurrency=concurrency
            )
        )

    async def _market_bundle_async(
        self,
        market: str,
        timeframe: str,
        *,
        mode: str,
        persist: bool,
        depth: int,
    ) -> CryptoMarketBundle:
        loader = self.data_loader()
        frame = await self._ohlcv_async(
            market,
            timeframe,
            persist=persist,
        )
        ticker_record = await loader.download_ticker(
            provider="bitvavo",
            market=market,
            persist=persist,
            mode=mode,
        )
        trade_records = await loader.download_trades(
            provider="bitvavo",
            market=market,
            persist=persist,
            mode=mode,
        )
        book_record = await loader.download_orderbook_snapshot(
            provider="bitvavo",
            market=market,
            depth=depth,
            persist=persist,
            mode=mode,
        )
        ticker = self._record_values(ticker_record)
        trades = [self._record_values(x) for x in trade_records]
        orderbook = self._record_values(book_record)
        micro = self.microstructure_features(ticker, trades, orderbook)
        return CryptoMarketBundle(
            market,
            timeframe,
            frame,
            ticker,
            trades,
            orderbook,
            micro,
        )

    def market_bundle(
        self,
        market: str,
        timeframe: str = "1h",
        *,
        mode: str = "shadow",
        persist: bool = False,
        depth: int = 100,
    ) -> CryptoMarketBundle:
        return self._run(
            self._market_bundle_async(
                market,
                timeframe,
                mode=mode,
                persist=persist,
                depth=depth,
            )
        )

    def multi_timeframe_frames(
        self,
        market: str,
        timeframes: Iterable[str] = ("15m", "1h", "2h", "4h", "1d", "1w"),
        *,
        persist: bool = False,
    ) -> dict[str, pd.DataFrame]:
        return {
            str(tf): self.ohlcv(market, str(tf), persist=persist)
            for tf in timeframes
        }

    async def _prospective_context_async(
        self,
        markets: Iterable[str],
        observed_at: datetime | None,
    ) -> dict[str, Any]:
        module = self.import_module("data.prospective_context")
        collector_cls = getattr(module, "ProspectiveContextCollector", None)
        if collector_cls is None:
            raise CryptoLibraryError(
                "ProspectiveContextCollector is unavailable"
            )
        settings = self.settings()
        collector = collector_cls(
            checkpoint_path=(
                settings.paths.checkpoints_dir
                / "prospective_context_hourly.json"
            ),
            snapshot_directory=(
                settings.paths.context_data_dir
                / "prospective_hourly"
            ),
        )
        result = await collector.collect(
            loader=self.data_loader(),
            markets=tuple(markets),
            observed_at=observed_at or datetime.now(timezone.utc),
        )
        return dict(result) if isinstance(result, Mapping) else {"value": result}

    def prospective_context(
        self,
        markets: Iterable[str],
        observed_at: datetime | None = None,
        *,
        cache_seconds: float = 300.0,
    ) -> dict[str, Any]:
        now = time.time()
        if (
            self._prospective_cache
            and now - self._prospective_cache[0] <= cache_seconds
        ):
            return dict(self._prospective_cache[1])
        result = self._run(
            self._prospective_context_async(markets, observed_at)
        )
        self._prospective_cache = (now, dict(result))
        return result

    def microstructure_readiness(self) -> dict[str, Any]:
        module = self.import_module("data.orderflow_recorder")
        fn = getattr(module, "current_microstructure_readiness", None)
        if not callable(fn):
            return {"status": "UNAVAILABLE"}
        settings = self.settings()
        feature_dir = (
            settings.paths.context_data_dir / "microstructure_hourly"
        )
        try:
            result = fn(feature_dir)
        except TypeError:
            result = fn()
        return dict(result) if isinstance(result, Mapping) else {"value": result}

    def execution_authority_interfaces(self) -> dict[str, str]:
        return self.public_interfaces("core.execution_authority")

    def write_integration_report(self, path: Path) -> dict[str, Any]:
        report = self.integration_status()
        try:
            report["execution_authority_interfaces"] = (
                self.execution_authority_interfaces()
            )
        except Exception as exc:
            report["execution_authority_interfaces"] = {}
            report["execution_authority_error"] = (
                f"{type(exc).__name__}: {str(exc)[:300]}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )
        return report
