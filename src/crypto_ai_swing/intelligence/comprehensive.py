
from __future__ import annotations

import gzip
import hashlib
import json
import math
import subprocess
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge


SCHEMA = "round44_comprehensive_intelligence_v2"
L3_STATUS = "L3_UNSUPPORTED_BY_EXECUTION_VENUE"
TIMEFRAMES = ("15m", "1h", "2h", "4h", "1d", "1w")
PROVIDER_REQUIRED = "UNAVAILABLE/DATA_PROVIDER_REQUIRED"


def _finite(value: Any) -> float | None:
    try:
        selected = float(value)
    except (TypeError, ValueError):
        return None
    return selected if np.isfinite(selected) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _hash(payload: Any) -> str:
    raw = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _git_sha(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except Exception:
        return None


def _flatten_numeric(
    prefix: str,
    value: Any,
    *,
    maximum: int = 256,
    depth: int = 0,
    output: dict[str, float] | None = None,
) -> dict[str, float]:
    result = output if output is not None else {}
    if len(result) >= maximum or depth > 6:
        return result
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            _flatten_numeric(
                f"{prefix}.{key}" if prefix else str(key),
                value[key],
                maximum=maximum,
                depth=depth + 1,
                output=result,
            )
            if len(result) >= maximum:
                break
        return result
    if isinstance(value, (list, tuple)):
        return result
    number = _finite(value)
    if number is not None:
        result[prefix] = number
    return result


def _availability(
    *,
    ready: bool,
    source: str,
    reason: str | None = None,
    stale: bool = False,
    unsupported: bool = False,
    provider_required: bool = False,
) -> dict[str, Any]:
    if unsupported:
        status = "UNSUPPORTED"
    elif ready and not stale:
        status = "READY"
    elif ready and stale:
        status = "STALE"
    elif provider_required:
        status = PROVIDER_REQUIRED
    else:
        status = "UNAVAILABLE"
    return {
        "status": status,
        "source": source,
        "reason": reason,
        "stale": bool(stale),
        "synthetic_data_used": False,
    }


class ComprehensiveIntelligenceEngine:
    """Point-in-time research fusion over canonical crypto infrastructure.

    Round44 persists exactly one replayable snapshot per 15-minute bucket. It
    never creates exchange orders and never fabricates historical provider data.
    """

    def __init__(self, settings) -> None:
        self.settings = settings
        self.crypto = CryptoLibraryBridge(settings.crypto_repo_root)
        self.root = settings.project_root / "output/crypto_ai_swing/round44"
        self.snapshot_root = self.root / "pit_snapshots"
        self.latest_path = self.root / "latest.json"
        self._feature_cache: dict[tuple[Any, ...], tuple[dict[str, Any], dict[str, Any]]] = {}
        self._macro_cache: tuple[float, dict[str, Any]] | None = None
        self._intelligence_cache: tuple[float, list[Any], dict[str, Any]] | None = None
        try:
            from crypto_ai_swing.agents.prospective_context import ProspectiveContextAgent

            self.context_challenger = ProspectiveContextAgent(settings)
        except Exception:
            self.context_challenger = None

    @staticmethod
    def _bucket(observed_at: datetime) -> str:
        return pd.Timestamp(observed_at).floor("15min").isoformat()

    def _canonical_intelligence(
        self, observed_at: datetime
    ) -> tuple[list[Any], dict[str, Any]]:
        now = time.time()
        if self._intelligence_cache and now - self._intelligence_cache[0] <= 300:
            return list(self._intelligence_cache[1]), dict(self._intelligence_cache[2])
        try:
            native = self.crypto.settings()
            path = native.paths.intelligence_dir / "crypto_intelligence.parquet"
            if not path.is_file():
                status = _availability(
                    ready=False,
                    source="Sjagil/crypto:scrapers.intelligence",
                    reason="CANONICAL_INTELLIGENCE_FILE_MISSING",
                )
                rows: list[Any] = []
            else:
                module = self.crypto.import_module("scrapers.intelligence")
                loaded = list(module.load_intelligence(path))
                decision = observed_at.astimezone(UTC)
                rows = []
                for record in loaded:
                    usable = getattr(record, "usable_at", None)
                    if usable is None:
                        continue
                    ts = pd.Timestamp(usable)
                    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
                    if ts.to_pydatetime() <= decision:
                        rows.append(record)
                status = {
                    **_availability(
                        ready=bool(rows),
                        source="Sjagil/crypto:scrapers.intelligence",
                        reason=None if rows else "NO_CAUSALLY_USABLE_RECORDS",
                    ),
                    "record_count": len(rows),
                    "path": str(path),
                }
        except Exception as exc:
            rows = []
            status = _availability(
                ready=False,
                source="Sjagil/crypto:scrapers.intelligence",
                reason=f"{type(exc).__name__}:{str(exc)[:220]}",
            )
        self._intelligence_cache = (now, list(rows), dict(status))
        return rows, status

    def _canonical_feature_snapshot(
        self,
        market: str,
        timeframe: str,
        frame: pd.DataFrame | None,
        *,
        benchmark: pd.DataFrame | None,
        intelligence: list[Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if frame is None or frame.empty:
            return (
                {
                    "status": "NO_DATA",
                    "feature_count": 0,
                    "feature_hash": None,
                    "timeframe": timeframe,
                    "synthetic_data_used": False,
                },
                {},
            )
        latest = pd.Timestamp(frame.index[-1])
        benchmark_latest = (
            pd.Timestamp(benchmark.index[-1]).isoformat()
            if benchmark is not None and not benchmark.empty
            else "NONE"
        )
        key = (
            market,
            timeframe,
            latest.isoformat(),
            benchmark_latest,
            len(intelligence),
        )
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached

        module = self.crypto.import_module("research.features")
        selected = frame.copy()
        selected.attrs.update(
            {
                "market": market,
                "timeframe": "1W" if timeframe == "1w" else timeframe,
                "data_provenance": {
                    "source_type": "REAL_PROVIDER_DATA",
                    "synthetic_data_used": False,
                },
            }
        )
        benchmark_selected = None
        if benchmark is not None and not benchmark.empty:
            benchmark_selected = benchmark.loc[benchmark.index <= selected.index.max()].copy()
            benchmark_selected.attrs.update(
                {
                    "market": "BTC-EUR",
                    "timeframe": "1W" if timeframe == "1w" else timeframe,
                    "data_provenance": {
                        "source_type": "REAL_PROVIDER_DATA",
                        "synthetic_data_used": False,
                    },
                }
            )
        pipeline = module.FeaturePipeline(
            include_optional_garch=False,
            include_advanced_fractal_estimators=False,
        )
        try:
            features = pipeline.build(
                selected,
                market=market,
                benchmark=benchmark_selected,
                intelligence=intelligence or None,
            )
        except Exception as exc:
            result = (
                {
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}:{str(exc)[:240]}",
                    "feature_count": 0,
                    "feature_hash": None,
                    "timeframe": timeframe,
                    "synthetic_data_used": False,
                },
                {},
            )
            self._feature_cache[key] = result
            return result

        row = features.iloc[-1]
        values: dict[str, Any] = {}
        for name, value in row.items():
            if isinstance(value, (bool, np.bool_)):
                values[str(name)] = bool(value)
            elif (number := _finite(value)) is not None:
                values[str(name)] = number
        metadata = {
            "status": "READY",
            "timeframe": timeframe,
            "latest_bar": latest.isoformat(),
            "feature_count": len(values),
            "feature_hash": _hash(values),
            "benchmark": "BTC-EUR" if benchmark_selected is not None else None,
            "benchmark_relative_features_available": bool(
                any("benchmark_" in str(name) or "relative" in str(name).lower() for name in values)
            ),
            "lookahead_safe": True,
            "closed_candles_only": True,
            "research_labels_excluded": list(
                features.attrs.get("research_labels_excluded") or []
            ),
            "synthetic_data_used": False,
        }
        result = (metadata, values)
        self._feature_cache[key] = result
        if len(self._feature_cache) > 2000:
            for old in list(self._feature_cache)[:750]:
                self._feature_cache.pop(old, None)
        return result

    def _rest_l2(self, bundle: Mapping[str, Any]) -> dict[str, Any]:
        orderbook = dict(bundle.get("orderbook") or {})
        ticker = dict(bundle.get("ticker") or {})
        trades = list(bundle.get("trades") or [])
        micro = dict(bundle.get("microstructure") or {})
        module = self.crypto.import_module("data.orderbook_l2")
        book = module.Level2OrderBook(
            provider="bitvavo",
            market=str(bundle.get("market") or "UNKNOWN"),
            maximum_depth=1000,
        )
        bids = orderbook.get("bids") or orderbook.get("book_bids") or []
        asks = orderbook.get("asks") or orderbook.get("book_asks") or []
        try:
            self.crypto._run(book.initialize(bids=bids, asks=asks))
        except Exception as exc:
            return {
                "status": "UNAVAILABLE",
                "error": f"{type(exc).__name__}:{str(exc)[:200]}",
                "l1": {},
                "l2": {},
                "l3_status": L3_STATUS,
                "synthetic_l3_used": False,
            }

        mid = _finite(book.mid_price)
        l2: dict[str, Any] = {
            "spread_bps": _finite(book.spread_bps),
            "microprice": _finite(book.microprice),
            "top_level_imbalance": _finite(book.top_level_imbalance),
            "book_pressure": _finite(book.book_pressure),
            "bid_depth_base": _finite(book.cumulative_bid_depth),
            "ask_depth_base": _finite(book.cumulative_ask_depth),
            "bid_levels": len(book.bids),
            "ask_levels": len(book.asks),
        }
        for bps in (5, 10, 25, 50, 100):
            l2[f"depth_imbalance_{bps}bps"] = _finite(book.depth_imbalance(bps))
        if mid and mid > 0:
            for notional in (10.0, 50.0, 100.0):
                quantity = Decimal(str(notional / mid))
                for side in ("buy", "sell"):
                    l2[
                        f"estimated_{side}_slippage_{int(notional)}eur_fraction"
                    ] = _finite(book.estimated_slippage(side=side, quantity=quantity))
                    l2[
                        f"estimated_{side}_impact_{int(notional)}eur_fraction"
                    ] = _finite(book.estimated_market_impact(side=side, quantity=quantity))

        l1 = {
            "best_bid": _finite(book.best_bid),
            "best_ask": _finite(book.best_ask),
            "mid_price": mid,
            "last_price": _finite(
                ticker.get("price") or ticker.get("last") or ticker.get("lastPrice")
            ),
            "spread_bps": _finite(book.spread_bps),
            "trade_count": int(micro.get("trade_count") or len(trades)),
            "buy_volume": _finite(micro.get("buy_volume")),
            "sell_volume": _finite(micro.get("sell_volume")),
            "buy_notional": _finite(micro.get("buy_notional")),
            "sell_notional": _finite(micro.get("sell_notional")),
            "cvd_ratio": _finite(micro.get("cvd_ratio")),
            "cvd_notional_ratio": _finite(micro.get("cvd_notional_ratio")),
        }
        if l1["buy_volume"] is not None and l1["sell_volume"] is not None:
            l1["trade_delta_base"] = l1["buy_volume"] - l1["sell_volume"]
        if l1["buy_notional"] is not None and l1["sell_notional"] is not None:
            l1["trade_delta_quote"] = l1["buy_notional"] - l1["sell_notional"]
        return {
            "status": "READY",
            "source": "BITVAVO_REST_L1_L2_SNAPSHOT",
            "sequence_tracked": False,
            "l1": l1,
            "l2": l2,
            "l3_status": L3_STATUS,
            "individual_order_ids_available": False,
            "synthetic_l3_used": False,
        }

    def _mdpro(self) -> dict[str, Any]:
        try:
            native = self.crypto.settings()
            path = native.paths.context_data_dir / "mdpro/latest.json"
            if not path.is_file():
                return {}
            payload = json.loads(path.read_text(encoding="utf-8"))
            generated = pd.Timestamp(payload.get("generated_at"))
            generated = generated.tz_localize("UTC") if generated.tzinfo is None else generated.tz_convert("UTC")
            age = max(0.0, (pd.Timestamp.now(tz="UTC") - generated).total_seconds())
            payload["age_seconds"] = age
            payload["stale"] = age > 30.0
            return payload
        except Exception as exc:
            return {"status": "ERROR", "error": f"{type(exc).__name__}:{str(exc)[:200]}"}

    def _market_mechanics(self, markets: list[str]) -> dict[str, Any]:
        try:
            module = self.crypto.import_module("core.market_mechanics")
            return dict(
                module.build_market_mechanics_snapshot(
                    self.crypto.settings(), markets=tuple(markets), write_artifact=False
                )
            )
        except Exception as exc:
            return {
                "status": "UNAVAILABLE",
                "error": f"{type(exc).__name__}:{str(exc)[:240]}",
                "markets": {},
            }

    def _macro(self) -> dict[str, Any]:
        now = time.time()
        if self._macro_cache and now - self._macro_cache[0] <= 3600:
            return dict(self._macro_cache[1])
        try:
            module = self.crypto.import_module("core.active_trading")
            refresh = getattr(module, "refresh_public_macro_context", None)
            build = getattr(module, "build_crypto_macro_snapshot", None)
            refreshed = (
                self.crypto._run(refresh(self.crypto.settings())) if callable(refresh) else {}
            )
            payload = (
                dict(build(self.crypto.settings(), refreshed=refreshed))
                if callable(build)
                else {
                    "status": PROVIDER_REQUIRED,
                    "reason": "CANONICAL_MACRO_BUILDER_UNAVAILABLE",
                }
            )
        except Exception as exc:
            payload = {
                "status": PROVIDER_REQUIRED,
                "error": f"{type(exc).__name__}:{str(exc)[:240]}",
                "synthetic_data_used": False,
            }
        payload.setdefault("synthetic_data_used", False)
        self._macro_cache = (now, dict(payload))
        return payload

    def _indicator_coverage(self) -> dict[str, Any]:
        try:
            module = self.crypto.import_module("research.indicator_registry")
            report = dict(module.indicator_coverage_report())
            report["registry_hash"] = _hash(report)
            return report
        except Exception as exc:
            return {
                "status": "UNAVAILABLE",
                "error": f"{type(exc).__name__}:{str(exc)[:220]}",
                "registry_hash": None,
            }

    @staticmethod
    def _rich_mdpro_l1_l2(
        manager_row: Mapping[str, Any],
        realtime: Mapping[str, Any],
        rest: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        rest_l1 = dict(rest.get("l1") or {})
        rest_l2 = dict(rest.get("l2") or {})
        ticker = dict(manager_row.get("ticker") or {})
        l1 = dict(rest_l1)
        replacements = {
            "best_bid": ticker.get("best_bid"),
            "best_ask": ticker.get("best_ask"),
            "best_bid_size": ticker.get("best_bid_size"),
            "best_ask_size": ticker.get("best_ask_size"),
            "last_price": ticker.get("last_price"),
            "spread_bps": manager_row.get("spread_bps"),
            "microprice": manager_row.get("microprice"),
        }
        for key, value in replacements.items():
            number = _finite(value)
            if number is not None:
                l1[key] = number
        for key, value in _flatten_numeric("", realtime, maximum=160).items():
            clean = key.lstrip(".")
            lower = clean.lower()
            if any(token in lower for token in ("trade", "cvd", "spread", "best_bid", "best_ask")):
                l1[clean] = value

        l2 = dict(rest_l2)
        for key, value in _flatten_numeric("", manager_row, maximum=160).items():
            clean = key.lstrip(".")
            lower = clean.lower()
            if any(token in lower for token in ("depth", "imbalance", "pressure", "microprice", "level", "spread")):
                l2[clean] = value
        for key, value in _flatten_numeric("", realtime, maximum=256).items():
            clean = key.lstrip(".")
            lower = clean.lower()
            if any(
                token in lower
                for token in (
                    "ofi",
                    "order_flow",
                    "imbalance",
                    "liquidity",
                    "depth",
                    "gap",
                    "slippage",
                    "impact",
                    "microprice",
                    "book",
                    "resilien",
                    "spread",
                    "sequence",
                    "cvd",
                    "absorption",
                )
            ):
                l2[clean] = value
        return dict(sorted(l1.items())), dict(sorted(l2.items()))

    @staticmethod
    def _learning_vector(
        *,
        technical_values: Mapping[str, Mapping[str, Any]],
        l1: Mapping[str, Any],
        l2: Mapping[str, Any],
        mechanics: Mapping[str, Any],
        macro: Mapping[str, Any],
        cmc_global: Mapping[str, Any],
        cmc_asset: Mapping[str, Any],
        news_nlp: Mapping[str, Any],
        regime: Mapping[str, Any],
        supervised: Mapping[str, Any],
        rl: Mapping[str, Any],
    ) -> dict[str, float]:
        output: dict[str, float] = {}
        technical_tokens = (
            "return",
            "relative",
            "rsi",
            "macd",
            "stoch",
            "atr",
            "volatility",
            "volume",
            "vwap",
            "mfi",
            "sma",
            "ema",
            "supertrend",
            "donchian",
            "aroon",
            "keltner",
            "bollinger",
            "hammer",
            "doji",
            "engulf",
            "harami",
            "inside",
            "outside",
            "morning",
            "evening",
            "soldier",
            "crow",
            "pin",
            "marubozu",
            "fractal",
            "breakout",
            "breakdown",
            "sweep",
            "bos",
            "choch",
            "structure",
            "hurst",
            "entropy",
            "drawdown",
        )
        for timeframe in TIMEFRAMES:
            values = dict(technical_values.get(timeframe) or {})
            picked = 0
            for name in sorted(values):
                lower = str(name).lower()
                if not any(token in lower for token in technical_tokens):
                    continue
                number = _finite(values[name])
                if number is None:
                    continue
                output[f"tech.{timeframe}.{name}"] = number
                picked += 1
                if picked >= 72:
                    break
        for prefix, payload, maximum in (
            ("l1", l1, 80),
            ("l2", l2, 180),
            ("mechanics", mechanics, 120),
            ("macro", macro, 120),
            ("cmc.global", cmc_global, 80),
            ("cmc.asset", cmc_asset, 80),
            ("news", news_nlp, 80),
            ("regime", regime, 100),
            ("supervised", supervised, 60),
            ("rl", rl, 40),
        ):
            output.update(_flatten_numeric(prefix, payload, maximum=maximum))
        return dict(sorted(output.items()))

    def build(
        self,
        *,
        markets: list[str],
        frames: Mapping[str, Mapping[str, pd.DataFrame]],
        bundles: Mapping[str, Mapping[str, Any]],
        context: Mapping[str, Mapping[str, Any]],
        cmc_context: Mapping[str, Any],
        news: Mapping[str, Any],
        observed_at: datetime,
    ) -> dict[str, Any]:
        selected_markets = list(dict.fromkeys(str(m).upper() for m in markets))[:25]
        bucket = self._bucket(observed_at)
        intelligence, intelligence_status = self._canonical_intelligence(observed_at)
        mechanics = self._market_mechanics(selected_markets)
        macro = self._macro()
        mdpro = self._mdpro()
        indicator_coverage = self._indicator_coverage()
        manager = dict(mdpro.get("manager") or {}) if isinstance(mdpro, Mapping) else {}
        mdpro_markets = dict(manager.get("markets") or {})
        realtime_rows = {
            str(row.get("market")).upper(): dict(row)
            for row in (
                ((mdpro.get("microstructure") or {}).get("markets") or [])
                if isinstance(mdpro, Mapping)
                else []
            )
            if isinstance(row, Mapping) and row.get("market")
        }
        mechanics_markets = dict(mechanics.get("markets") or {})
        cmc_assets = dict(cmc_context.get("runtime_assets") or {})
        cmc_global = {
            str(key): value
            for key, value in cmc_context.items()
            if str(key) != "runtime_assets"
        }

        full_artifact: dict[str, Any] = {
            "schema_version": SCHEMA,
            "observed_at": observed_at.astimezone(UTC).isoformat(),
            "decision_bucket": bucket,
            "repo_versions": {
                "swing": _git_sha(Path(self.settings.project_root)),
                "canonical_crypto": _git_sha(Path(self.settings.crypto_repo_root)),
            },
            "markets": {},
            "macro": macro,
            "cmc_context": dict(cmc_context),
            "news_source_health": dict(news),
            "canonical_intelligence": intelligence_status,
            "indicator_coverage": indicator_coverage,
            "l3_status": L3_STATUS,
            "synthetic_data_used": False,
        }
        compact_markets: dict[str, Any] = {}
        technical_ready = 0
        l1_l2_snapshot_ready = 0
        mdpro_ready = 0
        all_six_ready = 0

        for market in selected_markets:
            market_frames = dict(frames.get(market) or {})
            technical_meta: dict[str, Any] = {}
            technical_values: dict[str, dict[str, Any]] = {}
            for timeframe in TIMEFRAMES:
                frame = market_frames.get(timeframe)
                if frame is None:
                    frame = market_frames.get("1W" if timeframe == "1w" else timeframe)
                btc_frames = dict(frames.get("BTC-EUR") or {})
                benchmark = btc_frames.get(timeframe)
                if benchmark is None:
                    benchmark = btc_frames.get("1W" if timeframe == "1w" else timeframe)
                meta, values = self._canonical_feature_snapshot(
                    market,
                    timeframe,
                    frame,
                    benchmark=benchmark,
                    intelligence=intelligence,
                )
                technical_meta[timeframe] = meta
                technical_values[timeframe] = values
            ready_timeframes = sum(
                row.get("status") == "READY" for row in technical_meta.values()
            )
            if ready_timeframes:
                technical_ready += 1
            if ready_timeframes == len(TIMEFRAMES):
                all_six_ready += 1

            rest = self._rest_l2({"market": market, **dict(bundles.get(market) or {})})
            mdpro_row = dict(mdpro_markets.get(market) or {})
            realtime = dict(realtime_rows.get(market) or {})
            sequence_valid = realtime.get("sequence_valid") is True
            mdpro_usable = bool(
                not mdpro.get("stale")
                and mdpro_row.get("status") == "READY"
                and sequence_valid
            )
            l1, l2 = self._rich_mdpro_l1_l2(mdpro_row, realtime, rest)
            snapshot_ready = bool(rest.get("status") == "READY" or mdpro_row.get("status") == "READY")
            if snapshot_ready:
                l1_l2_snapshot_ready += 1
            if mdpro_usable:
                mdpro_ready += 1
            market_data = {
                "preferred_source": "BITVAVO_WS_MARKET_DATA_PRO" if mdpro_usable else "BITVAVO_REST_SNAPSHOT",
                "stream_status": "READY" if mdpro_usable else ("STALE" if mdpro.get("stale") else PROVIDER_REQUIRED),
                "sequence_valid": sequence_valid,
                "sequence": mdpro_row.get("sequence"),
                "l1": l1,
                "l2": l2,
                "mdpro_raw": {
                    "manager": mdpro_row,
                    "realtime": realtime,
                    "stale": bool(mdpro.get("stale", False)),
                    "age_seconds": mdpro.get("age_seconds"),
                },
                "rest_snapshot": rest,
                "l3_status": L3_STATUS,
                "individual_order_ids_available": False,
                "synthetic_l3_used": False,
            }

            decision = dict(context.get(market) or {})
            supervised = dict(decision.get("agents") or {})
            for key in ("ml_probability", "forecast_score", "predicted_return", "predicted_mae"):
                if key in decision:
                    supervised[key] = decision.get(key)
            rl = dict(decision.get("rl") or {})
            if "rl_score" in decision:
                rl["score"] = decision.get("rl_score")
            news_nlp = {
                "source_health": dict(news),
                "nlp_score": decision.get("nlp_score"),
                "nlp_confidence": decision.get("nlp_confidence"),
                "nlp_severe_negative": decision.get("nlp_severe_negative"),
                "nlp_model": decision.get("nlp_model"),
                "nlp_diagnostics": decision.get("nlp_diagnostics"),
                "event_tags": decision.get("event_tags") or [],
            }
            regime = {
                "timeframe_pipeline": decision.get("timeframe_pipeline"),
                "technical_intelligence": decision.get("technical_intelligence"),
                "mtf_score": decision.get("mtf_score"),
                "cmc": decision.get("cmc"),
                "macro_blocked": decision.get("macro_blocked"),
                "entry_blocked": decision.get("entry_blocked"),
            }
            mechanics_row = dict(mechanics_markets.get(market) or {})
            cmc_row = dict(cmc_assets.get(market) or {})
            vector = self._learning_vector(
                technical_values=technical_values,
                l1=l1,
                l2=l2,
                mechanics=mechanics_row,
                macro=macro,
                cmc_global=cmc_global,
                cmc_asset=cmc_row,
                news_nlp=news_nlp,
                regime=regime,
                supervised=supervised,
                rl=rl,
            )
            challenger = (
                self.context_challenger.predict_vector(vector)
                if self.context_challenger is not None
                else {"status": "UNAVAILABLE", "score": None, "live_decision_influence": False}
            )
            component_hashes = {
                "technical": _hash(technical_values),
                "l1": _hash(l1),
                "l2": _hash(l2),
                "macro": _hash(macro),
                "cmc": _hash({"global": cmc_global, "asset": cmc_row}),
                "news_nlp": _hash(news_nlp),
                "regime": _hash(regime),
                "supervised": _hash(supervised),
                "rl": _hash(rl),
                "learning_vector": _hash(vector),
                "context_challenger": _hash(challenger),
            }
            data_completeness = {
                "technical": {
                    tf: _availability(
                        ready=technical_meta[tf].get("status") == "READY",
                        source="Sjagil/crypto:research.features.FeaturePipeline",
                        reason=technical_meta[tf].get("error") or technical_meta[tf].get("status"),
                    )
                    for tf in TIMEFRAMES
                },
                "l1": _availability(
                    ready=bool(l1),
                    source=market_data["preferred_source"],
                    reason=None if l1 else "NO_L1_DATA",
                    provider_required=not bool(l1),
                ),
                "l2_snapshot": _availability(
                    ready=snapshot_ready,
                    source=market_data["preferred_source"],
                    reason=None if snapshot_ready else "NO_L2_SNAPSHOT",
                    provider_required=not snapshot_ready,
                ),
                "l2_stream": _availability(
                    ready=mdpro_usable,
                    source="BITVAVO_MARKET_DATA_PRO",
                    reason=None if mdpro_usable else "MDPRO_NOT_READY_OR_SEQUENCE_INVALID",
                    stale=bool(mdpro.get("stale", False)),
                    provider_required=not bool(mdpro),
                ),
                "l3": _availability(
                    ready=False,
                    source="BITVAVO",
                    reason=L3_STATUS,
                    unsupported=True,
                ),
                "macro": _availability(
                    ready=str(macro.get("status") or "").upper() not in {"", "UNAVAILABLE", PROVIDER_REQUIRED},
                    source="Sjagil/crypto:core.active_trading",
                    reason=macro.get("error") or macro.get("reason"),
                    provider_required=str(macro.get("status") or "").upper() in {"UNAVAILABLE", PROVIDER_REQUIRED},
                ),
                "cmc": _availability(
                    ready=bool(cmc_row) and bool(cmc_context.get("global_metrics")),
                    source="CMC/canonical DataLoader",
                    reason=None if cmc_row else "CMC_ASSET_CONTEXT_MISSING",
                    provider_required=not bool(cmc_row),
                ),
                "news_nlp": _availability(
                    ready=decision.get("nlp_score") is not None,
                    source="Sjagil/crypto:web+RSS+NLP",
                    reason=None if decision.get("nlp_score") is not None else "NO_CAUSAL_NLP_ASSESSMENT",
                ),
                "supervised": _availability(
                    ready=bool(supervised),
                    source="crypto-ai-swing-layer:AgentRuntime",
                    reason=None if supervised else "NO_SUPERVISED_RUNTIME_OUTPUT",
                ),
                "rl": _availability(
                    ready=bool(rl),
                    source="crypto-ai-swing-layer:RLRuntime",
                    reason=None if rl else "NO_RL_RUNTIME_OUTPUT",
                ),
            }
            full_row = {
                "market": market,
                "technical": {
                    "required_timeframes": list(TIMEFRAMES),
                    "ready_timeframes": ready_timeframes,
                    "timeframes": technical_meta,
                    "values": technical_values,
                },
                "market_data": market_data,
                "market_mechanics": mechanics_row,
                "macro": macro,
                "cmc_fundamental": cmc_row,
                "cmc_global": cmc_global,
                "news_nlp": news_nlp,
                "regime": regime,
                "supervised": supervised,
                "rl": rl,
                "context_challenger": challenger,
                "learning_vector": vector,
                "outcome_horizons_hours": [1, 4, 24, 72, 168],
                "outcomes_status": "PENDING_MATURATION_IN_FORWARD_LEDGER",
                "component_hashes": component_hashes,
                "data_completeness": data_completeness,
                "l3_status": L3_STATUS,
                "synthetic_data_used": False,
            }
            full_artifact["markets"][market] = full_row
            compact_markets[market] = {
                "schema_version": SCHEMA,
                "snapshot_bucket": bucket,
                "feature_pipeline_version": "Sjagil/crypto:research.features",
                "required_timeframes": list(TIMEFRAMES),
                "technical_timeframes": technical_meta,
                "market_data": {
                    "preferred_source": market_data["preferred_source"],
                    "stream_status": market_data["stream_status"],
                    "sequence_valid": market_data["sequence_valid"],
                    "l1": l1,
                    "l2": l2,
                    "l3_status": L3_STATUS,
                },
                "data_completeness": data_completeness,
                "component_hashes": component_hashes,
                "learning_vector": vector,
                "learning_vector_hash": component_hashes["learning_vector"],
                "context_challenger": challenger,
                "outcome_horizons_hours": [1, 4, 24, 72, 168],
                "outcomes_status": "PENDING_MATURATION_IN_FORWARD_LEDGER",
                "l3_status": L3_STATUS,
                "lookahead_safe_design": True,
                "synthetic_data_used": False,
                "live_decision_influence": False,
            }

        body_hash = _hash(full_artifact)
        full_artifact["snapshot_hash"] = body_hash
        day = pd.Timestamp(observed_at).strftime("%Y/%m/%d")
        name = pd.Timestamp(bucket).strftime("%Y%m%dT%H%M00Z") + ".json.gz"
        target = self.snapshot_root / day / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            with gzip.open(target, "wt", encoding="utf-8") as fh:
                json.dump(_json_safe(full_artifact), fh, sort_keys=True)

        summary = {
            "schema_version": SCHEMA,
            "generated_at": observed_at.astimezone(UTC).isoformat(),
            "decision_bucket": bucket,
            "snapshot_hash": body_hash,
            "snapshot_path": str(target),
            "requested_markets": len(selected_markets),
            "technical_ready_markets": technical_ready,
            "six_timeframe_ready_markets": all_six_ready,
            "l1_l2_ready_markets": l1_l2_snapshot_ready,
            "mdpro_ready_markets": mdpro_ready,
            "market_mechanics_status": mechanics.get("status"),
            "macro_status": macro.get("status"),
            "news_status": news.get("status"),
            "canonical_intelligence_status": intelligence_status.get("status"),
            "indicator_registry": indicator_coverage,
            "l1": "BITVAVO_NATIVE",
            "l2": "BITVAVO_NATIVE_AGGREGATED_PRICE_LEVELS",
            "l3": L3_STATUS,
            "all_25_six_timeframe_technical_ready": len(selected_markets) == 25 and all_six_ready == 25,
            "all_25_l1_l2_snapshot_ready": len(selected_markets) == 25 and l1_l2_snapshot_ready == 25,
            "all_25_mdpro_stream_ready": len(selected_markets) == 25 and mdpro_ready == 25,
            "synthetic_data_used": False,
            "automatic_live_authority": False,
            "automatic_live_promotion": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_path.write_text(
            json.dumps(
                {"summary": summary, "markets": compact_markets},
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        return {"summary": summary, "markets": compact_markets}
