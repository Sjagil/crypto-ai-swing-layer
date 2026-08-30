from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import json
import sqlite3
import time
from typing import Any

import numpy as np
import pandas as pd

from crypto_ai_swing.bridge.crypto_library import (
    CryptoLibraryBridge,
    CryptoLibraryError,
)
from crypto_ai_swing.contracts import Authority, TradeIntent
from crypto_ai_swing.execution.bitvavo import (
    BitvavoREST,
    BitvavoError,
    live_gate_status,
)
from crypto_ai_swing.nlp.engine import NLPMarketEngine
from crypto_ai_swing.nlp.sources import (
    discover_crypto_repo_documents,
    fetch_rss_documents,
)
from crypto_ai_swing.intelligence.crypto_news import CryptoNewsCollector
from crypto_ai_swing.research.forward import ForwardEvidenceLedger
from crypto_ai_swing.orchestration.pipeline import SwingPipeline


@dataclass(frozen=True)
class Position:
    market: str
    amount: Decimal
    entry_price: Decimal
    highest_price: Decimal
    stop_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    opened_at: str


class ProactiveState:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                market TEXT NOT NULL,
                candle_ts TEXT NOT NULL,
                action TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (market, candle_ts, action)
            );
            CREATE TABLE IF NOT EXISTS positions (
                market TEXT PRIMARY KEY,
                amount TEXT NOT NULL,
                entry_price TEXT NOT NULL,
                highest_price TEXT NOT NULL,
                stop_pct REAL NOT NULL,
                take_profit_pct REAL NOT NULL,
                trailing_stop_pct REAL NOT NULL,
                opened_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent_id TEXT,
                market TEXT NOT NULL,
                side TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def close(self):
        self.conn.close()

    def seen(self, market: str, candle_ts: str, action: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM decisions WHERE market=? AND candle_ts=? AND action=?",
            (market, candle_ts, action),
        ).fetchone()
        return bool(row)

    def mark(
        self,
        market: str,
        candle_ts: str,
        action: str,
        payload: dict,
    ):
        self.conn.execute(
            "INSERT OR IGNORE INTO decisions VALUES (?,?,?,?,?)",
            (
                market,
                candle_ts,
                action,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(payload, sort_keys=True, default=str),
            ),
        )
        self.conn.commit()

    def positions(self) -> dict[str, Position]:
        rows = self.conn.execute(
            "SELECT market,amount,entry_price,highest_price,stop_pct,"
            "take_profit_pct,trailing_stop_pct,opened_at FROM positions"
        ).fetchall()
        return {
            r[0]: Position(
                r[0],
                Decimal(r[1]),
                Decimal(r[2]),
                Decimal(r[3]),
                float(r[4]),
                float(r[5]),
                float(r[6]),
                r[7],
            )
            for r in rows
        }

    def upsert_position(self, pos: Position):
        self.conn.execute(
            "INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?,?)",
            (
                pos.market,
                str(pos.amount),
                str(pos.entry_price),
                str(pos.highest_price),
                pos.stop_pct,
                pos.take_profit_pct,
                pos.trailing_stop_pct,
                pos.opened_at,
            ),
        )
        self.conn.commit()

    def delete_position(self, market: str):
        self.conn.execute(
            "DELETE FROM positions WHERE market=?", (market,)
        )
        self.conn.commit()

    def record_order(
        self,
        intent_id: str | None,
        market: str,
        side: str,
        payload: dict,
    ):
        self.conn.execute(
            "INSERT INTO orders(intent_id,market,side,created_at,payload) "
            "VALUES (?,?,?,?,?)",
            (
                intent_id,
                market,
                side,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(payload, sort_keys=True, default=str),
            ),
        )
        self.conn.commit()


class ProactiveTrader:
    def __init__(self, settings, mode: str = "shadow"):
        self.settings = settings
        self.mode = mode.lower()
        state_rel = settings.proactive.get(
            "state_path",
            "output/crypto_ai_swing/proactive/state.sqlite",
        )
        self.state = ProactiveState(settings.project_root / state_rel)
        self.nlp = NLPMarketEngine(settings.nlp)
        self.crypto = CryptoLibraryBridge(settings.crypto_repo_root)
        # Direct REST is retained only for the already-existing private
        # account preflight/bootstrap. Round 2 market data is sourced from
        # Sjagil/crypto. Live order submission is intentionally disabled
        # until core.execution_authority is mapped exactly.
        self.client = BitvavoREST()
        news_cfg = settings.autonomy.get("news", {}) if hasattr(settings, "autonomy") else {}
        self.news_collector = CryptoNewsCollector(
            self.crypto,
            output_path=(
                settings.project_root
                / "output/crypto_ai_swing/intelligence/news.jsonl"
            ),
            cache_seconds=float(news_cfg.get("cache_seconds", 300)),
            maximum_documents=int(news_cfg.get("maximum_documents", 500)),
        )
        forward_cfg = settings.autonomy.get("forward_evidence", {}) if hasattr(settings, "autonomy") else {}
        forward_rel = forward_cfg.get(
            "path", "output/crypto_ai_swing/forward/forward.sqlite"
        )
        self.forward = ForwardEvidenceLedger(settings.project_root / forward_rel)
        self._news_cache: tuple[float, list] | None = None
        self._last_news_status: dict[str, Any] = {
            "status": "NOT_RUN",
            "source": "Sjagil/crypto:scrapers.rss",
            "source_statuses": [],
        }

    def close(self):
        self.state.close()
        self.forward.close()
        self.client.close()

    def _markets(self) -> list[str]:
        import os

        raw = os.getenv("CRYPTO_SWING_MARKETS")
        if raw:
            return [
                x.strip().upper()
                for x in raw.split(",")
                if x.strip()
            ]
        values = self.settings.proactive.get(
            "markets", ["BTC-EUR", "ETH-EUR", "SOL-EUR"]
        )
        return [str(x).upper() for x in values]

    def _fallback_account(self) -> tuple[Decimal, Decimal, Decimal]:
        fallback = Decimal(
            str(
                self.settings.proactive.get(
                    "shadow_equity_eur", 10000
                )
            )
        )
        return fallback, fallback, Decimal("0")

    def _private_account_enabled(self) -> bool:
        account_cfg = self.settings.proactive.get("account", {}) or {}
        if self.mode == "live":
            return True
        return bool(
            account_cfg.get(
                f"use_private_balances_in_{self.mode}",
                False,
            )
        )

    def _account(
        self, markets: list[str]
    ) -> tuple[Decimal, Decimal, Decimal]:
        if not self._private_account_enabled():
            return self._fallback_account()
        if not self.client.api_key or not self.client.api_secret:
            if self.mode == "live":
                raise BitvavoError(
                    "Private Bitvavo credentials are required in live mode"
                )
            return self._fallback_account()
        try:
            balances = self.client.balances()
        except BitvavoError:
            if self.mode == "live":
                raise
            return self._fallback_account()
        by_symbol = {
            str(x.get("symbol")): (
                Decimal(str(x.get("available", "0")))
                + Decimal(str(x.get("inOrder", "0")))
            )
            for x in balances
        }
        cash = by_symbol.get("EUR", Decimal("0"))
        equity = cash
        exposure = Decimal("0")
        for market in markets:
            base = market.split("-", 1)[0]
            amount = by_symbol.get(base, Decimal("0"))
            if amount <= 0:
                continue
            try:
                bundle = self.crypto.market_bundle(
                    market,
                    "1h",
                    mode=self.mode,
                    persist=False,
                    depth=5,
                )
                price = Decimal(
                    str(bundle.microstructure.get("best_bid") or "0")
                )
            except Exception:
                price = Decimal("0")
            value = amount * price
            equity += value
            exposure += value
        if equity <= 0:
            if self.mode == "live":
                raise BitvavoError(
                    "Authenticated Bitvavo account returned no usable equity"
                )
            return self._fallback_account()
        return equity, cash, exposure

    def _news(self):
        now = time.time()
        ttl = float(
            self.settings.proactive.get("nlp_cache_seconds", 300)
        )
        if self._news_cache and now - self._news_cache[0] <= ttl:
            return list(self._news_cache[1])
        docs = []
        news_cfg = self.settings.autonomy.get("news", {}) if hasattr(self.settings, "autonomy") else {}
        if bool(news_cfg.get("enabled", True)):
            try:
                snapshot = self.news_collector.collect(
                    persist=bool(news_cfg.get("persist", True))
                )
                docs.extend(snapshot.documents)
                self._last_news_status = {
                    "status": snapshot.status,
                    "source": snapshot.source,
                    "observed_at": snapshot.observed_at,
                    "source_statuses": list(snapshot.source_statuses),
                    "live_documents": len(snapshot.documents),
                }
            except Exception as exc:
                self._last_news_status = {
                    "status": "ERROR",
                    "source": "Sjagil/crypto:scrapers.rss",
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    "source_statuses": [],
                    "live_documents": 0,
                }
        docs.extend(discover_crypto_repo_documents(self.settings.crypto_repo_root))
        rss_urls = list(self.settings.nlp.get("rss_urls", []) or [])
        if rss_urls:
            docs.extend(fetch_rss_documents(rss_urls))
        unique = {}
        for doc in docs:
            key = (doc.source, doc.url or "", doc.title or doc.text[:200], doc.usable_at.isoformat())
            unique[key] = doc
        selected = sorted(unique.values(), key=lambda x: x.usable_at, reverse=True)
        maximum = int(news_cfg.get("maximum_documents", 500))
        selected = selected[:maximum]
        self._news_cache = (now, list(selected))
        return selected

    @staticmethod
    def _avg_fill_price(order: dict[str, Any]) -> Decimal:
        fills = order.get("fills") or []
        total_amount = Decimal("0")
        total_quote = Decimal("0")
        for fill in fills:
            try:
                amount = Decimal(str(fill.get("amount", "0")))
                price = Decimal(str(fill.get("price", "0")))
            except Exception:
                continue
            total_amount += amount
            total_quote += amount * price
        if total_amount > 0:
            return total_quote / total_amount
        filled = Decimal(str(order.get("filledAmount") or "0"))
        quote = Decimal(
            str(order.get("filledAmountQuote") or "0")
        )
        return quote / filled if filled > 0 else Decimal("0")

    def _execute_buy(self, intent: TradeIntent) -> dict[str, Any]:
        if self.mode != "live":
            return {
                "mode": self.mode,
                "simulated": True,
                "intent_id": intent.intent_id,
                "execution_backend": "shadow_or_paper",
            }
        gate = live_gate_status(self.settings.execution)
        if not gate.ready:
            return {
                "mode": "live",
                "accepted": False,
                "blockers": list(gate.blockers),
            }
        # The existing crypto repository owns execution authority. Do not
        # bypass it with the temporary direct Bitvavo client.
        interfaces = self.crypto.execution_authority_interfaces()
        return {
            "mode": "live",
            "accepted": False,
            "blockers": ["CRYPTO_EXECUTION_AUTHORITY_ADAPTER_NOT_MAPPED"],
            "execution_backend": "core.execution_authority",
            "available_interfaces": interfaces,
        }

    def _manage_exits(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for market, pos in list(self.state.positions().items()):
            try:
                bundle = self.crypto.market_bundle(
                    market,
                    "1h",
                    mode=self.mode,
                    persist=False,
                    depth=5,
                )
                price = Decimal(
                    str(
                        bundle.microstructure.get("best_bid")
                        or bundle.microstructure.get("best_ask")
                        or "0"
                    )
                )
            except Exception:
                continue
            if price <= 0:
                continue
            highest = max(pos.highest_price, price)
            if highest != pos.highest_price:
                pos = Position(
                    pos.market,
                    pos.amount,
                    pos.entry_price,
                    highest,
                    pos.stop_pct,
                    pos.take_profit_pct,
                    pos.trailing_stop_pct,
                    pos.opened_at,
                )
                self.state.upsert_position(pos)
            hard_stop = pos.entry_price * (
                Decimal("1") - Decimal(str(pos.stop_pct))
            )
            target = pos.entry_price * (
                Decimal("1") + Decimal(str(pos.take_profit_pct))
            )
            trailing = highest * (
                Decimal("1") - Decimal(str(pos.trailing_stop_pct))
            )
            reason = None
            if price <= hard_stop:
                reason = "STOP_LOSS"
            elif price >= target:
                reason = "TAKE_PROFIT"
            elif highest > pos.entry_price and price <= trailing:
                reason = "TRAILING_STOP"
            if not reason:
                continue
            if self.mode == "live":
                events.append(
                    {
                        "market": market,
                        "action": "EXIT_BLOCKED",
                        "reason": reason,
                        "blockers": [
                            "CRYPTO_EXECUTION_AUTHORITY_ADAPTER_NOT_MAPPED"
                        ],
                    }
                )
                continue
            events.append(
                {
                    "market": market,
                    "action": "SIMULATED_SELL",
                    "reason": reason,
                }
            )
            self.state.delete_position(market)
        return events

    @staticmethod
    def _trend_component(frame: pd.DataFrame) -> float:
        if frame is None or frame.empty or "close" not in frame:
            return 0.0
        close = pd.to_numeric(frame["close"], errors="coerce").dropna()
        if len(close) < 20:
            return 0.0
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = (
            close.ewm(span=50, adjust=False).mean().iloc[-1]
            if len(close) >= 50
            else close.mean()
        )
        last = close.iloc[-1]
        bullish = float(last > ema20) + float(ema20 > ema50)
        bearish = float(last < ema20) + float(ema20 < ema50)
        return float(np.clip((bullish - bearish) / 2.0, -1.0, 1.0))

    @classmethod
    def _mtf_score(cls, frames: dict[str, pd.DataFrame]) -> float:
        weights = {
            "15m": 0.05,
            "1h": 0.15,
            "2h": 0.15,
            "4h": 0.25,
            "1d": 0.25,
            "1w": 0.15,
        }
        total = 0.0
        used = 0.0
        for timeframe, weight in weights.items():
            frame = frames.get(timeframe)
            if frame is None or frame.empty:
                continue
            total += cls._trend_component(frame) * weight
            used += weight
        return float(np.clip(total / used, -1.0, 1.0)) if used else 0.0

    @staticmethod
    def _orderflow_score(micro: dict[str, float]) -> float:
        book = float(micro.get("book_imbalance", 0.0))
        cvd = float(micro.get("cvd_ratio", 0.0))
        micro_edge = float(micro.get("microprice_edge_bps", 0.0))
        edge_scaled = float(np.tanh(micro_edge / 10.0))
        return float(
            np.clip(0.45 * book + 0.35 * cvd + 0.20 * edge_scaled, -1.0, 1.0)
        )

    def _persist_market_data(self) -> bool:
        return bool(
            self.settings.proactive.get(
                f"market_data_persist_{self.mode}",
                self.mode != "shadow",
            )
        )

    def cycle(self) -> dict[str, Any]:
        markets = self._markets()
        exit_events = self._manage_exits()
        docs = self._news()
        frames: dict[str, pd.DataFrame] = {}
        spreads: dict[str, float] = {}
        context: dict[str, dict] = {}
        skipped: list[dict] = []
        mtf_summary: dict[str, dict[str, int]] = {}
        persist = self._persist_market_data()
        timeframes = tuple(
            str(x)
            for x in self.settings.proactive.get(
                "timeframes",
                ["15m", "1h", "2h", "4h", "1d", "1w"],
            )
        )
        primary = str(
            self.settings.proactive.get(
                "primary_signal_timeframe", "1h"
            )
        )
        depth = int(
            self.settings.proactive.get("orderbook_depth", 100)
        )

        library_status = self.crypto.integration_status()
        prospective_context: dict[str, Any] = {}
        prospective_error: str | None = None
        try:
            prospective_context = self.crypto.prospective_context(
                markets,
                cache_seconds=float(
                    self.settings.proactive.get(
                        "prospective_context_cache_seconds", 300
                    )
                ),
            )
        except Exception as exc:
            prospective_error = (
                f"{type(exc).__name__}: {str(exc)[:300]}"
            )
        try:
            micro_readiness = self.crypto.microstructure_readiness()
        except Exception as exc:
            micro_readiness = {
                "status": "UNAVAILABLE",
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            }

        prospective_status = str(
            prospective_context.get("status") or "UNKNOWN"
        ).upper()
        prospective_block = prospective_status in {
            "BLOCK_NEW_ENTRIES",
            "BLOCKED",
            "FAILED",
            "NOT_READY",
        }

        for market in markets:
            try:
                bundle = self.crypto.market_bundle(
                    market,
                    primary,
                    mode=self.mode,
                    persist=persist,
                    depth=depth,
                )
                if bundle.frame.empty:
                    skipped.append(
                        {"market": market, "reason": "NO_CANDLES"}
                    )
                    continue
                mtf_frames = self.crypto.multi_timeframe_frames(
                    market,
                    timeframes,
                    persist=persist,
                )
                mtf_frames[primary] = bundle.frame
                primary_seconds = self.crypto.timeframe_seconds(primary)
                decision_at = bundle.frame.index[-1] + pd.to_timedelta(primary_seconds, unit="s")
                causal_frames = {
                    tf: self.crypto.causal_frame(frame, tf, decision_at)
                    for tf, frame in mtf_frames.items()
                }
                if causal_frames.get(primary) is None or causal_frames[primary].empty:
                    skipped.append({"market": market, "reason": "NO_CAUSAL_PRIMARY_CANDLE"})
                    continue
                mtf = self._mtf_score(causal_frames)
                orderflow = self._orderflow_score(bundle.microstructure)
                quote_volume = self.crypto.quote_volume_24h(
                    bundle.ticker, bundle.microstructure
                )
                frame = causal_frames[primary].copy()
                frame["quote_volume_24h"] = quote_volume
                spreads[market] = float(
                    bundle.microstructure.get("spread_bps", 999.0)
                )
                nlp_aggregation = self.nlp.aggregate_with_diagnostics(docs, market)
                assessment = nlp_aggregation.assessment
                entry_blocked = bool(
                    prospective_block
                    and self.settings.proactive.get(
                        "context_gates", {}
                    ).get("block_on_prospective_context_block", True)
                )
                context[market] = {
                    "nlp_score": assessment.score,
                    "nlp_confidence": assessment.confidence,
                    "nlp_severe_negative": assessment.severe_negative,
                    "event_tags": list(assessment.event_tags),
                    "nlp_model": assessment.model,
                    "nlp_diagnostics": nlp_aggregation.diagnostics,
                    "mtf_score": mtf,
                    "orderflow_score": orderflow,
                    "book_imbalance": bundle.microstructure.get(
                        "book_imbalance", 0.0
                    ),
                    "cvd_ratio": bundle.microstructure.get(
                        "cvd_ratio", 0.0
                    ),
                    "cvd_notional_ratio": bundle.microstructure.get(
                        "cvd_notional_ratio", 0.0
                    ),
                    "buy_volume": bundle.microstructure.get(
                        "buy_volume", 0.0
                    ),
                    "sell_volume": bundle.microstructure.get(
                        "sell_volume", 0.0
                    ),
                    "buy_notional": bundle.microstructure.get(
                        "buy_notional", 0.0
                    ),
                    "sell_notional": bundle.microstructure.get(
                        "sell_notional", 0.0
                    ),
                    "trade_count": bundle.microstructure.get(
                        "trade_count", 0.0
                    ),
                    "classified_trade_count": bundle.microstructure.get(
                        "classified_trade_count", 0.0
                    ),
                    "unclassified_trade_count": bundle.microstructure.get(
                        "unclassified_trade_count", 0.0
                    ),
                    "quote_volume_24h": quote_volume,
                    "decision_at": decision_at.isoformat(),
                    "microprice_edge_bps": bundle.microstructure.get(
                        "microprice_edge_bps", 0.0
                    ),
                    "spread_bps": bundle.microstructure.get(
                        "spread_bps", 999.0
                    ),
                    "prospective_context_status": prospective_status,
                    "entry_blocked": entry_blocked,
                    "data_source": "Sjagil/crypto",
                }
                mtf_summary[market] = {
                    tf: len(causal_frames.get(tf, pd.DataFrame()))
                    for tf in timeframes
                }
                frames[market] = frame
            except Exception as exc:
                skipped.append(
                    {
                        "market": market,
                        "reason": type(exc).__name__,
                        "detail": str(exc)[:300],
                    }
                )

        equity, cash, exposure = self._account(markets)
        authority = (
            Authority.LIVE
            if self.mode == "live"
            else Authority.PAPER
            if self.mode == "paper"
            else Authority.SHADOW
        )
        result = SwingPipeline(self.settings).run(
            frames,
            equity_eur=equity,
            cash_eur=cash,
            exposure_eur=exposure,
            open_risk_eur=Decimal("0"),
            spread_bps=spreads,
            market_context=context,
            authority=authority,
        )

        executions = []
        positions = self.state.positions()
        for intent in result.intents:
            frame = frames.get(intent.market)
            if (
                frame is None
                or frame.empty
                or intent.market in positions
            ):
                continue
            candle_ts = frame.index[-1].isoformat()
            if self.state.seen(intent.market, candle_ts, "BUY"):
                continue
            execution = self._execute_buy(intent)
            self.state.mark(
                intent.market,
                candle_ts,
                "BUY",
                {
                    "intent": intent.to_dict(),
                    "execution": execution,
                },
            )
            executions.append(
                {"intent": intent.to_dict(), "execution": execution}
            )

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "data_source": "Sjagil/crypto python library",
            "markets": markets,
            "equity_eur": str(equity),
            "cash_eur": str(cash),
            "exposure_eur": str(exposure),
            "nlp_documents": len(docs),
            "news": self._last_news_status,
            "crypto_library_ready": library_status.get("ready", False),
            "crypto_library_imported_modules": library_status.get(
                "imported_modules", 0
            ),
            "crypto_library_required_modules": library_status.get(
                "required_modules", 0
            ),
            "prospective_context_status": prospective_status,
            "prospective_context_error": prospective_error,
            "microstructure_readiness": micro_readiness,
            "mtf_rows": mtf_summary,
            "signals": [
                {
                    "market": s.market,
                    "side": s.side.value,
                    "score": s.score,
                    "edge_bps": s.expected_edge_bps,
                    "votes": [v.source for v in s.votes],
                }
                for s in result.signals
            ],
            "market_context": context,
            "blocked": result.blocked,
            "executions": executions,
            "exit_events": exit_events,
            "skipped": skipped,
            "positions": {
                k: {
                    **asdict(v),
                    "amount": str(v.amount),
                    "entry_price": str(v.entry_price),
                    "highest_price": str(v.highest_price),
                }
                for k, v in self.state.positions().items()
            },
        }
        candle_times = {
            market: frame.index[-1].isoformat()
            for market, frame in frames.items()
            if frame is not None and not frame.empty
        }
        forward_cfg = self.settings.autonomy.get("forward_evidence", {}) if hasattr(self.settings, "autonomy") else {}
        should_record = bool(forward_cfg.get("enabled", True)) and bool(
            forward_cfg.get(f"record_{self.mode}", True)
        )
        if should_record:
            payload["forward_evidence"] = self.forward.append_cycle(payload, candle_times)
            if bool(forward_cfg.get("mature_on_cycle", True)):
                horizons = tuple(
                    int(value)
                    for value in (forward_cfg.get("horizons_hours", [1, 4, 24]) or [])
                    if int(value) > 0
                )
                payload["forward_evidence"]["maturation"] = self.forward.mature_from_frames(
                    frames,
                    horizons_hours=horizons or (1, 4, 24),
                )
            payload["forward_evidence"]["ledger"] = self.forward.status()
            payload["forward_evidence"]["outcomes"] = self.forward.outcome_status()
        else:
            payload["forward_evidence"] = {
                "recorded": False,
                "ledger": self.forward.status(),
                "outcomes": self.forward.outcome_status(),
            }

        out = (
            self.settings.project_root
            / "output/crypto_ai_swing/proactive/latest.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        return payload

    def run_forever(self, interval_seconds: int = 60) -> None:
        while True:
            started = time.time()
            try:
                self.cycle()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                out = (
                    self.settings.project_root
                    / "output/crypto_ai_swing/proactive/errors.jsonl"
                )
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "at": datetime.now(timezone.utc).isoformat(),
                                "error": type(exc).__name__,
                                "detail": str(exc)[:1000],
                            }
                        )
                        + "\n"
                    )
            sleep_for = max(
                1.0,
                float(interval_seconds) - (time.time() - started),
            )
            time.sleep(sleep_for)
