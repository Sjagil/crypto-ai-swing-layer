from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import json
import sqlite3
import time
from typing import Any

from crypto_ai_swing.contracts import Authority, TradeIntent
from crypto_ai_swing.execution.bitvavo import BitvavoREST, BitvavoError, live_gate_status
from crypto_ai_swing.nlp.engine import NLPMarketEngine
from crypto_ai_swing.nlp.sources import discover_crypto_repo_documents, fetch_rss_documents
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

    def mark(self, market: str, candle_ts: str, action: str, payload: dict):
        self.conn.execute(
            "INSERT OR IGNORE INTO decisions VALUES (?,?,?,?,?)",
            (market, candle_ts, action, datetime.now(timezone.utc).isoformat(), json.dumps(payload, sort_keys=True, default=str)),
        )
        self.conn.commit()

    def positions(self) -> dict[str, Position]:
        rows = self.conn.execute("SELECT market,amount,entry_price,highest_price,stop_pct,take_profit_pct,trailing_stop_pct,opened_at FROM positions").fetchall()
        return {
            r[0]: Position(r[0], Decimal(r[1]), Decimal(r[2]), Decimal(r[3]), float(r[4]), float(r[5]), float(r[6]), r[7])
            for r in rows
        }

    def upsert_position(self, pos: Position):
        self.conn.execute(
            "INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?,?)",
            (pos.market, str(pos.amount), str(pos.entry_price), str(pos.highest_price), pos.stop_pct, pos.take_profit_pct, pos.trailing_stop_pct, pos.opened_at),
        )
        self.conn.commit()

    def delete_position(self, market: str):
        self.conn.execute("DELETE FROM positions WHERE market=?", (market,))
        self.conn.commit()

    def record_order(self, intent_id: str | None, market: str, side: str, payload: dict):
        self.conn.execute(
            "INSERT INTO orders(intent_id,market,side,created_at,payload) VALUES (?,?,?,?,?)",
            (intent_id, market, side, datetime.now(timezone.utc).isoformat(), json.dumps(payload, sort_keys=True, default=str)),
        )
        self.conn.commit()


class ProactiveTrader:
    def __init__(self, settings, mode: str = "shadow"):
        self.settings = settings
        self.mode = mode.lower()
        state_rel = settings.proactive.get("state_path", "output/crypto_ai_swing/proactive/state.sqlite")
        self.state = ProactiveState(settings.project_root / state_rel)
        self.nlp = NLPMarketEngine(settings.nlp)
        self.client = BitvavoREST()

    def close(self):
        self.state.close()
        self.client.close()

    def _markets(self) -> list[str]:
        import os
        raw = os.getenv("CRYPTO_SWING_MARKETS")
        if raw:
            return [x.strip().upper() for x in raw.split(",") if x.strip()]
        values = self.settings.proactive.get("markets", ["BTC-EUR", "ETH-EUR", "SOL-EUR"])
        return [str(x).upper() for x in values]

    def _fallback_account(self) -> tuple[Decimal, Decimal, Decimal]:
        fallback = Decimal(
            str(self.settings.proactive.get("shadow_equity_eur", 10000))
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

    def _account(self, markets: list[str]) -> tuple[Decimal, Decimal, Decimal]:
        # Shadow and paper must remain runnable without authenticated
        # exchange access. Live always requires actual account truth.
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
                price = Decimal(
                    str(self.client.ticker_book(market).get("bid") or "0")
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
        docs = discover_crypto_repo_documents(self.settings.crypto_repo_root)
        rss_urls = list(self.settings.nlp.get("rss_urls", []) or [])
        if rss_urls:
            docs.extend(fetch_rss_documents(rss_urls))
        return docs

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
        quote = Decimal(str(order.get("filledAmountQuote") or "0"))
        return quote / filled if filled > 0 else Decimal("0")

    def _execute_buy(self, intent: TradeIntent) -> dict[str, Any]:
        if self.mode != "live":
            return {"mode": self.mode, "simulated": True, "intent_id": intent.intent_id}
        gate = live_gate_status(self.settings.execution)
        if not gate.ready:
            return {"mode": "live", "accepted": False, "blockers": list(gate.blockers)}
        order = self.client.place_market_buy(intent.market, intent.notional_eur, client_order_id=intent.intent_id)
        self.state.record_order(intent.intent_id, intent.market, "BUY", order)
        amount = Decimal(str(order.get("filledAmount") or "0"))
        entry = self._avg_fill_price(order)
        if amount > 0 and entry > 0:
            self.state.upsert_position(
                Position(intent.market, amount, entry, entry, intent.stop_pct, intent.take_profit_pct, intent.trailing_stop_pct, datetime.now(timezone.utc).isoformat())
            )
        return order

    def _manage_exits(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        balances = self.client.balances() if self.mode == "live" else []
        available = {str(x.get("symbol")): Decimal(str(x.get("available", "0"))) for x in balances}
        for market, pos in list(self.state.positions().items()):
            try:
                book = self.client.ticker_book(market)
                price = Decimal(str(book.get("bid") or book.get("ask") or "0"))
            except Exception:
                continue
            if price <= 0:
                continue
            highest = max(pos.highest_price, price)
            if highest != pos.highest_price:
                pos = Position(pos.market, pos.amount, pos.entry_price, highest, pos.stop_pct, pos.take_profit_pct, pos.trailing_stop_pct, pos.opened_at)
                self.state.upsert_position(pos)
            hard_stop = pos.entry_price * (Decimal("1") - Decimal(str(pos.stop_pct)))
            target = pos.entry_price * (Decimal("1") + Decimal(str(pos.take_profit_pct)))
            trailing = highest * (Decimal("1") - Decimal(str(pos.trailing_stop_pct)))
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
                gate = live_gate_status(self.settings.execution)
                if not gate.ready:
                    events.append({"market": market, "action": "EXIT_BLOCKED", "reason": reason, "blockers": list(gate.blockers)})
                    continue
                base = market.split("-", 1)[0]
                amount = min(pos.amount, available.get(base, pos.amount))
                if amount <= 0:
                    self.state.delete_position(market)
                    continue
                order = self.client.place_market_sell(market, amount)
                self.state.record_order(None, market, "SELL", {"reason": reason, **order})
                events.append({"market": market, "action": "SELL", "reason": reason, "order": order})
            else:
                events.append({"market": market, "action": "SIMULATED_SELL", "reason": reason})
            self.state.delete_position(market)
        return events

    def cycle(self) -> dict[str, Any]:
        markets = self._markets()
        exit_events = self._manage_exits()
        docs = self._news()
        frames = {}
        spreads = {}
        context = {}
        skipped = []

        for market in markets:
            try:
                frame = self.client.candles(market, interval="1h", limit=int(self.settings.proactive.get("candle_limit", 600)))
                if frame.empty:
                    skipped.append({"market": market, "reason": "NO_CANDLES"})
                    continue
                ticker = self.client.ticker_24h(market)
                quote_volume = float(ticker.get("volumeQuote") or 0.0)
                frame["quote_volume_24h"] = quote_volume
                book = self.client.ticker_book(market)
                bid = float(book.get("bid") or 0.0)
                ask = float(book.get("ask") or 0.0)
                spreads[market] = ((ask - bid) / ((ask + bid) / 2.0) * 10000.0) if bid > 0 and ask > 0 else 999.0
                assessment = self.nlp.aggregate(docs, market)
                context[market] = {
                    "nlp_score": assessment.score,
                    "nlp_confidence": assessment.confidence,
                    "nlp_severe_negative": assessment.severe_negative,
                    "event_tags": list(assessment.event_tags),
                    "nlp_model": assessment.model,
                }
                frames[market] = frame
            except Exception as exc:
                skipped.append({"market": market, "reason": type(exc).__name__, "detail": str(exc)[:240]})

        equity, cash, exposure = self._account(markets)
        authority = Authority.LIVE if self.mode == "live" else Authority.PAPER if self.mode == "paper" else Authority.SHADOW
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
            if frame is None or frame.empty or intent.market in positions:
                continue
            candle_ts = frame.index[-1].isoformat()
            if self.state.seen(intent.market, candle_ts, "BUY"):
                continue
            execution = self._execute_buy(intent)
            self.state.mark(intent.market, candle_ts, "BUY", {"intent": intent.to_dict(), "execution": execution})
            executions.append({"intent": intent.to_dict(), "execution": execution})

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "markets": markets,
            "equity_eur": str(equity),
            "cash_eur": str(cash),
            "exposure_eur": str(exposure),
            "nlp_documents": len(docs),
            "signals": [{"market": s.market, "side": s.side.value, "score": s.score, "edge_bps": s.expected_edge_bps} for s in result.signals],
            "blocked": result.blocked,
            "executions": executions,
            "exit_events": exit_events,
            "skipped": skipped,
            "positions": {k: {**asdict(v), "amount": str(v.amount), "entry_price": str(v.entry_price), "highest_price": str(v.highest_price)} for k, v in self.state.positions().items()},
        }
        out = self.settings.project_root / "output/crypto_ai_swing/proactive/latest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload

    def run_forever(self, interval_seconds: int = 60) -> None:
        while True:
            started = time.time()
            try:
                self.cycle()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                out = self.settings.project_root / "output/crypto_ai_swing/proactive/errors.jsonl"
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "error": type(exc).__name__, "detail": str(exc)[:1000]}) + "\n")
            sleep_for = max(1.0, float(interval_seconds) - (time.time() - started))
            time.sleep(sleep_for)
