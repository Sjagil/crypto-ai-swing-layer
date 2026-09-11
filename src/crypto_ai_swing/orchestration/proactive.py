from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crypto_ai_swing.accounting.paper_ledger import PaperPortfolioLedger
from crypto_ai_swing.agents.edge_manager import ResearchEdgeManager
from crypto_ai_swing.agents.rl_runtime import RLRuntime
from crypto_ai_swing.agents.runtime import AgentRuntime
from crypto_ai_swing.bridge.crypto_library import (
    CryptoLibraryBridge,
)
from crypto_ai_swing.contracts import Authority, TradeIntent
from crypto_ai_swing.execution.active_swing_canary import (
    canonical_preflight_explicitly_denied,
    execution_validation_canary_config,
)
from crypto_ai_swing.execution.crypto_authority import CryptoAuthorityAdapter
from crypto_ai_swing.intelligence.cmc_context import CMCContextCollector
from crypto_ai_swing.intelligence.crypto_news import CryptoNewsCollector
from crypto_ai_swing.intelligence.technical import (
    multi_timeframe_snapshot,
    technical_snapshot,
)
from crypto_ai_swing.nlp.engine import NLPMarketEngine
from crypto_ai_swing.nlp.sources import (
    discover_crypto_repo_documents,
    fetch_rss_documents,
)
from crypto_ai_swing.orchestration.mtf_challenger import evaluate_mtf_challenger
from crypto_ai_swing.orchestration.pipeline import SwingPipeline
from crypto_ai_swing.orchestration.rally_capture import (
    assess_rally,
    macro_override_allowed,
)
from crypto_ai_swing.orchestration.timeframe_pipeline import evaluate_timeframe_pipeline
from crypto_ai_swing.production.guard import LiveExecutionGuard
from crypto_ai_swing.quant.bayesian_forward import (
    build_bayesian_forward_snapshot,
)
from crypto_ai_swing.quant.bayesian_forward import (
    persist as persist_bayesian_forward,
)
from crypto_ai_swing.research.forward import ForwardEvidenceLedger
from crypto_ai_swing.research.strategy_challenger import StrategyChallengerLab
from crypto_ai_swing.universe.runtime import UniverseManager, screen_frame


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
                datetime.now(UTC).isoformat(),
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
                datetime.now(UTC).isoformat(),
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
        state_name = Path(str(state_rel)).name or "state.sqlite"
        public_mode = "canary" if self.mode == "live" else self.mode
        self.mode_root = (
            settings.project_root
            / "output/crypto_ai_swing/modes"
            / public_mode
        )
        self.state = ProactiveState(
            self.mode_root / "proactive" / state_name
        )
        self.paper_ledger = PaperPortfolioLedger(
            self.mode_root / "proactive" / "paper_ledger.sqlite",
            starting_equity=Decimal(str(
                self.settings.proactive.get("shadow_equity_eur", 10000)
            )),
        )
        self.nlp = NLPMarketEngine(settings.nlp)
        self.crypto = CryptoLibraryBridge(settings.crypto_repo_root)
        # Exchange/account truth is owned by Sjagil/crypto.
        # No Bitvavo transport is instantiated in the swing application.
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
        self.forward = ForwardEvidenceLedger(
            settings.project_root / forward_rel,
            decision_bucket_minutes=int(forward_cfg.get("decision_bucket_minutes", 15)),
        )
        self.agents = AgentRuntime(settings, mode=self.mode)
        self.rl = RLRuntime(settings)
        self.cmc_context = CMCContextCollector(settings)
        self.edge_manager = ResearchEdgeManager(settings, mode=self.mode)
        self.strategy_lab = StrategyChallengerLab(
            settings,
            mode="shadow" if self.mode == "live" else self.mode,
        )
        self.edge_manager.strategy_lab = self.strategy_lab
        self.execution_authority = CryptoAuthorityAdapter(settings.crypto_repo_root)
        self.production_guard = LiveExecutionGuard(
            settings, self.execution_authority
        )
        self.universe = UniverseManager(settings)
        self._news_cache: tuple[float, list] | None = None
        self._last_news_status: dict[str, Any] = {
            "status": "NOT_RUN",
            "source": "Sjagil/crypto:scrapers.rss",
            "source_statuses": [],
        }

    def close(self):
        self.state.close()
        self.forward.close()
        guard = getattr(self, "production_guard", None)
        if guard is not None:
            guard.close()

    def _markets(self) -> list[str]:
        import os

        raw = os.getenv("CRYPTO_SWING_MARKETS")
        if raw:
            values = [x.strip().upper() for x in raw.split(",") if x.strip()]
        else:
            configured = [
                str(x).upper()
                for x in (self.settings.proactive.get("markets", []) or [])
                if str(x).strip()
            ]
            values = configured or list(self.universe.current()["markets"])
        # Never stop managing a locally open position just because universe
        # membership changes at the next six-hour refresh.
        for market in self.state.positions():
            if market not in values:
                values.append(market)
        return values

    def _fallback_account(self) -> tuple[Decimal, Decimal, Decimal]:
        fallback = Decimal(
            str(
                self.settings.proactive.get(
                    "shadow_equity_eur", 10000
                )
            )
        )
        return fallback, fallback, Decimal(0)

    def _paper_fee_bps(self) -> Decimal:
        costs = dict(self.settings.execution.get("costs", {}) or {})
        return Decimal(str(costs.get("fee_bps_per_side", 25.0)))

    def _paper_mark_prices(self, positions) -> dict[str, Decimal]:
        marks: dict[str, Decimal] = {}
        for market, pos in positions.items():
            mark = pos.entry_price
            try:
                bundle = self.crypto.market_bundle(
                    market, "1h", mode=self.mode, persist=False, depth=5
                )
                raw = (
                    bundle.microstructure.get("best_bid")
                    or bundle.microstructure.get("best_ask")
                    or "0"
                )
                candidate = Decimal(str(raw))
                if candidate > 0:
                    mark = candidate
            except Exception:
                pass
            marks[str(market).upper()] = mark
        return marks

    def _simulated_portfolio_account(
        self,
    ) -> tuple[Decimal, Decimal, Decimal]:
        starting_equity = Decimal(
            str(self.settings.proactive.get("shadow_equity_eur", 10000))
        )
        state = getattr(self, "state", None)
        positions = state.positions() if state is not None else {}
        ledger = getattr(self, "paper_ledger", None)
        if ledger is None:
            cost_basis = Decimal(0)
            exposure = Decimal(0)
            marks = self._paper_mark_prices(positions)
            for market, pos in positions.items():
                cost_basis += pos.amount * pos.entry_price
                exposure += pos.amount * marks.get(
                    str(market).upper(), pos.entry_price
                )
            cash = starting_equity - cost_basis
            return cash + exposure, cash, exposure

        ledger.ensure_open_positions(
            positions, fee_bps=self._paper_fee_bps()
        )
        snapshot = ledger.snapshot(
            positions, self._paper_mark_prices(positions)
        )
        self._last_paper_accounting = snapshot
        return (
            Decimal(str(snapshot["equity_eur"])),
            Decimal(str(snapshot["cash_eur"])),
            Decimal(str(snapshot["exposure_eur"])),
        )


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
            if self.mode in {"shadow", "paper"}:
                return self._simulated_portfolio_account()
            return self._fallback_account()

        authority = getattr(self, "execution_authority", None)
        if authority is None:
            if self.mode == "live":
                raise RuntimeError("CANONICAL_CRYPTO_ACCOUNT_AUTHORITY_MISSING")
            return self._simulated_portfolio_account()

        try:
            snapshot = authority.account_snapshot(markets)
        except Exception as exc:
            if self.mode == "live":
                raise RuntimeError(
                    "Canonical crypto account snapshot unavailable: "
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                ) from exc
            return self._simulated_portfolio_account()

        if snapshot.get("status") != "READY":
            if self.mode == "live":
                raise RuntimeError(
                    "Canonical crypto account health is not READY: "
                    + ",".join(str(v) for v in snapshot.get("failures", []))
                )
            return self._simulated_portfolio_account()

        return (
            Decimal(str(snapshot.get("equity_eur") or "0")),
            Decimal(str(snapshot.get("cash_eur") or "0")),
            Decimal(str(snapshot.get("exposure_eur") or "0")),
        )

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
        total_amount = Decimal(0)
        total_quote = Decimal(0)
        for fill in fills:
            try:
                amount = Decimal(str(fill.get("amount", "0")))
                price = Decimal(str(fill.get("price", "0")))
            except Exception:  # noqa: S112
                continue
            total_amount += amount
            total_quote += amount * price
        if total_amount > 0:
            return total_quote / total_amount
        filled = Decimal(str(order.get("filledAmount") or "0"))
        quote = Decimal(
            str(order.get("filledAmountQuote") or "0")
        )
        return quote / filled if filled > 0 else Decimal(0)

    def _prospective_canary_readiness(self) -> dict[str, Any]:
        forward_cfg = (
            self.settings.autonomy.get("forward_evidence", {})
            if hasattr(self.settings, "autonomy")
            else {}
        )
        cfg = dict(forward_cfg.get("canary_readiness", {}) or {})
        return self.forward.canary_readiness(
            primary_horizon_hours=int(cfg.get("primary_horizon_hours", 4)),
            minimum_unblocked_buy_outcomes=int(
                cfg.get("minimum_unblocked_buy_outcomes", 30)
            ),
            minimum_distinct_markets=int(cfg.get("minimum_distinct_markets", 5)),
            minimum_observation_span_hours=float(
                cfg.get("minimum_observation_span_hours", 72)
            ),
            minimum_mean_return_bps=float(
                cfg.get("minimum_mean_return_bps", 0.0)
            ),
            minimum_positive_return_rate=float(
                cfg.get("minimum_positive_return_rate", 0.50)
            ),
        )

    def _execute_buy(self, intent: TradeIntent) -> dict[str, Any]:
        if self.mode != "live":
            equity_now, cash_now, exposure_now = self._simulated_portfolio_account()
            paper_state = getattr(self, "_last_paper_accounting", {}) or {}
            if paper_state.get("reconciliation_required"):
                return {
                    "mode": self.mode,
                    "accepted": False,
                    "simulated": True,
                    "execution_backend": "paper_portfolio_ledger",
                    "reason_code": "SIMULATED_ACCOUNT_RECONCILIATION_REQUIRED",
                    "intent_id": intent.intent_id,
                }
            fee_fraction = self._paper_fee_bps() / Decimal(10000)
            required_cash = intent.notional_eur * (Decimal(1) + fee_fraction)
            if cash_now < required_cash:
                return {
                    "mode": self.mode,
                    "accepted": False,
                    "simulated": True,
                    "execution_backend": "paper_portfolio_ledger",
                    "reason_code": "SIMULATED_INSUFFICIENT_CASH",
                    "required_cash_eur": str(required_cash),
                    "available_cash_eur": str(cash_now),
                    "intent_id": intent.intent_id,
                }
            portfolio_cfg = dict(self.settings.risk.get("portfolio", {}) or {})
            max_fraction = Decimal(str(portfolio_cfg.get(
                "max_total_exposure_fraction",
                portfolio_cfg.get("max_exposure_fraction", 0.85),
            )))
            projected_exposure = exposure_now + intent.notional_eur
            if (
                equity_now > 0
                and max_fraction > 0
                and projected_exposure / equity_now > max_fraction
            ):
                return {
                    "mode": self.mode,
                    "accepted": False,
                    "simulated": True,
                    "execution_backend": "paper_portfolio_ledger",
                    "reason_code": "SIMULATED_EXPOSURE_CAP",
                    "projected_exposure_eur": str(projected_exposure),
                    "equity_eur": str(equity_now),
                    "maximum_exposure_fraction": str(max_fraction),
                    "intent_id": intent.intent_id,
                }
        if self.mode != "live":
            return {
                "mode": self.mode,
                "simulated": True,
                "accepted": True,
                "intent_id": intent.intent_id,
                "execution_backend": "shadow_or_paper",
            }
        readiness = self._prospective_canary_readiness()
        validation_canary = bool(
            (getattr(intent, "metadata", None) or {}).get(
                "execution_validation_canary", False
            )
        )
        if not bool(readiness.get("eligible")) and not validation_canary:
            return {
                "mode": "live",
                "accepted": False,
                "execution_backend": "prospective_canary_gate",
                "reason_code": "PROSPECTIVE_CANARY_EVIDENCE_NOT_READY",
                "blockers": list(readiness.get("blockers") or []),
                "prospective_readiness": readiness,
                "orders_submitted": 0,
            }

        if validation_canary:
            cfg = execution_validation_canary_config(
                self.settings.proactive
            )
            if not bool(cfg.get("enabled", False)):
                return {
                    "mode": "live",
                    "accepted": False,
                    "execution_backend": "execution_validation_canary_gate",
                    "reason_code": "EXECUTION_VALIDATION_CANARY_DISABLED",
                    "orders_submitted": 0,
                }
            maximum = Decimal(str(cfg.get("maximum_order_eur", 10.0)))
            if intent.notional_eur > maximum:
                return {
                    "mode": "live",
                    "accepted": False,
                    "execution_backend": "execution_validation_canary_gate",
                    "reason_code": "EXECUTION_VALIDATION_NOTIONAL_EXCEEDED",
                    "maximum_order_eur": str(maximum),
                    "orders_submitted": 0,
                }
            try:
                preflight = self.execution_authority.preflight(intent)
            except Exception as exc:
                return {
                    "mode": "live",
                    "accepted": False,
                    "execution_backend": "Sjagil/crypto:core.swing_layer_live",
                    "reason_code": "CANONICAL_LIVE_PREFLIGHT_ERROR",
                    "blockers": [
                        f"{type(exc).__name__}:{str(exc)[:500]}"
                    ],
                    "orders_submitted": 0,
                }
            denied, reasons = canonical_preflight_explicitly_denied(
                preflight
            )
            if denied:
                return {
                    "mode": "live",
                    "accepted": False,
                    "execution_backend": "Sjagil/crypto:core.swing_layer_live",
                    "reason_code": "CANONICAL_LIVE_PREFLIGHT_DENIED",
                    "blockers": reasons,
                    "canonical_preflight": preflight,
                    "orders_submitted": 0,
                }
        else:
            preflight = None

        try:
            guard = getattr(self, "production_guard", None)
            if guard is None:
                raise RuntimeError("PRODUCTION_GUARD_NOT_INITIALIZED")
            guarded = guard.submit_buy(
                intent,
                markets=self._markets(),
                canonical_preflight=preflight,
            )
            return {
                "mode": "live",
                "execution_backend": "Sjagil/crypto:core.swing_layer_live",
                "execution_validation_canary": validation_canary,
                "economic_edge_unproven": validation_canary,
                "alpha_evidence_authorized": False,
                "autoscale_authorized": False,
                "prospective_readiness": readiness,
                "canonical_preflight": preflight,
                **guarded,
            }
        except Exception as exc:
            return {
                "mode": "live",
                "accepted": False,
                "execution_backend": "Sjagil/crypto:core.swing_layer_live",
                "execution_validation_canary": validation_canary,
                "blockers": [
                    f"{type(exc).__name__}:{str(exc)[:500]}"
                ],
                "orders_submitted": 0,
            }

    def _sync_live_positions(self, markets: list[str]) -> None:
        if self.mode != "live":
            return
        try:
            guard = getattr(self, "production_guard", None)
            if guard is not None:
                reconciliation = guard.reconcile(markets)
                self._last_live_reconciliation = reconciliation
                if not bool(reconciliation.get("ready")):
                    return
            else:
                reconciliation = self.execution_authority.reconcile(markets)
                if str(reconciliation.get("status") or "").upper() != "READY":
                    return
            rows = list(
                dict(
                    self.execution_authority.portfolio().get("positions")
                    or {}
                ).values()
            )
            canonical_markets = {
                str(row.get("market") or "").upper()
                for row in rows
                if isinstance(row, dict) and row.get("market")
            }
        except Exception:
            return
        local = self.state.positions()
        for market in list(local):
            if market not in canonical_markets:
                self.state.delete_position(market)

    def _record_simulated_position(self, intent: TradeIntent, execution: dict[str, Any], frame: pd.DataFrame) -> None:
        if self.mode == "live" or execution.get("accepted") is not True: return
        try:
            price=Decimal(str(frame["close"].iloc[-1])); amount=intent.notional_eur/price
        except Exception: return
        if price<=0 or amount<=0:return
        self.state.upsert_position(Position(intent.market,amount,price,price,intent.stop_pct,intent.take_profit_pct,intent.trailing_stop_pct,datetime.now(UTC).isoformat()))
        ledger = getattr(self, "paper_ledger", None)
        if ledger is not None:
            ledger.record_buy(
                event_id=f"paper-buy:{intent.intent_id}",
                market=intent.market,
                quantity=amount,
                price=price,
                fee_bps=self._paper_fee_bps(),
                payload={
                    "mode": self.mode,
                    "intent_id": intent.intent_id,
                    "execution_backend": execution.get("execution_backend"),
                    "strategy": intent.strategy,
                    "expected_edge_bps": str(intent.expected_edge_bps),
                    "estimated_round_trip_cost_bps": str(
                        intent.estimated_round_trip_cost_bps
                    ),
                    "net_edge_bps": str(intent.net_edge_bps),
                    "stop_pct": float(intent.stop_pct),
                    "take_profit_pct": float(intent.take_profit_pct),
                    "trailing_stop_pct": float(intent.trailing_stop_pct),
                    "intent_metadata": intent.metadata,
                    "attribution_schema": "paper_intent_context_v2",
                },
            )

    def _record_position_after_execution(
        self,
        intent: TradeIntent,
        execution: dict[str, Any],
        frame: pd.DataFrame,
    ) -> None:
        if execution.get("accepted") is not True:
            return
        if self.mode != "live":
            self._record_simulated_position(intent, execution, frame)
            return
        canonical = dict(execution.get("canonical_result") or execution)
        order = dict(canonical.get("order") or {})
        try:
            amount = Decimal(
                str(
                    order.get("filled_quantity")
                    or order.get("filledAmount")
                    or order.get("quantity")
                    or "0"
                )
            )
            price = Decimal(
                str(
                    order.get("average_price")
                    or order.get("averagePrice")
                    or order.get("price")
                    or frame["close"].iloc[-1]
                )
            )
        except Exception:
            return
        if amount <= 0 or price <= 0:
            return
        self.state.upsert_position(
            Position(
                intent.market,
                amount,
                price,
                price,
                intent.stop_pct,
                intent.take_profit_pct,
                intent.trailing_stop_pct,
                datetime.now(UTC).isoformat(),
            )
        )

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
            except Exception:  # noqa: S112
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
                Decimal(1) - Decimal(str(pos.stop_pct))
            )
            target = pos.entry_price * (
                Decimal(1) + Decimal(str(pos.take_profit_pct))
            )
            trailing = highest * (
                Decimal(1) - Decimal(str(pos.trailing_stop_pct))
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
            if self.mode == "live" and reason == "STOP_LOSS":
                try:
                    guard = getattr(self, "production_guard", None)
                    if guard is None:
                        raise RuntimeError("PRODUCTION_GUARD_NOT_INITIALIZED")
                    reconciliation = guard.reconcile(self._markets())
                    self._last_live_reconciliation = reconciliation
                    self._sync_live_positions([market])
                    events.append(
                        {
                            "market": market,
                            "action": "LIVE_NATIVE_STOP_MONITOR",
                            "reason": "NATIVE_STOP_LOSS",
                            "reconciliation": reconciliation,
                            "duplicate_market_sell_submitted": False,
                        }
                    )
                except Exception as exc:
                    events.append(
                        {
                            "market": market,
                            "action": "NATIVE_STOP_MONITOR_BLOCKED",
                            "reason": "STOP_LOSS",
                            "blockers": [
                                f"{type(exc).__name__}:{str(exc)[:300]}"
                            ],
                        }
                    )
                continue
            if self.mode == "live":
                try:
                    guard = getattr(self, "production_guard", None)
                    if guard is None:
                        raise RuntimeError("PRODUCTION_GUARD_NOT_INITIALIZED")
                    guarded = guard.submit_exit(
                        market=market,
                        reason=reason,
                        quantity=str(pos.amount),
                        markets=self._markets(),
                    )
                    accepted = bool(guarded.get("accepted"))
                    events.append({
                        "market": market,
                        "action": "LIVE_EXIT",
                        "reason": reason,
                        "accepted": accepted,
                        "execution": guarded,
                    })
                    if accepted:
                        self._sync_live_positions([market])
                except Exception as exc:
                    events.append({"market":market,"action":"EXIT_BLOCKED","reason":reason,"blockers":[f"{type(exc).__name__}:{str(exc)[:300]}"]})
                continue
            events.append(
                {
                    "market": market,
                    "action": "SIMULATED_SELL",
                    "reason": reason,
                }
            )
            ledger = getattr(self, "paper_ledger", None)
            if ledger is not None:
                ledger.record_sell(
                    event_id=f"paper-sell:{market}:{pos.opened_at}:{reason}",
                    market=market,
                    quantity=pos.amount,
                    entry_price=pos.entry_price,
                    exit_price=price,
                    fee_bps=self._paper_fee_bps(),
                    payload={"mode": self.mode, "reason": reason},
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
        universe_snapshot = self.universe.current()
        try:
            cmc_context = self.cmc_context.current(markets)
        except Exception as exc:
            cmc_context = {
                'status': 'ERROR',
                'error': f'{type(exc).__name__}:{str(exc)[:300]}',
                'runtime_assets': {},
                'authority': 'CONTEXT_ONLY',
                'live_decision_influence': False,
            }
        universe_candidates = {
            str(row.get("market")): dict(row)
            for row in (universe_snapshot.get("candidates") or [])
            if isinstance(row, dict) and row.get("market")
        }
        try:
            edge_policy = self.edge_manager.refresh_policy(self.forward.path)
        except Exception as exc:
            edge_policy = {
                "status": "ERROR",
                "error": f"{type(exc).__name__}:{str(exc)[:300]}",
                "shadow_influence": False,
                "live_decision_influence": False,
            }
        self._sync_live_positions(markets)
        exit_events = self._manage_exits()
        docs = self._news()
        spreads: dict[str, float] = {}
        context: dict[str, dict] = {}
        skipped: list[dict] = []
        mtf_summary: dict[str, dict[str, int]] = {}
        persist = self._persist_market_data()
        timeframes = tuple(
            str(x)
            for x in self.settings.proactive.get(
                "timeframes", ["15m", "1h", "2h", "4h", "1d", "1w"]
            )
        )
        primary = str(self.settings.proactive.get("primary_signal_timeframe", "1h"))
        execution_tf = str(self.settings.proactive.get("execution_timeframe", "15m"))
        observed_at = datetime.now(UTC)
        depth = int(self.settings.proactive.get("orderbook_depth", 100))
        deep_scan_limit = max(
            1,
            int(self.settings.proactive.get("deep_scan_limit", 8)),
        )
        rally_cfg = dict(
            self.settings.proactive.get("rally_capture", {}) or {}
        )
        rally_scan_limit = max(
            0,
            int(rally_cfg.get("additional_deep_scan_limit", 6)),
        )
        concurrency = max(
            1,
            min(
                8,
                int(self.settings.proactive.get("screen_concurrency", 4)),
            ),
        )
        library_status = self.crypto.integration_status()

        # Stage A: screen the full 25-market universe with closed primary bars only.
        try:
            primary_frames = self.crypto.ohlcv_many(
                markets, primary, persist=persist, concurrency=concurrency
            )
        except Exception as exc:
            primary_frames = {}
            skipped.append({"market": "UNIVERSE", "reason": type(exc).__name__, "detail": str(exc)[:300]})
        screen: dict[str, dict[str, float]] = {}
        causal_primary: dict[str, pd.DataFrame] = {}
        for market in markets:
            frame = primary_frames.get(market)
            if frame is None or frame.empty:
                skipped.append({"market": market, "reason": "NO_PRIMARY_SCREEN_CANDLES"})
                continue
            causal = self.crypto.causal_frame(frame, primary, observed_at)
            if causal.empty:
                skipped.append({"market": market, "reason": "NO_CAUSAL_PRIMARY_SCREEN_CANDLE"})
                continue
            causal_primary[market] = causal
            screen_row = screen_frame(causal)
            screen_row["technical"] = technical_snapshot(
                causal, timeframe=primary
            )
            universe_row = universe_candidates.get(market, {})
            try:
                spread_hint = float(universe_row.get("spread_bps"))
            except (TypeError, ValueError):
                spread_hint = 35.0
            if not np.isfinite(spread_hint) or spread_hint < 0:
                spread_hint = 35.0
            spread_scale = max(1.0, float(
                self.settings.proactive.get("screen_spread_quality_scale_bps", 15.0)
            ))
            execution_quality = float(np.exp(-spread_hint / spread_scale))
            liquidity_component = 2.0 * execution_quality - 1.0
            raw_score = float(screen_row.get("opportunity_score", -999.0))
            screen_row.update({
                "universe_spread_bps": spread_hint,
                "universe_quality_tier": universe_row.get("quality_tier"),
                "execution_quality": execution_quality,
                "execution_adjusted_score": float(
                    0.85 * raw_score + 0.15 * liquidity_component
                ),
            })
            screen_row["rally_capture"] = assess_rally(
                screen_row,
                rally_cfg,
            ).to_dict()
            cmc_asset = dict(
                (cmc_context.get('runtime_assets') or {}).get(market) or {}
            )
            agent_preview = self.agents.predict_frame(
                market, causal, {'spread_bps': spread_hint, 'cmc': cmc_asset}
            )
            rl_preview = self.rl.predict_frame(causal)
            screen_row["agent_preview"] = {
                'alpha_probability': agent_preview.alpha_probability,
                'regime_score': agent_preview.regime_score,
                'forecast_score': agent_preview.forecast_score,
                'predicted_return': agent_preview.predicted_return,
                'predicted_mae': agent_preview.predicted_mae,
                'execution_score': agent_preview.execution_score,
                'live_influence': agent_preview.live_influence,
            }
            screen_row["rl_preview"] = rl_preview
            screen_row["cmc"] = cmc_asset
            screen[market] = screen_row

        ranked = sorted(
            screen,
            key=lambda market: float(
                screen[market].get("execution_adjusted_score", -999.0)
            ),
            reverse=True,
        )
        open_markets = set(self.state.positions())
        rally_ranked = sorted(
            [
                market
                for market in screen
                if bool(
                    (screen[market].get("rally_capture") or {}).get(
                        "deep_scan_candidate"
                    )
                )
            ],
            key=lambda market: float(
                (screen[market].get("rally_capture") or {}).get(
                    "score",
                    -999.0,
                )
            ),
            reverse=True,
        )
        base_deep_markets = ranked[:deep_scan_limit]
        rally_deep_markets = rally_ranked[:rally_scan_limit]
        deep_markets: list[str] = []
        for market in (
            list(open_markets)
            + base_deep_markets
            + rally_deep_markets
        ):
            if market in causal_primary and market not in deep_markets:
                deep_markets.append(market)

        # Expensive context is limited to ordinary leaders, independent
        # acceleration candidates, and already-open positions.
        prospective_context: dict[str, Any] = {}
        prospective_error: str | None = None
        try:
            prospective_context = self.crypto.prospective_context(
                deep_markets,
                cache_seconds=float(self.settings.proactive.get("prospective_context_cache_seconds", 300)),
            )
        except Exception as exc:
            prospective_error = f"{type(exc).__name__}: {str(exc)[:300]}"
        try:
            micro_readiness = self.crypto.microstructure_readiness()
        except Exception as exc:
            micro_readiness = {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}

        prospective_status = str(prospective_context.get("status") or "UNKNOWN").upper()
        prospective_block = prospective_status in {"BLOCK_NEW_ENTRIES", "BLOCKED", "FAILED", "NOT_READY"}
        frames: dict[str, pd.DataFrame] = {}
        forward_frames: dict[str, pd.DataFrame] = {}

        for market in deep_markets:
            try:
                bundle = self.crypto.market_bundle(
                    market, primary, mode=self.mode, persist=persist, depth=depth
                )
                if bundle.frame.empty:
                    skipped.append({"market": market, "reason": "NO_CANDLES"})
                    continue
                mtf_frames = self.crypto.multi_timeframe_frames(
                    market, timeframes, persist=persist
                )
                mtf_frames[primary] = bundle.frame
                decision_at = observed_at
                causal_frames = {tf: self.crypto.causal_frame(frame, tf, decision_at) for tf, frame in mtf_frames.items()}
                if causal_frames.get(primary) is None or causal_frames[primary].empty:
                    skipped.append({"market": market, "reason": "NO_CAUSAL_PRIMARY_CANDLE"})
                    continue
                technical_mtf = multi_timeframe_snapshot(
                    causal_frames
                )
                tf_decision = evaluate_timeframe_pipeline(
                    causal_frames, bundle.microstructure, observed_at=observed_at,
                    maximum_spread_bps=float(
                        self.settings.execution.get(
                            "liquidity", {}
                        ).get("maximum_spread_bps", 35.0)
                    ),
                    policy=dict(
                        self.settings.proactive.get("timeframe_pipeline", {})
                        or {}
                    ),
                )
                mtf = tf_decision.mtf_score
                orderflow = self._orderflow_score(bundle.microstructure)
                quote_volume = self.crypto.quote_volume_24h(bundle.ticker, bundle.microstructure)
                frame = causal_frames[primary].copy()
                frame["quote_volume_24h"] = quote_volume
                spreads[market] = float(bundle.microstructure.get("spread_bps", 999.0))
                nlp_aggregation = self.nlp.aggregate_with_diagnostics(docs, market)
                assessment = nlp_aggregation.assessment
                rally_assessment = dict(
                    screen.get(market, {}).get("rally_capture") or {}
                )
                rally_macro_override = macro_override_allowed(
                    tf_decision,
                    rally_assessment,
                    mode=self.mode,
                    config=rally_cfg,
                )
                timeframe_entry_blocked = bool(
                    tf_decision.entry_blocked
                    and not rally_macro_override
                )
                entry_blocked = bool(
                    (
                        prospective_block
                        and self.settings.proactive.get(
                            "context_gates",
                            {},
                        ).get(
                            "block_on_prospective_context_block",
                            True,
                        )
                    )
                    or timeframe_entry_blocked
                )
                context[market] = {
                    "nlp_score": assessment.score,
                    "nlp_confidence": assessment.confidence,
                    "nlp_severe_negative": assessment.severe_negative,
                    "event_tags": list(assessment.event_tags),
                    "nlp_model": assessment.model,
                    "nlp_diagnostics": nlp_aggregation.diagnostics,
                    "mtf_score": mtf,
                    "timeframe_pipeline": tf_decision.to_dict(),
                    "technical_intelligence": technical_mtf,
                    "orderflow_score": orderflow,
                    "book_imbalance": bundle.microstructure.get("book_imbalance", 0.0),
                    "cvd_ratio": bundle.microstructure.get("cvd_ratio", 0.0),
                    "cvd_notional_ratio": bundle.microstructure.get("cvd_notional_ratio", 0.0),
                    "buy_volume": bundle.microstructure.get("buy_volume", 0.0),
                    "sell_volume": bundle.microstructure.get("sell_volume", 0.0),
                    "buy_notional": bundle.microstructure.get("buy_notional", 0.0),
                    "sell_notional": bundle.microstructure.get("sell_notional", 0.0),
                    "trade_count": bundle.microstructure.get("trade_count", 0.0),
                    "classified_trade_count": bundle.microstructure.get("classified_trade_count", 0.0),
                    "unclassified_trade_count": bundle.microstructure.get("unclassified_trade_count", 0.0),
                    "quote_volume_24h": quote_volume,
                    "decision_at": decision_at.isoformat(),
                    "microprice_edge_bps": bundle.microstructure.get("microprice_edge_bps", 0.0),
                    "spread_bps": bundle.microstructure.get("spread_bps", 999.0),
                    "prospective_context_status": prospective_status,
                    "rally_capture": rally_assessment,
                    "rally_macro_override_applied": rally_macro_override,
                    "raw_timeframe_entry_blocked": tf_decision.entry_blocked,
                    "entry_blocked": entry_blocked,
                    "data_source": "Sjagil/crypto",
                    "universe_screen": screen.get(market, {}),
                }
                agent_decision = self.agents.predict_frame(market, frame, context[market])
                context[market]["agents"] = {
                    "alpha_probability": agent_decision.alpha_probability,
                    "forecast_score": agent_decision.forecast_score,
                    "regime_score": agent_decision.regime_score,
                    "predicted_return": agent_decision.predicted_return,
                    "predicted_mae": agent_decision.predicted_mae,
                    "execution_score": agent_decision.execution_score,
                    "blocker_codes": list(agent_decision.blocker_codes),
                    "live_influence": agent_decision.live_influence,
                    "diagnostics": agent_decision.diagnostics,
                }
                head_influence = dict(
                    agent_decision.diagnostics.get("head_influence") or {}
                )
                context[market]["ml_probability"] = (
                    agent_decision.alpha_probability
                    if head_influence.get("alpha")
                    else None
                )
                context[market]["forecast_score"] = (agent_decision.forecast_score if head_influence.get("return") else None)
                context[market]["predicted_return"] = (agent_decision.predicted_return if head_influence.get("return") else None)
                context[market]["predicted_mae"] = (agent_decision.predicted_mae if head_influence.get("risk") else None)
                context[market]["agent_entry_blocked"] = bool(
                    agent_decision.entry_blocked
                )
                rl_preview = dict(
                    screen.get(market, {}).get('rl_preview') or {}
                )
                context[market]["rl"] = rl_preview
                context[market]["cmc"] = dict(
                    (cmc_context.get('runtime_assets') or {}).get(market) or {}
                )
                rl_cfg = dict(self.settings.agents.get('rl', {}) or {})
                context[market]["rl_score"] = (
                    float(rl_preview['score'])
                    if self.mode != 'live'
                    and bool(rl_cfg.get('shadow_signal_influence', True))
                    and bool(rl_preview.get('qualified', False))
                    and rl_preview.get('score') is not None
                    else None
                )
                mtf_challenger = evaluate_mtf_challenger(
                    tf_decision,
                    microstructure=bundle.microstructure,
                    cmc_asset=context[market].get("cmc", {}),
                    cmc_context=cmc_context,
                )
                context[market]["mtf_challenger"] = mtf_challenger.to_dict()
                edge_decision = self.edge_manager.evaluate_market(
                    context[market],
                    policy=edge_policy,
                )
                context[market]["edge_manager"] = edge_decision
                context[market]["research_meta_score"] = edge_decision.get("signal_score")
                context[market]["research_strategy_hint"] = (
                    edge_decision.get("strategy_hint")
                    or mtf_challenger.strategy_family
                    if edge_decision.get("signal_score") is not None
                    else None
                )
                mtf_summary[market] = {
                    tf: len(causal_frames.get(tf, pd.DataFrame())) for tf in timeframes
                }
                frames[market] = frame
                execution_frame = causal_frames.get(execution_tf)
                if execution_frame is not None and not execution_frame.empty:
                    forward_frames[market] = execution_frame.copy()
            except Exception as exc:
                skipped.append({"market": market, "reason": type(exc).__name__, "detail": str(exc)[:300]})

        equity, cash, exposure = self._account(markets)
        positions_for_risk = self.state.positions()
        open_risk_eur = sum(
            (
                pos.amount
                * pos.entry_price
                * Decimal(str(pos.stop_pct))
                for pos in positions_for_risk.values()
            ),
            Decimal(0),
        )
        authority = Authority.LIVE if self.mode == "live" else Authority.PAPER if self.mode == "paper" else Authority.SHADOW
        portfolio_positions = [
            {
                "market": pos.market,
                "amount": pos.amount,
                "entry_price": pos.entry_price,
                "open_risk_eur": (
                    pos.amount
                    * pos.entry_price
                    * Decimal(str(pos.stop_pct))
                ),
            }
            for pos in positions_for_risk.values()
        ]
        result = SwingPipeline(self.settings).run(
            frames,
            equity_eur=equity,
            cash_eur=cash,
            exposure_eur=exposure,
            open_risk_eur=open_risk_eur,
            spread_bps=spreads,
            market_context=context,
            authority=authority,
            portfolio_positions=portfolio_positions,
        )

        executions = []
        positions = self.state.positions()
        maximum_positions = int(
            self.settings.risk.get("portfolio", {}).get(
                "max_positions", 5
            )
        )
        positions_for_execution = self.state.positions()
        remaining_position_slots = max(
            0,
            maximum_positions - len(positions_for_execution),
        )
        for intent in result.intents:
            if (
                intent.market not in positions_for_execution
                and remaining_position_slots <= 0
            ):
                result.blocked.append(
                    {
                        "market": intent.market,
                        "blockers": ["MAX_POSITIONS_REACHED"],
                    }
                )
                continue
            frame = frames.get(intent.market)
            if frame is None or frame.empty or intent.market in positions:
                continue
            candle_ts = frame.index[-1].isoformat()
            if self.state.seen(intent.market, candle_ts, "BUY"):
                continue
            execution = self._execute_buy(intent)
            self.state.mark(intent.market, candle_ts, "BUY", {"intent": intent.to_dict(), "execution": execution})
            executions.append({"intent": intent.to_dict(), "execution": execution})
            self._record_position_after_execution(intent, execution, frame)
            if (
                execution.get("accepted") is True
                and intent.market not in positions_for_execution
            ):
                remaining_position_slots = max(
                    0, remaining_position_slots - 1
                )
                positions_for_execution = self.state.positions()

        # post_execution_account_refresh_v0232
        paper_accounting = None
        if self.mode in {"shadow", "paper"}:
            equity, cash, exposure = self._account(markets)
            positions_for_risk = self.state.positions()
            open_risk_eur = sum(
                (
                    pos.amount
                    * pos.entry_price
                    * Decimal(str(pos.stop_pct))
                    for pos in positions_for_risk.values()
                ),
                Decimal(0),
            )
            paper_accounting = getattr(
                self, "_last_paper_accounting", None
            )

        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "mode": self.mode,
            "data_source": "Sjagil/crypto python library",
            "markets": markets,
            "universe": {
                "selected_size": universe_snapshot.get("selected_size"),
                "markets": universe_snapshot.get("markets"),
                "generated_at": universe_snapshot.get("generated_at"),
                "expires_at": universe_snapshot.get("expires_at"),
                "liquidity_degraded": universe_snapshot.get("liquidity_degraded"),
                "preferred_liquidity_count": universe_snapshot.get("preferred_liquidity_count"),
                "fallback_liquidity_count": universe_snapshot.get("fallback_liquidity_count"),
                "policy": universe_snapshot.get("policy"),
                "current_liquidity_snapshot_not_point_in_time": universe_snapshot.get("current_liquidity_snapshot_not_point_in_time"),
            },
            "screened_markets": len(causal_primary),
            "deep_scan_markets": deep_markets,
            "deep_scan_limit": deep_scan_limit,
            "rally_deep_scan_limit": rally_scan_limit,
            "rally_deep_scan_markets": rally_deep_markets,
            "screen": screen,
            "cmc_context": cmc_context,
            "edge_policy": edge_policy,
            "full_universe_agent_inference": {
                "market_count": sum(bool(row.get("agent_preview")) for row in screen.values()),
                "rl_market_count": sum((row.get("rl_preview") or {}).get("score") is not None for row in screen.values()),
                "live_decision_influence": False,
            },
            "equity_eur": str(equity),
            "cash_eur": str(cash),
            "exposure_eur": str(exposure),
            "paper_accounting": paper_accounting,
            "open_risk_eur": str(open_risk_eur),
            "position_capacity": {
                "maximum_positions": maximum_positions,
                "positions_before_cycle_entries": len(
                    positions_for_risk
                ),
                "positions_after_cycle": len(
                    self.state.positions()
                ),
                "remaining_slots_after_cycle": (
                    remaining_position_slots
                ),
            },
            "nlp_documents": len(docs),
            "news": self._last_news_status,
            "crypto_library_ready": library_status.get("ready", False),
            "crypto_library_imported_modules": library_status.get("imported_modules", 0),
            "crypto_library_required_modules": library_status.get("required_modules", 0),
            "prospective_context_status": prospective_status,
            "prospective_context_error": prospective_error,
            "microstructure_readiness": micro_readiness,
            "agent_runtime": self.agents.status(),
            "mtf_rows": mtf_summary,
            "signals": [
                {"market": s.market, "side": s.side.value, "score": s.score,
                 "edge_bps": s.expected_edge_bps, "edge_source": s.edge_source,
                 "votes": [v.source for v in s.votes]}
                for s in result.signals
            ],
            "market_context": context,
            "blocked": result.blocked,
            "executions": executions,
            "exit_events": exit_events,
            "skipped": skipped,
            "positions": {
                k: {**asdict(v), "amount": str(v.amount), "entry_price": str(v.entry_price), "highest_price": str(v.highest_price)}
                for k, v in self.state.positions().items()
            },
        }
        candle_times = {
            market: frame.index[-1].isoformat()
            for market, frame in frames.items()
            if frame is not None and not frame.empty
        }
        forward_cfg = self.settings.autonomy.get("forward_evidence", {}) if hasattr(self.settings, "autonomy") else {}
        should_record = bool(forward_cfg.get("enabled", True)) and bool(forward_cfg.get(f"record_{self.mode}", True))
        if should_record:
            payload["forward_evidence"] = self.forward.append_cycle(payload, candle_times)
            if bool(forward_cfg.get("mature_on_cycle", True)):
                horizons = tuple(int(value) for value in (forward_cfg.get("horizons_hours", [1, 4, 24]) or []) if int(value) > 0)
                payload["forward_evidence"]["maturation"] = self.forward.mature_from_frames(
                    forward_frames, horizons_hours=horizons or (1, 4, 24), execution_timeframe=execution_tf
                )
            payload["forward_evidence"]["ledger"] = self.forward.status()
            payload["forward_evidence"]["outcomes"] = self.forward.outcome_status()
        else:
            payload["forward_evidence"] = {"recorded": False, "ledger": self.forward.status(), "outcomes": self.forward.outcome_status()}

        payload["prospective_canary_readiness"] = self._prospective_canary_readiness()

        try:
            bayesian = build_bayesian_forward_snapshot(self.forward.path, horizon_hours=4, draws=5000)
            persist_bayesian_forward(
                bayesian,
                self.settings.project_root / "output/crypto_ai_swing/agents/bayesian_forward.json",
            )
        except Exception as exc:
            bayesian = {
                "status": "ERROR",
                "error": f"{type(exc).__name__}:{str(exc)[:300]}",
                "authority": "RESEARCH_ONLY",
                "live_decision_influence": False,
            }
        payload["bayesian_forward"] = bayesian

        out = self.mode_root / "proactive" / "latest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        if self.mode == "shadow":
            legacy = (
                self.settings.project_root
                / "output/crypto_ai_swing/proactive/latest.json"
            )
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_text(
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
                    self.mode_root
                    / "proactive"
                    / "errors.jsonl"
                )
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "at": datetime.now(UTC).isoformat(),
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
