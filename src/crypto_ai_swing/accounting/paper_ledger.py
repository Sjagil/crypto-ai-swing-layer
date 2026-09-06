from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

ZERO = Decimal("0")
BPS = Decimal("10000")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _d(value: Any, default: Decimal = ZERO) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return default


class PaperPortfolioLedger:
    SCHEMA = "crypto_ai_swing_paper_portfolio_ledger_v1"

    def __init__(self, path: Path, *, starting_equity: Decimal) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.starting_equity = _d(starting_equity)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    side TEXT NOT NULL,
                    market TEXT NOT NULL,
                    quantity TEXT NOT NULL,
                    price TEXT NOT NULL,
                    gross_eur TEXT NOT NULL,
                    fee_eur TEXT NOT NULL,
                    realized_pnl_eur TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_paper_events_market
                ON events(market, created_at);
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version',?)",
                (self.SCHEMA,),
            )
            conn.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('starting_equity_eur',?)",
                (str(self.starting_equity),),
            )
            conn.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('high_water_equity_eur',?)",
                (str(self.starting_equity),),
            )
            conn.commit()
        self.starting_equity = self._meta_decimal(
            "starting_equity_eur", self.starting_equity
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _meta_decimal(self, key: str, default: Decimal) -> Decimal:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key=?", (key,)
            ).fetchone()
        return _d(row[0], default) if row else default

    def _set_meta(self, key: str, value: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)),
            )
            conn.commit()

    @staticmethod
    def _fee(gross: Decimal, fee_bps: Decimal) -> Decimal:
        return gross * _d(fee_bps) / BPS

    def _insert_event(
        self,
        *,
        event_id: str,
        side: str,
        market: str,
        quantity: Decimal,
        price: Decimal,
        fee_bps: Decimal,
        realized_pnl_eur: Decimal = ZERO,
        payload: Mapping[str, Any] | None = None,
    ) -> bool:
        quantity = _d(quantity)
        price = _d(price)
        if quantity <= 0 or price <= 0:
            return False
        gross = quantity * price
        fee = self._fee(gross, fee_bps)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO events(
                    event_id,created_at,side,market,quantity,price,
                    gross_eur,fee_eur,realized_pnl_eur,payload
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    str(event_id), _now(), str(side).upper(),
                    str(market).upper(), str(quantity), str(price),
                    str(gross), str(fee), str(_d(realized_pnl_eur)),
                    json.dumps(dict(payload or {}), sort_keys=True, default=str),
                ),
            )
            conn.commit()
            return cursor.rowcount > 0

    def record_buy(
        self,
        *,
        event_id: str,
        market: str,
        quantity: Decimal,
        price: Decimal,
        fee_bps: Decimal,
        payload: Mapping[str, Any] | None = None,
    ) -> bool:
        return self._insert_event(
            event_id=event_id,
            side="BUY",
            market=market,
            quantity=quantity,
            price=price,
            fee_bps=fee_bps,
            payload=payload,
        )

    def record_sell(
        self,
        *,
        event_id: str,
        market: str,
        quantity: Decimal,
        entry_price: Decimal,
        exit_price: Decimal,
        fee_bps: Decimal,
        payload: Mapping[str, Any] | None = None,
    ) -> bool:
        quantity = _d(quantity)
        entry_price = _d(entry_price)
        exit_price = _d(exit_price)
        entry_gross = quantity * entry_price
        exit_gross = quantity * exit_price
        realized = (
            exit_gross - entry_gross
            - self._fee(entry_gross, fee_bps)
            - self._fee(exit_gross, fee_bps)
        )
        return self._insert_event(
            event_id=event_id,
            side="SELL",
            market=market,
            quantity=quantity,
            price=exit_price,
            fee_bps=fee_bps,
            realized_pnl_eur=realized,
            payload=payload,
        )

    def event_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM events").fetchone()
        return int(row[0] if row else 0)

    def net_event_quantities(self) -> dict[str, Decimal]:
        net: dict[str, Decimal] = {}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT side,market,quantity FROM events ORDER BY rowid"
            ).fetchall()
        for side, market, quantity in rows:
            key = str(market).upper()
            qty = _d(quantity)
            if str(side).upper() == "BUY":
                net[key] = net.get(key, ZERO) + qty
            elif str(side).upper() == "SELL":
                net[key] = net.get(key, ZERO) - qty
        return net

    def ensure_open_positions(
        self,
        positions: Mapping[str, Any],
        *,
        fee_bps: Decimal,
    ) -> int:
        """Bootstrap only positions that have no event-side inventory.

        v0.23.2/v0.24.0 bootstrapped every open position under a second event
        id. A newly recorded paper BUY could therefore be counted twice on the
        first post-execution account snapshot. This method now bootstraps only
        when the event ledger has zero inventory for that market.

        Any non-zero partial mismatch is deliberately left untouched so the
        normal reconciliation gate fails closed instead of manufacturing an
        accounting event.
        """
        inserted = 0
        net = self.net_event_quantities()

        for market, pos in positions.items():
            key = str(market).upper()
            state_qty = _d(getattr(pos, "amount", 0))
            event_qty = net.get(key, ZERO)
            tolerance = max(
                Decimal("1e-12"),
                abs(state_qty) * Decimal("1e-9"),
            )

            if abs(event_qty - state_qty) <= tolerance:
                continue

            # Bootstrap is for migration/pre-ledger positions only. If the
            # ledger already has any non-zero quantity, do not add another BUY.
            if abs(event_qty) > tolerance:
                continue

            opened_at = str(getattr(pos, "opened_at", "unknown"))
            if self.record_buy(
                event_id=f"bootstrap-buy:{key}:{opened_at}",
                market=key,
                quantity=state_qty,
                price=_d(getattr(pos, "entry_price", 0)),
                fee_bps=fee_bps,
                payload={
                    "bootstrap": True,
                    "reason": "PREEXISTING_SIMULATED_POSITION",
                },
            ):
                inserted += 1
                net[key] = state_qty

        return inserted

    def _events(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_id,created_at,side,market,quantity,price,
                       gross_eur,fee_eur,realized_pnl_eur,payload
                FROM events ORDER BY rowid
                """
            ).fetchall()
        return [
            {
                "event_id": r[0],
                "created_at": r[1],
                "side": r[2],
                "market": r[3],
                "quantity": _d(r[4]),
                "price": _d(r[5]),
                "gross_eur": _d(r[6]),
                "fee_eur": _d(r[7]),
                "realized_pnl_eur": _d(r[8]),
                "payload": r[9],
            }
            for r in rows
        ]

    def snapshot(
        self,
        positions: Mapping[str, Any],
        marks: Mapping[str, Decimal],
    ) -> dict[str, Any]:
        events = self._events()
        cash = self.starting_equity
        fees = ZERO
        realized = ZERO
        net_event_qty: dict[str, Decimal] = {}

        for row in events:
            gross = row["gross_eur"]
            fee = row["fee_eur"]
            qty = row["quantity"]
            market = row["market"]
            fees += fee
            if row["side"] == "BUY":
                cash -= gross + fee
                net_event_qty[market] = (
                    net_event_qty.get(market, ZERO) + qty
                )
            elif row["side"] == "SELL":
                cash += gross - fee
                realized += row["realized_pnl_eur"]
                net_event_qty[market] = (
                    net_event_qty.get(market, ZERO) - qty
                )

        exposure = ZERO
        unrealized = ZERO
        cost_basis = ZERO
        state_qty: dict[str, Decimal] = {}
        marks_used: dict[str, str] = {}

        for market, pos in positions.items():
            key = str(market).upper()
            qty = _d(getattr(pos, "amount", 0))
            entry = _d(getattr(pos, "entry_price", 0))
            mark = _d(marks.get(key), entry)
            if mark <= 0:
                mark = entry
            state_qty[key] = qty
            marks_used[key] = str(mark)
            exposure += qty * mark
            cost_basis += qty * entry
            unrealized += qty * (mark - entry)

        mismatches = []
        for market in sorted(set(net_event_qty) | set(state_qty)):
            e = net_event_qty.get(market, ZERO)
            s = state_qty.get(market, ZERO)
            tolerance = max(
                Decimal("1e-12"),
                abs(s) * Decimal("1e-9"),
            )
            if abs(e - s) > tolerance:
                mismatches.append({
                    "market": market,
                    "event_quantity": str(e),
                    "state_quantity": str(s),
                })

        equity = cash + exposure
        high_water = self._meta_decimal(
            "high_water_equity_eur", self.starting_equity
        )
        if equity > high_water:
            high_water = equity
            self._set_meta("high_water_equity_eur", high_water)
        drawdown = (
            (high_water - equity) / high_water
            if high_water > 0
            else ZERO
        )

        return {
            "schema_version": self.SCHEMA,
            "status": (
                "READY"
                if not mismatches
                else "RECONCILIATION_REQUIRED"
            ),
            "starting_equity_eur": str(self.starting_equity),
            "cash_eur": str(cash),
            "exposure_eur": str(exposure),
            "equity_eur": str(equity),
            "open_cost_basis_eur": str(cost_basis),
            "realized_pnl_eur": str(realized),
            "unrealized_pnl_eur": str(unrealized),
            "fees_eur": str(fees),
            "high_water_equity_eur": str(high_water),
            "drawdown_fraction": str(drawdown),
            "event_count": len(events),
            "position_count": len(positions),
            "marks": marks_used,
            "quantity_mismatches": mismatches,
            "reconciliation_required": bool(mismatches),
            "orders_submitted": 0,
        }
