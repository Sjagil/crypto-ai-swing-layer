#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import sqlite3
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from crypto_ai_swing.settings import Settings


ZERO = Decimal("0")


def d(value: Any) -> Decimal:
    return Decimal(str(value))


def backup_sqlite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    src = sqlite3.connect(source, timeout=5.0)
    dst = sqlite3.connect(destination)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def state_positions(path: Path) -> dict[str, Decimal]:
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT market,amount FROM positions"
        ).fetchall()
    finally:
        conn.close()
    return {
        str(market).upper(): d(amount)
        for market, amount in rows
    }


def events(path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            """
            SELECT event_id,side,market,quantity,payload
            FROM events
            ORDER BY rowid
            """
        ).fetchall()
    finally:
        conn.close()

    result = []
    for event_id, side, market, quantity, raw_payload in rows:
        try:
            payload = json.loads(raw_payload or "{}")
        except Exception:
            payload = {}
        result.append({
            "event_id": str(event_id),
            "side": str(side).upper(),
            "market": str(market).upper(),
            "quantity": d(quantity),
            "payload": payload if isinstance(payload, dict) else {},
        })
    return result


def net_quantities(rows: list[dict[str, Any]]) -> dict[str, Decimal]:
    net: dict[str, Decimal] = {}
    for row in rows:
        market = row["market"]
        qty = row["quantity"]
        if row["side"] == "BUY":
            net[market] = net.get(market, ZERO) + qty
        elif row["side"] == "SELL":
            net[market] = net.get(market, ZERO) - qty
    return net


def tolerance(qty: Decimal) -> Decimal:
    return max(Decimal("1e-12"), abs(qty) * Decimal("1e-9"))


def main() -> int:
    settings = Settings.load()
    mode_root = (
        Path(settings.project_root)
        / "output/crypto_ai_swing/modes/paper/proactive"
    )

    state_rel = settings.proactive.get(
        "state_path",
        "output/crypto_ai_swing/proactive/state.sqlite",
    )
    state_db = mode_root / (
        Path(str(state_rel)).name or "state.sqlite"
    )
    ledger_db = mode_root / "paper_ledger.sqlite"

    if not state_db.is_file():
        raise SystemExit(f"Paper state DB missing: {state_db}")
    if not ledger_db.is_file():
        raise SystemExit(f"Paper ledger DB missing: {ledger_db}")

    positions = state_positions(state_db)
    rows = events(ledger_db)
    before_net = net_quantities(rows)

    mismatches = {}
    for market in sorted(set(before_net) | set(positions)):
        event_qty = before_net.get(market, ZERO)
        state_qty = positions.get(market, ZERO)
        if abs(event_qty - state_qty) > tolerance(state_qty):
            mismatches[market] = {
                "event_quantity": str(event_qty),
                "state_quantity": str(state_qty),
            }

    print("BEFORE")
    print(json.dumps({
        "state_db": str(state_db),
        "ledger_db": str(ledger_db),
        "event_count": len(rows),
        "mismatches": mismatches,
    }, indent=2))

    if not mismatches:
        print("NO_REPAIR_NEEDED")
        return 0

    removals: list[str] = []

    for market, mismatch in mismatches.items():
        event_qty = d(mismatch["event_quantity"])
        state_qty = d(mismatch["state_quantity"])
        excess = event_qty - state_qty

        if excess <= tolerance(state_qty):
            raise SystemExit(
                f"BLOCKED: {market} is short in event inventory; "
                "automatic deletion cannot repair it"
            )

        candidates = [
            row
            for row in rows
            if row["market"] == market
            and row["side"] == "BUY"
            and bool(row["payload"].get("bootstrap"))
        ]

        if not candidates:
            raise SystemExit(
                f"BLOCKED: {market} excess has no bootstrap BUY to remove"
            )

        chosen = None
        # Usually one candidate. Brute-force remains deterministic and safe
        # for migration histories with a handful of bootstrap events.
        for count in range(1, len(candidates) + 1):
            for combo in itertools.combinations(candidates, count):
                qty = sum(
                    (row["quantity"] for row in combo),
                    ZERO,
                )
                if abs(qty - excess) <= tolerance(state_qty):
                    chosen = combo
                    break
            if chosen is not None:
                break

        if chosen is None:
            raise SystemExit(
                f"BLOCKED: {market} excess {excess} cannot be explained "
                "exactly by bootstrap BUY events"
            )

        removals.extend(row["event_id"] for row in chosen)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = ledger_db.with_name(
        f"{ledger_db.name}.pre_v0241_{stamp}.bak"
    )
    backup_sqlite(ledger_db, backup)

    conn = sqlite3.connect(ledger_db, timeout=5.0)
    try:
        conn.execute("BEGIN IMMEDIATE")
        for event_id in removals:
            conn.execute(
                "DELETE FROM events WHERE event_id=?",
                (event_id,),
            )
        conn.commit()
    except BaseException:
        conn.rollback()
        conn.close()
        backup_sqlite(backup, ledger_db)
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass

    after_rows = events(ledger_db)
    after_net = net_quantities(after_rows)
    after_mismatches = {}

    for market in sorted(set(after_net) | set(positions)):
        event_qty = after_net.get(market, ZERO)
        state_qty = positions.get(market, ZERO)
        if abs(event_qty - state_qty) > tolerance(state_qty):
            after_mismatches[market] = {
                "event_quantity": str(event_qty),
                "state_quantity": str(state_qty),
            }

    if after_mismatches:
        backup_sqlite(backup, ledger_db)
        raise SystemExit(
            "POST_REPAIR_VERIFICATION_FAILED: backup restored"
        )

    print()
    print("REPAIR COMPLETE")
    print(json.dumps({
        "removed_bootstrap_event_ids": removals,
        "event_count_before": len(rows),
        "event_count_after": len(after_rows),
        "mismatches_after": after_mismatches,
        "backup_retained": str(backup),
        "orders_submitted": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
