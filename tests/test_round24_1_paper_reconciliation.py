from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from crypto_ai_swing.accounting.paper_ledger import PaperPortfolioLedger


def pos(amount="2", entry="10", opened="t0"):
    return SimpleNamespace(
        amount=Decimal(amount),
        entry_price=Decimal(entry),
        opened_at=opened,
    )


def test_existing_regular_buy_is_not_bootstrapped_again(tmp_path: Path):
    ledger = PaperPortfolioLedger(
        tmp_path / "paper.sqlite",
        starting_equity=Decimal("1000"),
    )
    ledger.record_buy(
        event_id="paper-buy:intent-1",
        market="ENA-EUR",
        quantity=Decimal("10"),
        price=Decimal("1"),
        fee_bps=Decimal("25"),
        payload={"intent_id": "intent-1"},
    )
    positions = {"ENA-EUR": pos("10", "1", "opened")}
    assert ledger.ensure_open_positions(
        positions,
        fee_bps=Decimal("25"),
    ) == 0
    assert ledger.event_count() == 1
    snap = ledger.snapshot(
        positions,
        {"ENA-EUR": Decimal("1")},
    )
    assert snap["status"] == "READY"
    assert snap["quantity_mismatches"] == []


def test_preledger_position_is_bootstrapped_once(tmp_path: Path):
    ledger = PaperPortfolioLedger(
        tmp_path / "paper.sqlite",
        starting_equity=Decimal("1000"),
    )
    positions = {"LINK-EUR": pos("5", "20", "opened")}
    assert ledger.ensure_open_positions(
        positions,
        fee_bps=Decimal("25"),
    ) == 1
    assert ledger.ensure_open_positions(
        positions,
        fee_bps=Decimal("25"),
    ) == 0
    assert ledger.event_count() == 1
    assert ledger.snapshot(
        positions,
        {"LINK-EUR": Decimal("21")},
    )["status"] == "READY"


def test_partial_mismatch_fails_closed_without_manufacturing_buy(tmp_path: Path):
    ledger = PaperPortfolioLedger(
        tmp_path / "paper.sqlite",
        starting_equity=Decimal("1000"),
    )
    ledger.record_buy(
        event_id="paper-buy:old",
        market="SOL-EUR",
        quantity=Decimal("1"),
        price=Decimal("100"),
        fee_bps=Decimal("0"),
    )
    positions = {"SOL-EUR": pos("2", "100", "opened")}
    assert ledger.ensure_open_positions(
        positions,
        fee_bps=Decimal("0"),
    ) == 0
    assert ledger.event_count() == 1
    snap = ledger.snapshot(
        positions,
        {"SOL-EUR": Decimal("100")},
    )
    assert snap["status"] == "RECONCILIATION_REQUIRED"
