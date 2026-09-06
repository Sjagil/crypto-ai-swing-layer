from __future__ import annotations
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from crypto_ai_swing.accounting.paper_ledger import PaperPortfolioLedger

def _p(amount="2", entry="10", opened="t0"):
    return SimpleNamespace(amount=Decimal(amount), entry_price=Decimal(entry), opened_at=opened)

def test_buy_sell_and_fees(tmp_path: Path):
    ledger = PaperPortfolioLedger(tmp_path / "x.sqlite", starting_equity=Decimal("1000"))
    fee = Decimal("25")
    assert ledger.record_buy(event_id="b", market="BTC-EUR", quantity=Decimal("2"), price=Decimal("10"), fee_bps=fee)
    s = ledger.snapshot({"BTC-EUR": _p()}, {"BTC-EUR": Decimal("11")})
    assert Decimal(s["cash_eur"]) == Decimal("979.950")
    assert Decimal(s["equity_eur"]) == Decimal("1001.950")
    assert ledger.record_sell(event_id="s", market="BTC-EUR", quantity=Decimal("2"), entry_price=Decimal("10"), exit_price=Decimal("12"), fee_bps=fee)
    c = ledger.snapshot({}, {})
    assert Decimal(c["cash_eur"]) == Decimal("1003.890")
    assert Decimal(c["realized_pnl_eur"]) == Decimal("3.890")
    assert c["status"] == "READY"

def test_idempotent_events(tmp_path: Path):
    ledger = PaperPortfolioLedger(tmp_path / "x.sqlite", starting_equity=Decimal("1000"))
    kwargs = dict(event_id="same", market="ETH-EUR", quantity=Decimal("1"), price=Decimal("100"), fee_bps=Decimal("25"))
    assert ledger.record_buy(**kwargs)
    assert ledger.record_buy(**kwargs) is False
    assert ledger.event_count() == 1

def test_divergence_blocks(tmp_path: Path):
    ledger = PaperPortfolioLedger(tmp_path / "x.sqlite", starting_equity=Decimal("1000"))
    ledger.record_buy(event_id="b", market="SOL-EUR", quantity=Decimal("1"), price=Decimal("100"), fee_bps=Decimal("0"))
    assert ledger.snapshot({}, {})["status"] == "RECONCILIATION_REQUIRED"

def test_bootstrap_once(tmp_path: Path):
    ledger = PaperPortfolioLedger(tmp_path / "x.sqlite", starting_equity=Decimal("1000"))
    positions = {"LINK-EUR": _p("5", "20", "opened")}
    assert ledger.ensure_open_positions(positions, fee_bps=Decimal("25")) == 1
    assert ledger.ensure_open_positions(positions, fee_bps=Decimal("25")) == 0
    assert ledger.snapshot(positions, {"LINK-EUR": Decimal("21")})["status"] == "READY"
