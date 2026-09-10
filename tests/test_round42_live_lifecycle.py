from decimal import Decimal
from types import SimpleNamespace

from crypto_ai_swing.production.live_lifecycle import (
    LiveLifecycleCoordinator,
)


class _Authority:
    def gate_status(self):
        return {"ready": True, "blockers": []}

    def account_snapshot(self, markets):
        return {"status": "READY", "markets": markets}

    def portfolio(self):
        return {"status": "READY", "positions": {}}

    def preflight(self, intent):
        return {"status": "READY", "accepted": True}


class _Guard:
    def __init__(self):
        self.buy_calls = 0
        self.exit_calls = 0

    def reconcile(self, markets):
        return {"status": "READY", "ready": True, "markets": markets}

    def submit_buy(self, intent, *, markets, canonical_preflight):
        self.buy_calls += 1
        return {
            "accepted": True,
            "canonical_result": {
                "native_protective_stop": {
                    "order_id": "stop-1",
                    "status": "open",
                }
            },
            "orders_submitted": 2,
        }

    def submit_exit(self, *, market, reason, quantity, markets):
        self.exit_calls += 1
        return {
            "accepted": True,
            "market": market,
            "reason": reason,
            "quantity": quantity,
            "orders_submitted": 1,
        }


def _settings(tmp_path):
    return SimpleNamespace(
        project_root=tmp_path,
        crypto_repo_root=tmp_path,
    )


def test_round42_live_coordinator_is_dry_by_default(tmp_path):
    guard = _Guard()
    lifecycle = LiveLifecycleCoordinator(
        _settings(tmp_path),
        authority=_Authority(),
        guard=guard,
    )
    intent = SimpleNamespace(market="BTC-EUR", intent_id="i-1")
    preview = lifecycle.submit_entry(
        intent, markets=["BTC-EUR"], execute=False
    )
    assert preview["orders_submitted"] == 0
    assert guard.buy_calls == 0


def test_round42_live_coordinator_reaches_guarded_buy_and_exit(tmp_path):
    guard = _Guard()
    lifecycle = LiveLifecycleCoordinator(
        _settings(tmp_path),
        authority=_Authority(),
        guard=guard,
    )
    intent = SimpleNamespace(market="BTC-EUR", intent_id="i-1")
    buy = lifecycle.submit_entry(
        intent, markets=["BTC-EUR"], execute=True
    )
    sell = lifecycle.submit_exit(
        market="BTC-EUR",
        reason="TAKE_PROFIT",
        quantity="0.001",
        markets=["BTC-EUR"],
        execute=True,
    )
    assert buy["accepted"] is True
    assert buy["orders_submitted"] == 2
    assert sell["accepted"] is True
    assert guard.buy_calls == 1
    assert guard.exit_calls == 1


def test_round42_native_stop_is_monitor_only_not_duplicate_market_sell(
    tmp_path,
):
    lifecycle = LiveLifecycleCoordinator(
        _settings(tmp_path),
        authority=_Authority(),
        guard=_Guard(),
    )
    pos = SimpleNamespace(
        entry_price=Decimal("100"),
        amount=Decimal("1"),
        highest_price=Decimal("105"),
        stop_pct=0.02,
        take_profit_pct=0.06,
        trailing_stop_pct=0.02,
    )
    decision = lifecycle.decision_for_position(
        pos, price=Decimal("97")
    )
    assert decision["action"] == "MONITOR_NATIVE_STOP"
    assert decision["reason"] == "NATIVE_STOP_LOSS"
