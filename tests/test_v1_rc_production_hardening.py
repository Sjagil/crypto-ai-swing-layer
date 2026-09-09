from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from crypto_ai_swing.contracts import Authority, Side, TradeIntent
from crypto_ai_swing.execution.crypto_authority import NativeAuthorityResult
from crypto_ai_swing.production.guard import LiveExecutionGuard
from crypto_ai_swing.production.readiness import ProductionReadinessEngine
from crypto_ai_swing.production.recovery import ExecutionRecoveryJournal


class FakeAuthority:
    def __init__(self):
        self.buy_calls = 0
        self.exit_calls = 0

    def authority_status(self):
        return {
            "status": "READY",
            "canary_active": True,
            "execution_environment_ready": True,
            "state_status": "READY",
        }

    def reconcile(self, markets):
        return {"status": "READY", "healthy": True, "failures": []}

    def account_snapshot(self, markets):
        return {
            "status": "READY",
            "equity_eur": "1000",
            "cash_eur": "1000",
            "exposure_eur": "0",
            "failures": [],
            "entry_allowed": True,
            "risk_reduction_allowed": True,
        }

    def portfolio(self):
        return {"status": "READY", "positions": {}}

    def preflight(self, intent):
        return {"status": "READY", "allowed": True, "blockers": []}

    def submit_buy(self, intent):
        self.buy_calls += 1
        return NativeAuthorityResult(
            True, {"accepted": True, "order_id": "test-buy"}
        )

    def submit_exit(self, *, market, reason, quantity=None):
        self.exit_calls += 1
        return NativeAuthorityResult(
            True, {"accepted": True, "order_id": "test-exit"}
        )


def settings(tmp_path: Path, *, leverage: bool = False):
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config/production.yaml").write_text(
        """
live_guard:
  require_remote_runtime_checks: true
  require_canary_authority_active: true
safe_canary_caps:
  maximum_order_eur: 10
  maximum_total_exposure_eur: 10
  maximum_positions: 1
  maximum_new_orders_per_day: 1
  maximum_risk_per_trade_eur: 1
  maximum_slippage_bps: 25
certification:
  minimum_real_roundtrips: 3
""",
        encoding="utf-8",
    )
    return SimpleNamespace(
        project_root=tmp_path,
        crypto_repo_root=tmp_path / "crypto",
        execution={
            "authority": {
                "withdrawals_enabled": False,
                "automatic_live_promotion": False,
                "strategy_direct_exchange_calls": False,
            },
            "swing_canary": {
                "native_crypto_authority": True,
                "maximum_order_eur": 10,
                "maximum_total_exposure_eur": 10,
                "maximum_positions": 1,
                "maximum_new_orders_per_day": 1,
                "maximum_risk_per_trade_eur": 1,
                "maximum_slippage_bps": 25,
                "native_exchange_stop_required": True,
                "autoscale": False,
                "automatic_authority": False,
            },
        },
        risk={
            "market_policy": {
                "spot_only": True,
                "long_only": True,
                "shorting_allowed": False,
                "leverage_allowed": leverage,
                "derivatives_allowed": False,
            }
        },
    )


def intent():
    now = datetime.now(UTC)
    return TradeIntent(
        intent_id="intent-1",
        created_at=now,
        market="BTC-EUR",
        side=Side.BUY,
        notional_eur=Decimal("10"),
        expected_edge_bps=200.0,
        estimated_round_trip_cost_bps=60.0,
        net_edge_bps=140.0,
        stop_pct=0.03,
        take_profit_pct=0.08,
        trailing_stop_pct=0.04,
        strategy="TEST",
        authority=Authority.LIVE,
        expires_at=now + timedelta(minutes=2),
        metadata={},
    )


def test_readiness_safe_canary_passes(tmp_path):
    engine = ProductionReadinessEngine(
        settings(tmp_path), authority=FakeAuthority()
    )
    result = engine.assess(["BTC-EUR"], network=True)
    assert result["live_canary_ready"] is True
    assert result["exit_ready"] is True
    assert result["orders_submitted"] == 0


def test_readiness_rejects_leverage(tmp_path):
    engine = ProductionReadinessEngine(
        settings(tmp_path, leverage=True), authority=FakeAuthority()
    )
    result = engine.assess(["BTC-EUR"], network=True)
    assert result["live_canary_ready"] is False
    assert "STATIC_LEVERAGE_DISABLED_FAILED" in result["blockers"]


def test_recovery_journal_duplicate_is_fail_closed(tmp_path):
    journal = ExecutionRecoveryJournal(tmp_path / "journal.sqlite")
    first = journal.begin(
        operation_key="BUY:x",
        market="BTC-EUR",
        side="BUY",
        intent_id="x",
    )
    second = journal.begin(
        operation_key="BUY:x",
        market="BTC-EUR",
        side="BUY",
        intent_id="x",
    )
    assert first["created"] is True
    assert second["created"] is False
    assert journal.status()["status"] == "RECOVERY_REQUIRED"


def test_guard_buy_journals_and_reconciles(tmp_path):
    fake = FakeAuthority()
    guard = LiveExecutionGuard(settings(tmp_path), fake)
    result = guard.submit_buy(intent(), markets=["BTC-EUR"])
    assert result["accepted"] is True
    assert result["manual_review_required"] is False
    assert fake.buy_calls == 1
    assert guard.journal.status()["status"] == "READY"


def test_guard_blocks_new_buy_when_unresolved_exists(tmp_path):
    fake = FakeAuthority()
    guard = LiveExecutionGuard(settings(tmp_path), fake)
    guard.journal.begin(
        operation_key="BUY:old",
        market="ETH-EUR",
        side="BUY",
        intent_id="old",
    )
    result = guard.submit_buy(
        intent(), markets=["BTC-EUR", "ETH-EUR"]
    )
    assert result["accepted"] is False
    assert result["reason_code"] == "PRODUCTION_READINESS_BLOCKED"
    assert fake.buy_calls == 0


def test_offline_preflight_names_remote_runtime_blocker(tmp_path):
    engine = ProductionReadinessEngine(
        settings(tmp_path), authority=FakeAuthority()
    )
    result = engine.assess(["BTC-EUR"], network=False)
    assert result["live_canary_ready"] is False
    assert "REMOTE_RUNTIME_NOT_CHECKED" in result["blockers"]


def test_execution_environment_false_is_fail_closed(tmp_path):
    fake = FakeAuthority()
    fake.authority_status = lambda: {
        "status": "READY",
        "canary_active": True,
        "execution_environment_ready": False,
        "state_status": "RECONCILIATION_REQUIRED",
    }
    engine = ProductionReadinessEngine(
        settings(tmp_path), authority=fake
    )
    result = engine.assess(["BTC-EUR"], network=True)
    assert result["live_canary_ready"] is False
    assert "CANONICAL_EXECUTION_ENVIRONMENT_NOT_READY" in result["blockers"]
    assert (
        "CANONICAL_AUTHORITY_STATE_RECONCILIATION_REQUIRED"
        in result["blockers"]
    )
