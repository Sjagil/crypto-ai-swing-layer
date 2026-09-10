from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from crypto_ai_swing.execution.crypto_authority import CryptoAuthorityAdapter
from crypto_ai_swing.production.guard import LiveExecutionGuard


def _decimal(value: Any, default: str = "0") -> Decimal:
    try:
        result = Decimal(str(default if value in (None, "") else value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)
    return result if result.is_finite() else Decimal(default)


class LiveLifecycleCoordinator:
    """Narrow production lifecycle over Round-40 canonical execution authority.

    This module never implements exchange transport. BUY, SELL, account truth,
    reconciliation and native protective stops remain Sjagil/crypto-owned.
    """

    SCHEMA = "crypto_ai_swing_live_lifecycle_v1"

    def __init__(self, settings, *, authority=None, guard=None) -> None:
        self.settings = settings
        self.authority = authority or CryptoAuthorityAdapter(
            settings.crypto_repo_root
        )
        self.guard = guard or LiveExecutionGuard(
            settings, self.authority
        )
        self.root = (
            Path(settings.project_root)
            / "output/crypto_ai_swing/live_lifecycle"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"

    def _event(self, event: str, payload: Mapping[str, Any]) -> None:
        row = {
            "at": datetime.now(UTC).isoformat(),
            "event": event,
            **dict(payload),
        }
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(row, sort_keys=True, default=str) + "\n"
            )

    def audit(self, markets: list[str]) -> dict[str, Any]:
        gate = self.authority.gate_status()
        try:
            reconciliation = self.guard.reconcile(markets)
        except Exception as exc:  # noqa: BLE001
            reconciliation = {
                "status": "ERROR",
                "ready": False,
                "blockers": [
                    f"{type(exc).__name__}:{str(exc)[:300]}"
                ],
            }
        try:
            snapshot = self.authority.account_snapshot(markets)
        except Exception as exc:  # noqa: BLE001
            snapshot = {
                "status": "ERROR",
                "failures": [
                    f"{type(exc).__name__}:{str(exc)[:300]}"
                ],
            }
        try:
            portfolio = self.authority.portfolio()
        except Exception as exc:  # noqa: BLE001
            portfolio = {
                "status": "ERROR",
                "failures": [
                    f"{type(exc).__name__}:{str(exc)[:300]}"
                ],
            }

        payload = {
            "schema_version": self.SCHEMA,
            "generated_at": datetime.now(UTC).isoformat(),
            "markets": markets,
            "gate": gate,
            "reconciliation": reconciliation,
            "account_snapshot": snapshot,
            "portfolio": portfolio,
            "capabilities": {
                "real_market_buy": True,
                "real_risk_reducing_sell": True,
                "native_stop_created_after_filled_buy": True,
                "take_profit_exit_via_canonical_sell": True,
                "trailing_exit_via_canonical_sell": True,
                "idempotent_guard": True,
                "post_submit_reconciliation": True,
                "direct_exchange_transport_in_swing_layer": False,
            },
            "ready_for_live_submission": bool(
                gate.get("ready")
                and reconciliation.get("ready")
            ),
            "live_authority_auto_enabled": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        self._event("AUDIT", payload)
        return payload

    def submit_entry(
        self,
        intent,
        *,
        markets: list[str],
        execute: bool = False,
    ) -> dict[str, Any]:
        preflight = self.authority.preflight(intent)
        if not execute:
            return {
                "schema_version": self.SCHEMA,
                "status": (
                    "READY_NOT_SUBMITTED"
                    if preflight.get("accepted") is True
                    or preflight.get("status") == "READY"
                    else "BLOCKED"
                ),
                "canonical_preflight": preflight,
                "native_stop_required_after_fill": True,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        result = self.guard.submit_buy(
            intent,
            markets=markets,
            canonical_preflight=preflight,
        )
        canonical = dict(
            result.get("canonical_result") or result
        )
        protective = dict(
            canonical.get("native_protective_stop") or {}
        )
        protective_status = str(
            protective.get("status") or ""
        ).replace("_", "").replace("-", "").casefold()
        protective_bad = protective_status in {
            "rejected",
            "cancelled",
            "canceled",
            "failed",
            "error",
        }
        if result.get("accepted") is True and (
            not protective or protective_bad
        ):
            result = {
                **result,
                "accepted": False,
                "manual_review_required": True,
                "reason_code": "NATIVE_PROTECTIVE_STOP_NOT_CONFIRMED",
            }
        self._event(
            "ENTRY_RESULT",
            {
                "market": getattr(intent, "market", None),
                "intent_id": getattr(intent, "intent_id", None),
                "accepted": bool(result.get("accepted")),
                "manual_review_required": bool(
                    result.get("manual_review_required")
                ),
            },
        )
        return result

    def submit_exit(
        self,
        *,
        market: str,
        reason: str,
        quantity: str | None,
        markets: list[str],
        execute: bool = False,
    ) -> dict[str, Any]:
        if not execute:
            return {
                "schema_version": self.SCHEMA,
                "status": "READY_NOT_SUBMITTED",
                "market": market,
                "reason": reason,
                "quantity": quantity,
                "risk_reduction_only": True,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        result = self.guard.submit_exit(
            market=market,
            reason=reason,
            quantity=quantity,
            markets=markets,
        )
        self._event(
            "EXIT_RESULT",
            {
                "market": market,
                "reason": reason,
                "quantity": quantity,
                "accepted": bool(result.get("accepted")),
                "manual_review_required": bool(
                    result.get("manual_review_required")
                ),
            },
        )
        return result

    @staticmethod
    def decision_for_position(
        position,
        *,
        price: Decimal,
        tp1_fraction: Decimal = Decimal("0.50"),
        tp1_multiple: Decimal = Decimal("0.50"),
    ) -> dict[str, Any]:
        """Pure lifecycle decision; native stop remains exchange-side authority."""
        entry = _decimal(getattr(position, "entry_price", 0))
        amount = _decimal(getattr(position, "amount", 0))
        highest = max(
            _decimal(getattr(position, "highest_price", 0)),
            price,
        )
        stop_pct = _decimal(getattr(position, "stop_pct", 0))
        take_pct = _decimal(
            getattr(position, "take_profit_pct", 0)
        )
        trailing_pct = _decimal(
            getattr(position, "trailing_stop_pct", 0)
        )
        if entry <= 0 or amount <= 0 or price <= 0:
            return {"action": "HOLD", "reason": "INVALID_POSITION"}

        hard_stop = entry * (Decimal("1") - stop_pct)
        target = entry * (Decimal("1") + take_pct)
        tp1 = entry * (
            Decimal("1") + take_pct * tp1_multiple
        )
        trailing = highest * (Decimal("1") - trailing_pct)

        if price <= hard_stop:
            return {
                "action": "MONITOR_NATIVE_STOP",
                "reason": "NATIVE_STOP_LOSS",
                "hard_stop": str(hard_stop),
            }
        if price >= target:
            return {
                "action": "EXIT_ALL",
                "reason": "TAKE_PROFIT",
                "quantity": str(amount),
                "target": str(target),
            }
        if highest > entry and price <= trailing:
            return {
                "action": "EXIT_ALL",
                "reason": "TRAILING_STOP",
                "quantity": str(amount),
                "trailing": str(trailing),
            }
        return {
            "action": "HOLD",
            "reason": "NO_EXIT_TRIGGER",
            "tp1_reference": str(tp1),
            "tp1_fraction_reference": str(tp1_fraction),
            "target": str(target),
            "hard_stop": str(hard_stop),
            "trailing": str(trailing),
        }
