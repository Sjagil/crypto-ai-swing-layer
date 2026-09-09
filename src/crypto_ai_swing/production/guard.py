from __future__ import annotations

from typing import Any

from crypto_ai_swing.production.audit import redact
from crypto_ai_swing.production.readiness import (
    ProductionReadinessEngine,
    _explicit_failures,
)

class LiveExecutionGuard:
    """Fail-closed wrapper around the canonical Sjagil/crypto authority."""

    def __init__(self, settings, authority) -> None:
        self.settings = settings
        self.authority = authority
        self.readiness = ProductionReadinessEngine(
            settings, authority=authority
        )
        self.journal = self.readiness.journal
        self.audit = self.readiness.audit

    def close(self) -> None:
        return None

    def status(
        self, markets: list[str], *, network: bool
    ) -> dict[str, Any]:
        return self.readiness.assess(
            markets, network=network, persist=True
        )

    def reconcile(self, markets: list[str]) -> dict[str, Any]:
        try:
            raw = dict(self.authority.reconcile(markets) or {})
        except Exception as exc:
            return {
                "status": "ERROR",
                "ready": False,
                "blockers": [
                    f"{type(exc).__name__}:{str(exc)[:300]}"
                ],
            }
        failures = _explicit_failures(raw)
        return {
            "status": "READY" if not failures else "BLOCKED",
            "ready": not failures,
            "blockers": failures,
            "canonical": redact(raw),
        }

    def _begin(
        self, *, operation_key: str, market: str, side: str,
        intent_id: str | None, payload: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        row = self.journal.begin(
            operation_key=operation_key,
            market=market,
            side=side,
            intent_id=intent_id,
            payload=payload,
        )
        if not row.get("created"):
            return False, {
                "accepted": False,
                "reason_code": "DUPLICATE_OR_RECOVERY_OPERATION",
                "recovery_state": row.get("state"),
                "operation_key": operation_key,
                "orders_submitted": 0,
            }
        return True, row

    def submit_buy(
        self, intent, *, markets: list[str],
        canonical_preflight: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        readiness = self.readiness.assess(
            markets, network=True, persist=True
        )
        if not readiness.get("live_canary_ready"):
            return {
                "accepted": False,
                "reason_code": "PRODUCTION_READINESS_BLOCKED",
                "production_readiness": readiness,
                "orders_submitted": 0,
            }

        preflight = (
            {
                "status": "READY",
                "allowed": True,
                "blockers": [],
                "canonical_preflight": redact(canonical_preflight),
            }
            if canonical_preflight is not None
            else self.readiness.preflight_intent(intent)
        )
        if not preflight.get("allowed"):
            return {
                "accepted": False,
                "reason_code": "CANONICAL_PREFLIGHT_BLOCKED",
                "preflight": preflight,
                "orders_submitted": 0,
            }

        key = f"BUY:{intent.intent_id}"
        created, state = self._begin(
            operation_key=key,
            market=intent.market,
            side="BUY",
            intent_id=intent.intent_id,
            payload={"intent": intent.to_dict()},
        )
        if not created:
            return state

        self.audit.append(
            "LIVE_BUY_SUBMITTING",
            {
                "operation_key": key,
                "market": intent.market,
                "intent_id": intent.intent_id,
            },
        )
        try:
            result = self.authority.submit_buy(intent)
        except Exception as exc:
            self.journal.transition(
                key,
                "MANUAL_REVIEW_REQUIRED",
                {
                    "exception": (
                        f"{type(exc).__name__}:{str(exc)[:300]}"
                    ),
                    "submission_outcome_unknown": True,
                },
            )
            return {
                "accepted": False,
                "reason_code": "CANONICAL_SUBMISSION_EXCEPTION",
                "submission_outcome_unknown": True,
                "manual_review_required": True,
                "operation_key": key,
                "orders_submitted": 0,
            }

        raw = dict(result.payload or {})
        accepted = bool(result.accepted)
        self.journal.transition(
            key,
            "ACCEPTED" if accepted else "REJECTED",
            {"canonical_result": redact(raw), "accepted": accepted},
        )
        if not accepted:
            return {
                "accepted": False,
                "operation_key": key,
                "canonical_result": redact(raw),
                **redact(raw),
            }

        reconciliation = self.reconcile(markets)
        manual = not bool(reconciliation.get("ready"))
        self.journal.transition(
            key,
            "MANUAL_REVIEW_REQUIRED" if manual else "RECONCILED",
            {"post_submit_reconciliation": reconciliation},
        )
        self.audit.append(
            "LIVE_BUY_RESULT",
            {
                "operation_key": key,
                "accepted": accepted,
                "manual_review_required": manual,
                "reconciliation": reconciliation,
            },
        )
        return {
            "accepted": accepted,
            "operation_key": key,
            "manual_review_required": manual,
            "post_submit_reconciliation": reconciliation,
            "canonical_result": redact(raw),
            **redact(raw),
        }

    def submit_exit(
        self, *, market: str, reason: str, quantity: str | None,
        markets: list[str],
    ) -> dict[str, Any]:
        readiness = self.readiness.assess(
            markets, network=True, persist=True
        )
        if not readiness.get("exit_ready"):
            return {
                "accepted": False,
                "reason_code": "PRODUCTION_EXIT_READINESS_BLOCKED",
                "production_readiness": readiness,
                "orders_submitted": 0,
            }

        key = (
            f"EXIT:{str(market).upper()}:{reason}:"
            f"{quantity or 'ALL'}"
        )
        created, state = self._begin(
            operation_key=key,
            market=market,
            side="SELL",
            intent_id=None,
            payload={
                "reason": reason,
                "requested_quantity": quantity,
            },
        )
        if not created:
            return state

        try:
            result = self.authority.submit_exit(
                market=market,
                reason=reason,
                quantity=quantity,
            )
        except Exception as exc:
            self.journal.transition(
                key,
                "MANUAL_REVIEW_REQUIRED",
                {
                    "exception": (
                        f"{type(exc).__name__}:{str(exc)[:300]}"
                    ),
                    "submission_outcome_unknown": True,
                },
            )
            return {
                "accepted": False,
                "reason_code": "CANONICAL_EXIT_EXCEPTION",
                "submission_outcome_unknown": True,
                "manual_review_required": True,
                "operation_key": key,
                "orders_submitted": 0,
            }

        raw = dict(result.payload or {})
        accepted = bool(result.accepted)
        self.journal.transition(
            key,
            "ACCEPTED" if accepted else "REJECTED",
            {"canonical_result": redact(raw), "accepted": accepted},
        )
        if not accepted:
            return {
                "accepted": False,
                "operation_key": key,
                "canonical_result": redact(raw),
                **redact(raw),
            }

        reconciliation = self.reconcile(markets)
        manual = not bool(reconciliation.get("ready"))
        self.journal.transition(
            key,
            "MANUAL_REVIEW_REQUIRED" if manual else "RECONCILED",
            {"post_submit_reconciliation": reconciliation},
        )
        return {
            "accepted": accepted,
            "operation_key": key,
            "manual_review_required": manual,
            "post_submit_reconciliation": reconciliation,
            "canonical_result": redact(raw),
            **redact(raw),
        }
