from __future__ import annotations

from decimal import Decimal
from typing import Any

from crypto_ai_swing.contracts import Authority, Signal


def execution_validation_canary_config(
    proactive: dict[str, Any],
) -> dict[str, Any]:
    active = dict((proactive or {}).get("active_swing", {}) or {})
    return dict(active.get("execution_validation_canary", {}) or {})


def evaluate_execution_validation_canary(
    *,
    authority: Authority,
    signal: Signal,
    context: dict[str, Any],
    proactive: dict[str, Any],
) -> dict[str, Any]:
    active = dict((proactive or {}).get("active_swing", {}) or {})
    cfg = execution_validation_canary_config(proactive)
    blockers: list[str] = []

    if not bool(active.get("enabled", True)):
        blockers.append("ACTIVE_SWING_DISABLED")
    if str(active.get("style", "ACTIVE_SWING")).upper() != "ACTIVE_SWING":
        blockers.append("ACTIVE_SWING_STYLE_MISMATCH")
    if bool(active.get("high_frequency_trading", False)):
        blockers.append("HFT_MODE_FORBIDDEN")
    if authority is not Authority.LIVE:
        blockers.append("LIVE_AUTHORITY_REQUIRED")
    if not bool(cfg.get("enabled", False)):
        blockers.append("EXECUTION_VALIDATION_CANARY_DISABLED")
    if bool(context.get("entry_blocked", False)):
        blockers.append("UPSTREAM_ENTRY_BLOCKED")
    if bool(context.get("agent_entry_blocked", False)):
        blockers.append("ACTIVE_AGENT_BLOCKER")

    minimum_score = float(cfg.get("minimum_signal_score", 0.74))
    if float(signal.score) < minimum_score:
        blockers.append("CANARY_SIGNAL_SCORE_TOO_LOW")

    mtf = float(context.get("mtf_score", 0.0) or 0.0)
    if mtf < float(cfg.get("minimum_mtf_score", 0.0)):
        blockers.append("CANARY_MTF_SCORE_TOO_LOW")

    orderflow = float(context.get("orderflow_score", 0.0) or 0.0)
    if orderflow < float(cfg.get("minimum_orderflow_score", -0.35)):
        blockers.append("CANARY_ORDERFLOW_TOO_WEAK")

    spread = float(context.get("spread_bps", 999.0) or 999.0)
    if spread > float(cfg.get("maximum_spread_bps", 15.0)):
        blockers.append("CANARY_SPREAD_TOO_WIDE")

    maximum_order_eur = Decimal(str(cfg.get("maximum_order_eur", 10.0)))
    if maximum_order_eur <= 0:
        blockers.append("INVALID_CANARY_NOTIONAL_CAP")

    return {
        "schema_version": "active_swing_execution_validation_canary_v1",
        "allowed": not blockers,
        "blockers": blockers,
        "maximum_order_eur": str(maximum_order_eur),
        "minimum_signal_score": minimum_score,
        "minimum_mtf_score": float(cfg.get("minimum_mtf_score", 0.0)),
        "minimum_orderflow_score": float(
            cfg.get("minimum_orderflow_score", -0.35)
        ),
        "maximum_spread_bps": float(cfg.get("maximum_spread_bps", 15.0)),
        "scope": "EXECUTION_VALIDATION_ONLY",
        "manual_authority_required": True,
        "alpha_evidence_authorized": False,
        "alpha_promotion_authorized": False,
        "prospective_evidence_required_for_scaling": True,
        "autoscale_authorized": False,
    }


def canonical_preflight_explicitly_denied(
    payload: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    row = dict(payload or {})
    reasons = [
        str(value)
        for value in [
            *(row.get("blockers") or []),
            *(row.get("failures") or []),
        ]
    ]
    for key in ("ready", "accepted", "approved", "eligible", "allowed"):
        if key in row and row.get(key) is False:
            reasons.append(f"CANONICAL_{key.upper()}_FALSE")
    status = str(row.get("status") or "").upper()
    if status.startswith(("BLOCKED", "REJECTED", "DENIED", "FAILED", "ERROR")):
        reasons.append(f"CANONICAL_STATUS_{status}")
    reasons = sorted(set(reasons))
    return bool(reasons), reasons
