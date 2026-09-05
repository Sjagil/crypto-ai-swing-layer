from __future__ import annotations

from datetime import UTC, datetime

from crypto_ai_swing.contracts import Authority, ModelVote, Side, Signal
from crypto_ai_swing.execution.active_swing_canary import (
    canonical_preflight_explicitly_denied,
    evaluate_execution_validation_canary,
)


def _signal(score: float = 0.80) -> Signal:
    return Signal(
        market="LINK-EUR",
        timestamp=datetime(2026, 9, 4, tzinfo=UTC),
        side=Side.BUY,
        score=score,
        confidence=0.8,
        expected_edge_bps=90.0,
        stop_pct=0.02,
        take_profit_pct=0.06,
        trailing_stop_pct=0.015,
        strategy="TREND_PULLBACK",
        edge_source="HEURISTIC_SCORE_PROXY_RESEARCH_ONLY",
        votes=(ModelVote("deterministic", score, 0.75),),
        features={"atr_pct": 0.01, "price": 10.0},
    )


def _proactive() -> dict:
    return {
        "active_swing": {
            "enabled": True,
            "style": "ACTIVE_SWING",
            "high_frequency_trading": False,
            "execution_validation_canary": {
                "enabled": True,
                "maximum_order_eur": 10.0,
                "minimum_signal_score": 0.74,
                "minimum_mtf_score": 0.0,
                "minimum_orderflow_score": -0.35,
                "maximum_spread_bps": 15.0,
            },
        }
    }


def test_strong_live_swing_canary_can_reach_canonical_preflight():
    result = evaluate_execution_validation_canary(
        authority=Authority.LIVE,
        signal=_signal(),
        context={
            "entry_blocked": False,
            "agent_entry_blocked": False,
            "mtf_score": 0.30,
            "orderflow_score": 0.10,
            "spread_bps": 2.0,
        },
        proactive=_proactive(),
    )
    assert result["allowed"] is True
    assert result["maximum_order_eur"] == "10.0"
    assert result["alpha_evidence_authorized"] is False
    assert result["autoscale_authorized"] is False


def test_canary_is_not_available_outside_live():
    result = evaluate_execution_validation_canary(
        authority=Authority.PAPER,
        signal=_signal(),
        context={"mtf_score": 0.3, "orderflow_score": 0.1, "spread_bps": 2.0},
        proactive=_proactive(),
    )
    assert result["allowed"] is False
    assert "LIVE_AUTHORITY_REQUIRED" in result["blockers"]


def test_canary_does_not_override_upstream_gate():
    result = evaluate_execution_validation_canary(
        authority=Authority.LIVE,
        signal=_signal(),
        context={
            "entry_blocked": True,
            "mtf_score": 0.3,
            "orderflow_score": 0.1,
            "spread_bps": 2.0,
        },
        proactive=_proactive(),
    )
    assert result["allowed"] is False
    assert "UPSTREAM_ENTRY_BLOCKED" in result["blockers"]


def test_canary_rejects_weak_signal_and_bad_liquidity():
    result = evaluate_execution_validation_canary(
        authority=Authority.LIVE,
        signal=_signal(0.60),
        context={
            "entry_blocked": False,
            "mtf_score": 0.1,
            "orderflow_score": -0.5,
            "spread_bps": 25.0,
        },
        proactive=_proactive(),
    )
    assert result["allowed"] is False
    assert "CANARY_SIGNAL_SCORE_TOO_LOW" in result["blockers"]
    assert "CANARY_ORDERFLOW_TOO_WEAK" in result["blockers"]
    assert "CANARY_SPREAD_TOO_WIDE" in result["blockers"]


def test_explicit_canonical_denial_is_fail_closed():
    denied, reasons = canonical_preflight_explicitly_denied(
        {"ready": False, "blockers": ["MANUAL_APPROVAL_REQUIRED"]}
    )
    assert denied is True
    assert "MANUAL_APPROVAL_REQUIRED" in reasons
    denied, reasons = canonical_preflight_explicitly_denied(
        {"status": "READY", "blockers": []}
    )
    assert denied is False
    assert reasons == []
