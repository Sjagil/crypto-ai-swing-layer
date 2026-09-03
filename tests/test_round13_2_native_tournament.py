from __future__ import annotations

from crypto_ai_swing.bridge.native_foundation import NativeFoundationBridge


def test_compact_campaign_extracts_nested_native_evidence():
    payload = {
        "campaign": "RESIDUAL_MOMENTUM_V1",
        "primary_strategy": "RM_R60_B180_EMA200",
        "trials": [{"id": index} for index in range(8)],
        "evidence": {
            "economic_pass": False,
            "statistical_pass": False,
            "paper_candidate_permitted": False,
        },
        "live_ready": False,
    }
    summary = NativeFoundationBridge._compact_campaign(payload)
    assert summary["selected_candidate"] == "RM_R60_B180_EMA200"
    assert summary["candidate_count"] == 8
    assert summary["economic_pass"] is False
    assert summary["statistical_pass"] is False
