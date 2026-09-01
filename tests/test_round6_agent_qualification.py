from crypto_ai_swing.agents.council import build_council_decision


def test_unqualified_shadow_agent_is_advisory_only():
    decision = build_council_decision(
        {
            "alpha_probability": 0.05,
            "regime_probability": 0.10,
            "predicted_return": -0.02,
            "predicted_mae": 0.10,
        },
        {"spread_bps": 1.0},
        artifact_status="SHADOW",
        live_decision_influence=False,
        mode="shadow",
        shadow_decision_qualified=False,
    )
    assert decision.blocker_codes
    assert decision.entry_blocked is False
    assert decision.diagnostics["decision_influence"] is False
