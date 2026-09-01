from crypto_ai_swing.agents.council import build_council_decision


def test_weak_unqualified_heads_cannot_veto_qualified_alpha():
    decision = build_council_decision(
        {
            "alpha_probability": 0.72,
            "regime_probability": 0.10,
            "predicted_return": -0.03,
            "predicted_mae": 0.15,
        },
        {"spread_bps": 1.0},
        artifact_status="SHADOW",
        live_decision_influence=False,
        mode="shadow",
        shadow_decision_qualified=True,
        head_qualifications={
            "alpha": True,
            "regime": False,
            "return": False,
            "risk": False,
            "execution": True,
        },
    )
    assert "AGENT_REGIME_UNFAVORABLE" in decision.blocker_codes
    assert "AGENT_PREDICTED_MAE_EXCESSIVE" in decision.blocker_codes
    assert decision.entry_blocked is False
    assert decision.diagnostics["head_influence"]["alpha"] is True
    assert decision.diagnostics["head_influence"]["regime"] is False
    assert decision.diagnostics["active_blocker_codes"] == []
