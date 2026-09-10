from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_round42_supervisor_delegates_agent_improvement_to_chief():
    text = (
        ROOT / "src/crypto_ai_swing/orchestration/supervisor.py"
    ).read_text(encoding="utf-8")
    assert "ChiefAgent" in text
    assert "self.chief_agent.cycle(" in text


def test_round42_chief_cannot_grant_live_authority():
    text = (
        ROOT / "src/crypto_ai_swing/agents/chief.py"
    ).read_text(encoding="utf-8")
    assert '"automatic_live_authority": False' in text
    assert '"automatic_live_promotion": False' in text
    assert "submit_buy" not in text
    assert "submit_exit" not in text
