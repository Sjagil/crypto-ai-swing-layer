from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_round42_chief_owned_research_keeps_cold_start_fallback():
    text = (
        ROOT
        / "src/crypto_ai_swing/research/autonomous_strategy_director.py"
    ).read_text(encoding="utf-8")
    assert "ColdStartResearchRunner" in text
    assert 'startswith("COLD_START")' in text
    assert 'tasks["cold_start_research"]' in text
    assert '"automatic_live_promotion"' in text
    assert '"orders_submitted"' in text
