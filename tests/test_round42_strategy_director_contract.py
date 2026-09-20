from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_round42_strategy_director_reuses_canonical_research():
    text = (
        ROOT
        / "src/crypto_ai_swing/research/autonomous_strategy_director.py"
    ).read_text(encoding="utf-8")
    assert "NativeResearchBridge" in text
    assert "classical_factory_plan" in text
    assert "run_factory_campaign" in text
    assert "run_native_alpha_tournament" in text
    assert "ResearchHypothesis.create" in text
    assert "PreregisteredExperiment.create" in text
    assert "automatic_live_promotion" in text


def test_round42_strategy_lab_is_available_to_edge_manager():
    chief = (
        ROOT / "src/crypto_ai_swing/agents/chief.py"
    ).read_text(encoding="utf-8")
    assert "edge_manager.strategy_lab" in chief
    proactive = (
        ROOT / "src/crypto_ai_swing/orchestration/proactive.py"
    ).read_text(encoding="utf-8")
    assert "self.edge_manager.strategy_lab" in proactive
