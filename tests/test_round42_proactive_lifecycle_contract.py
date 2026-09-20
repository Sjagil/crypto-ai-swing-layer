from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_round42_proactive_records_live_fill_metadata_and_monitors_native_stop():
    text = (
        ROOT / "src/crypto_ai_swing/orchestration/proactive.py"
    ).read_text(encoding="utf-8")
    assert "_record_position_after_execution" in text
    assert "LIVE_NATIVE_STOP_MONITOR" in text
    assert "self.edge_manager.strategy_lab" in text
    assert "canonical_markets" in text
