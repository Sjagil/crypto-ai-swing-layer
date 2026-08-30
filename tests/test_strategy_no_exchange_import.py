from pathlib import Path


def test_strategy_code_has_no_bitvavo_import():
    root = Path(__file__).resolve().parents[1]
    text = (root / "src/crypto_ai_swing/strategies/swing.py").read_text(encoding="utf-8").lower()
    assert "import bitvavo" not in text
    assert "placeorder" not in text
