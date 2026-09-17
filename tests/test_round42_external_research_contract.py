from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_round42_browser_plane_is_research_only():
    text = (
        ROOT
        / "src/crypto_ai_swing/intelligence/external_research.py"
    ).read_text(encoding="utf-8")
    assert 'import_module("scrapers.intelligence")' in text
    assert "CryptoAuthorityAdapter" not in text
    assert "LiveExecutionGuard" not in text
    assert "submit_buy" not in text
    assert "submit_exit" not in text
    assert "UNTRUSTED_EXTERNAL" in text
    assert "prompt_injection_suspected" in text
