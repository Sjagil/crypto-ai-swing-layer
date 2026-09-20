from __future__ import annotations

from pathlib import Path


def test_pi_systemd_services_are_shadow_and_non_execution():
    root = Path(__file__).resolve().parents[1]
    for name in (
        "crypto-swing-learning.service",
        "crypto-cmc-intelligence.service",
        "crypto-cmc-websocket.service",
    ):
        text = (root / "deploy" / name).read_text(encoding="utf-8")
        assert "CRYPTO_SWING_CANARY_EXECUTE=NO" in text
        assert "CRYPTO_SWING_FULL_LIVE=NO" in text
        assert "CRYPTO_FULL_LIVE=NO" in text
    learning=(root / "deploy/crypto-swing-learning.service").read_text(encoding="utf-8")
    assert "--mode shadow" in learning
    assert "MemoryMax=9G" in learning
    assert "CPUQuota=300%" in learning
