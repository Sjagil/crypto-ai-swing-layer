from pathlib import Path

from crypto_ai_swing.execution.crypto_authority import CryptoAuthorityAdapter


def _adapter(tmp_path: Path, payload: dict):
    adapter = CryptoAuthorityAdapter(tmp_path)
    adapter.authority_status = lambda: dict(payload)
    return adapter


def test_canonical_gate_fails_closed_when_authority_is_inactive(tmp_path):
    status = _adapter(tmp_path, {"active": False, "state_status": "DISABLED", "execution_environment_ready": False, "spot_only": True, "margin": False, "leverage": False, "shorting": False, "withdrawals": False}).gate_status()
    assert status["ready"] is False
    assert status["blockers"]


def test_canonical_gate_can_be_ready(tmp_path):
    status = _adapter(tmp_path, {"active": True, "state_status": "READY", "execution_environment_ready": True, "spot_only": True, "margin": False, "leverage": False, "shorting": False, "withdrawals": False}).gate_status()
    assert status["ready"] is True
    assert status["blockers"] == []


def test_canonical_gate_rejects_forbidden_capability(tmp_path):
    status = _adapter(tmp_path, {"active": True, "state_status": "READY", "execution_environment_ready": True, "spot_only": True, "margin": False, "leverage": True, "shorting": False, "withdrawals": False}).gate_status()
    assert status["ready"] is False
    assert "CANONICAL_FORBIDDEN_CAPABILITY_ENABLED" in status["blockers"]
