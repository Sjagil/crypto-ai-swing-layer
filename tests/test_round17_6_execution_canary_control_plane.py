from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import crypto_ai_swing.orchestration.control_plane as control_plane
from crypto_ai_swing.orchestration.control_plane import ModeController


class _Library:
    def __init__(self, root):
        self.root = root

    def integration_status(self):
        return {"ready": True}


class _Foundation:
    def __init__(self, root):
        self.root = root

    def status(self):
        return SimpleNamespace(ready=True)


class _Authority:
    def __init__(self, root):
        self.root = root

    def authority_status(self):
        return {
            "active": True,
            "execution_environment_ready": True,
            "state_status": "READY",
        }


def _settings(tmp_path: Path, *, canary_enabled: bool = True):
    crypto_root = tmp_path / "crypto"
    crypto_root.mkdir()
    return SimpleNamespace(
        project_root=tmp_path,
        crypto_repo_root=crypto_root,
        autonomy={"forward_evidence": {}},
        execution={
            "execution": {
                "paper_enabled": True,
                "live_enabled": True,
                "live_submission_enabled": True,
            }
        },
        proactive={
            "active_swing": {
                "enabled": True,
                "style": "ACTIVE_SWING",
                "high_frequency_trading": False,
                "execution_validation_canary": {
                    "enabled": canary_enabled,
                    "manual_authority_required": True,
                    "maximum_order_eur": 10.0,
                    "prospective_evidence_required_for_scaling": True,
                    "alpha_evidence_authorized": False,
                    "autoscale": False,
                },
            }
        },
    )


def _live_env(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ALLOWED", "true")
    monkeypatch.setenv("WM_REAL_ORDER_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("WM_I_UNDERSTAND_LIVE_RISK", "true")
    monkeypatch.setenv("WM_LIVE_APPROVAL", "APPROVED_LIVE_ACTIVE")
    monkeypatch.setenv("WITHDRAWALS_DISABLED", "true")
    monkeypatch.setenv("BITVAVO_TRADE_KEY_SCOPE", "trade")
    monkeypatch.setenv("BITVAVO_API_KEY", "test-key")
    monkeypatch.setenv("BITVAVO_API_SECRET", "test-secret")
    monkeypatch.setenv("WM_OPERATOR_ID", "1")
    monkeypatch.setenv("CRYPTO_SWING_CANARY_EXECUTE", "YES")


def _patch_dependencies(monkeypatch):
    monkeypatch.setattr(control_plane, "CryptoLibraryBridge", _Library)
    monkeypatch.setattr(control_plane, "NativeFoundationBridge", _Foundation)
    monkeypatch.setattr(control_plane, "CryptoAuthorityAdapter", _Authority)


def test_execution_validation_policy_is_bounded(tmp_path):
    controller = object.__new__(ModeController)
    controller.settings = _settings(tmp_path)

    policy = controller._execution_validation_mode_policy()

    assert policy["eligible"] is True
    assert policy["maximum_order_eur"] == 10.0
    assert policy["manual_authority_required"] is True
    assert policy["autoscale"] is False
    assert policy["alpha_evidence_authorized"] is False
    assert policy["high_frequency_trading"] is False


def test_canary_mode_allows_execution_validation_before_alpha_matures(
    tmp_path,
    monkeypatch,
):
    _live_env(monkeypatch)
    _patch_dependencies(monkeypatch)

    controller = object.__new__(ModeController)
    controller.settings = _settings(tmp_path)
    controller._forward_readiness = lambda: {
        "eligible": False,
        "blockers": [
            "INSUFFICIENT_UNBLOCKED_BUY_OUTCOMES",
            "INSUFFICIENT_PROSPECTIVE_TIME_SPAN",
        ],
    }

    payload = controller.preflight("canary")

    assert payload["ready"] is True
    assert payload["canary_scope"] == "EXECUTION_VALIDATION_ONLY"
    assert payload["execution_validation_canary_override_applied"] is True
    assert payload["alpha_evidence_authorized"] is False
    assert "INSUFFICIENT_UNBLOCKED_BUY_OUTCOMES" not in payload["blockers"]
    assert "INSUFFICIENT_PROSPECTIVE_TIME_SPAN" not in payload["blockers"]


def test_disabled_execution_validation_cannot_bypass_forward_gate(
    tmp_path,
    monkeypatch,
):
    _live_env(monkeypatch)
    _patch_dependencies(monkeypatch)

    controller = object.__new__(ModeController)
    controller.settings = _settings(tmp_path, canary_enabled=False)
    controller._forward_readiness = lambda: {
        "eligible": False,
        "blockers": ["INSUFFICIENT_PROSPECTIVE_TIME_SPAN"],
    }

    payload = controller.preflight("canary")

    assert payload["ready"] is False
    assert "INSUFFICIENT_PROSPECTIVE_TIME_SPAN" in payload["blockers"]
    assert payload["execution_validation_canary_override_applied"] is False
