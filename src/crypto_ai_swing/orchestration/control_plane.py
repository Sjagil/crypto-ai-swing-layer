from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.bridge.crypto_operations import NativeOperationsBridge
from crypto_ai_swing.bridge.native_foundation import NativeFoundationBridge
from crypto_ai_swing.execution.bitvavo import live_gate_status
from crypto_ai_swing.execution.crypto_authority import (
    CryptoAuthorityAdapter,
)
from crypto_ai_swing.research.forward import ForwardEvidenceLedger

PUBLIC_MODES = ("shadow", "paper", "canary")
RUNTIME_MODES = {
    "shadow": "shadow",
    "paper": "paper",
    "canary": "live",
}


def mode_runtime_path(project_root: Path, mode: str) -> Path:
    selected = str(mode).lower()
    if selected == "live":
        selected = "canary"
    if selected not in PUBLIC_MODES:
        raise ValueError(f"invalid mode: {mode}")
    return (
        Path(project_root)
        / "output/crypto_ai_swing/modes"
        / selected
    )


class ModeController:
    """Persistent fail-closed mode selection without live authority."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self.state_path = (
            settings.project_root
            / "output/crypto_ai_swing/control/mode_state.json"
        )
        self.audit_path = (
            settings.project_root
            / "output/crypto_ai_swing/control/mode_audit.jsonl"
        )
        self.operations = NativeOperationsBridge(
            settings.crypto_repo_root,
            project_root=settings.project_root,
        )

    def _load(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {
                "schema_version": "crypto_ai_swing_mode_state_v1",
                "selected_mode": "shadow",
                "runtime_mode": "shadow",
                "status": "DEFAULT_FAIL_CLOSED",
                "updated_at": None,
            }
        try:
            payload = json.loads(
                self.state_path.read_text(encoding="utf-8")
            )
            if not isinstance(payload, dict):
                raise ValueError("mode state must be object")
            return payload
        except Exception as exc:
            return {
                "schema_version": "crypto_ai_swing_mode_state_v1",
                "selected_mode": "shadow",
                "runtime_mode": "shadow",
                "status": "CORRUPT_STATE_FAIL_CLOSED",
                "error": (
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                ),
                "updated_at": None,
            }

    def status(self) -> dict[str, Any]:
        payload = self._load()
        selected = str(
            payload.get("selected_mode") or "shadow"
        ).lower()
        if selected not in PUBLIC_MODES:
            selected = "shadow"
        return {
            **payload,
            "selected_mode": selected,
            "runtime_mode": RUNTIME_MODES[selected],
            "mode_root": str(
                mode_runtime_path(
                    self.settings.project_root,
                    selected,
                )
            ),
            "live_authority_granted_by_mode_controller": False,
            "automatic_live_promotion": False,
        }

    def _forward_readiness(self) -> dict[str, Any]:
        cfg = dict(
            (getattr(self.settings, "autonomy", {}) or {}).get(
                "forward_evidence",
                {},
            )
        )
        relative = cfg.get(
            "path",
            "output/crypto_ai_swing/forward/forward.sqlite",
        )
        ledger = ForwardEvidenceLedger(
            self.settings.project_root / relative,
            decision_bucket_minutes=int(
                cfg.get("decision_bucket_minutes", 15)
            ),
        )
        try:
            ready_cfg = dict(
                cfg.get("canary_readiness", {}) or {}
            )
            return ledger.canary_readiness(
                primary_horizon_hours=int(
                    ready_cfg.get(
                        "primary_horizon_hours",
                        4,
                    )
                ),
                minimum_unblocked_buy_outcomes=int(
                    ready_cfg.get(
                        "minimum_unblocked_buy_outcomes",
                        30,
                    )
                ),
                minimum_distinct_markets=int(
                    ready_cfg.get(
                        "minimum_distinct_markets",
                        5,
                    )
                ),
                minimum_observation_span_hours=float(
                    ready_cfg.get(
                        "minimum_observation_span_hours",
                        72,
                    )
                ),
                minimum_mean_return_bps=float(
                    ready_cfg.get(
                        "minimum_mean_return_bps",
                        0.0,
                    )
                ),
                minimum_positive_return_rate=float(
                    ready_cfg.get(
                        "minimum_positive_return_rate",
                        0.50,
                    )
                ),
            )
        finally:
            ledger.close()

    def _execution_validation_mode_policy(self) -> dict[str, Any]:
        proactive = dict(
            getattr(self.settings, "proactive", {}) or {}
        )
        active = dict(proactive.get("active_swing", {}) or {})
        canary = dict(
            active.get("execution_validation_canary", {}) or {}
        )

        blockers: list[str] = []
        maximum_order = float(canary.get("maximum_order_eur", 10.0))

        if not bool(active.get("enabled", True)):
            blockers.append("ACTIVE_SWING_DISABLED")
        if str(active.get("style", "ACTIVE_SWING")).upper() != "ACTIVE_SWING":
            blockers.append("ACTIVE_SWING_STYLE_MISMATCH")
        if bool(active.get("high_frequency_trading", False)):
            blockers.append("HFT_MODE_FORBIDDEN")
        if not bool(canary.get("enabled", False)):
            blockers.append("EXECUTION_VALIDATION_CANARY_DISABLED")
        if not bool(canary.get("manual_authority_required", True)):
            blockers.append("MANUAL_AUTHORITY_MUST_REMAIN_REQUIRED")
        if maximum_order <= 0 or maximum_order > 10.0:
            blockers.append("EXECUTION_VALIDATION_NOTIONAL_CAP_INVALID")
        if bool(canary.get("alpha_evidence_authorized", False)):
            blockers.append("ALPHA_AUTHORITY_FORBIDDEN_FOR_EXECUTION_CANARY")
        if bool(canary.get("autoscale", False)):
            blockers.append("AUTOSCALE_FORBIDDEN_FOR_EXECUTION_CANARY")

        return {
            "schema_version": (
                "active_swing_execution_validation_mode_policy_v1"
            ),
            "eligible": not blockers,
            "blockers": blockers,
            "scope": "EXECUTION_VALIDATION_ONLY",
            "maximum_order_eur": maximum_order,
            "manual_authority_required": bool(
                canary.get("manual_authority_required", True)
            ),
            "prospective_evidence_required_for_scaling": bool(
                canary.get(
                    "prospective_evidence_required_for_scaling",
                    True,
                )
            ),
            "alpha_evidence_authorized": False,
            "autoscale": False,
            "high_frequency_trading": False,
        }

    def preflight(self, target: str) -> dict[str, Any]:
        selected = str(target).lower()
        if selected not in PUBLIC_MODES:
            raise ValueError(
                "target must be shadow, paper, or canary"
            )

        checks: list[dict[str, Any]] = []
        blockers: list[str] = []

        crypto_exists = self.settings.crypto_repo_root.is_dir()
        checks.append(
            {
                "check": "crypto_repo_exists",
                "passed": crypto_exists,
                "value": str(self.settings.crypto_repo_root),
            }
        )
        if not crypto_exists:
            blockers.append("CRYPTO_REPO_MISSING")

        library_ready = False
        foundation_ready = False
        if crypto_exists:
            try:
                library_ready = bool(
                    CryptoLibraryBridge(
                        self.settings.crypto_repo_root
                    )
                    .integration_status()
                    .get("ready")
                )
            except Exception:
                library_ready = False
            try:
                foundation_ready = bool(
                    NativeFoundationBridge(
                        self.settings.crypto_repo_root
                    ).status().ready
                )
            except Exception:
                foundation_ready = False

        checks.append(
            {
                "check": "crypto_library_ready",
                "passed": library_ready,
            }
        )
        if not library_ready:
            blockers.append("CRYPTO_LIBRARY_NOT_READY")

        checks.append(
            {
                "check": "native_foundation_ready",
                "passed": foundation_ready,
            }
        )
        if not foundation_ready:
            blockers.append("NATIVE_FOUNDATION_NOT_READY")

        execution_cfg = dict(
            (getattr(self.settings, "execution", {}) or {}).get(
                "execution",
                {},
            )
        )
        if selected == "paper":
            paper_enabled = bool(
                execution_cfg.get("paper_enabled", True)
            )
            checks.append(
                {
                    "check": "paper_enabled",
                    "passed": paper_enabled,
                }
            )
            if not paper_enabled:
                blockers.append("PAPER_DISABLED")

        readiness: dict[str, Any] | None = None
        authority: dict[str, Any] | None = None
        execution_validation_policy: dict[str, Any] | None = None
        canary_scope: str | None = None
        execution_validation_override = False
        live_gate_payload: dict[str, Any] | None = None

        if selected == "canary":
            try:
                readiness = self._forward_readiness()
            except Exception as exc:
                readiness = {
                    "eligible": False,
                    "blockers": [
                        "FORWARD_READINESS_"
                        + type(exc).__name__.upper()
                    ],
                }

            prospective_ready = bool(
                readiness.get("eligible")
            )
            execution_validation_policy = (
                self._execution_validation_mode_policy()
            )
            execution_validation_ready = bool(
                execution_validation_policy.get("eligible")
            )
            checks.append(
                {
                    "check": "prospective_canary_readiness",
                    "passed": prospective_ready,
                }
            )
            checks.append(
                {
                    "check": "execution_validation_canary_policy",
                    "passed": execution_validation_ready,
                }
            )

            if prospective_ready:
                canary_scope = "PROSPECTIVE_CANARY"
            elif execution_validation_ready:
                canary_scope = "EXECUTION_VALIDATION_ONLY"
                execution_validation_override = True
            else:
                blockers.extend(
                    str(value)
                    for value in readiness.get("blockers") or []
                )
                blockers.extend(
                    str(value)
                    for value in execution_validation_policy.get(
                        "blockers", []
                    )
                )

            try:
                authority = CryptoAuthorityAdapter(
                    self.settings.crypto_repo_root
                ).authority_status()
            except Exception as exc:
                authority = {
                    "active": False,
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{str(exc)[:300]}"
                    ),
                }

            authority_active = bool(
                authority.get("active")
            )
            checks.append(
                {
                    "check": "native_canary_authority_active",
                    "passed": authority_active,
                }
            )
            if not authority_active:
                blockers.append(
                    "NATIVE_CANARY_AUTHORITY_NOT_ACTIVE"
                )

            execution_environment = (
                os.getenv(
                    "CRYPTO_SWING_CANARY_EXECUTE",
                    "",
                )
                == "YES"
            )
            checks.append(
                {
                    "check": "explicit_execution_environment",
                    "passed": execution_environment,
                }
            )
            if not execution_environment:
                blockers.append(
                    "CANARY_EXECUTION_ENV_NOT_READY"
                )

            try:
                gate = live_gate_status(
                    getattr(self.settings, "execution", {}) or {}
                )
                live_gate_payload = {
                    "ready": gate.ready,
                    "blockers": list(gate.blockers),
                }
            except Exception as exc:
                live_gate_payload = {
                    "ready": False,
                    "blockers": [
                        "LIVE_GATE_"
                        + type(exc).__name__.upper()
                    ],
                }

            checks.append(
                {
                    "check": "swing_layer_live_gate",
                    "passed": bool(
                        live_gate_payload.get("ready")
                    ),
                }
            )
            if not bool(live_gate_payload.get("ready")):
                blockers.extend(
                    str(value)
                    for value in live_gate_payload.get(
                        "blockers", []
                    )
                )

        blockers = list(dict.fromkeys(blockers))
        return {
            "schema_version": "crypto_ai_swing_mode_preflight_v1",
            "target": selected,
            "runtime_mode": RUNTIME_MODES[selected],
            "ready": not blockers,
            "checks": checks,
            "blockers": blockers,
            "prospective_readiness": readiness,
            "execution_validation_policy": execution_validation_policy,
            "canary_scope": canary_scope,
            "execution_validation_canary_override_applied": (
                execution_validation_override
            ),
            "alpha_evidence_authorized": bool(
                canary_scope == "PROSPECTIVE_CANARY"
                and readiness
                and readiness.get("eligible")
            ),
            "live_gate": live_gate_payload,
            "native_authority": authority,
            "live_authority_granted_by_mode_controller": False,
            "automatic_live_promotion": False,
            "orders_submitted": 0,
        }

    def switch(self, target: str) -> dict[str, Any]:
        selected = str(target).lower()
        preflight = self.preflight(selected)
        if not bool(preflight.get("ready")):
            return {
                "status": "BLOCKED",
                "changed": False,
                "target": selected,
                "preflight": preflight,
                "orders_submitted": 0,
            }

        previous = self.status()
        payload = {
            "schema_version": "crypto_ai_swing_mode_state_v1",
            "selected_mode": selected,
            "runtime_mode": RUNTIME_MODES[selected],
            "status": "SELECTED",
            "previous_mode": previous.get(
                "selected_mode"
            ),
            "updated_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "mode_root": str(
                mode_runtime_path(
                    self.settings.project_root,
                    selected,
                )
            ),
            "automatic_live_promotion": False,
        }
        self.operations.atomic_write_json(
            self.state_path,
            payload,
        )

        common = CryptoLibraryBridge(
            self.settings.crypto_repo_root
        ).import_module("utils.common")
        common.append_jsonl(
            self.audit_path,
            {
                "event": "MODE_SWITCH",
                "at": payload["updated_at"],
                "from": previous.get("selected_mode"),
                "to": selected,
                "preflight_ready": True,
                "live_authority_granted_by_mode_controller": False,
            },
        )
        mode_runtime_path(
            self.settings.project_root,
            selected,
        ).mkdir(parents=True, exist_ok=True)

        return {
            "status": "SWITCHED",
            "changed": (
                previous.get("selected_mode")
                != selected
            ),
            "state": payload,
            "preflight": preflight,
            "orders_submitted": 0,
        }
