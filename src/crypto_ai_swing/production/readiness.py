from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from crypto_ai_swing.execution.active_swing_canary import (
    canonical_preflight_explicitly_denied,
)
from crypto_ai_swing.execution.crypto_authority import CryptoAuthorityAdapter
from crypto_ai_swing.production.audit import ProductionAuditLog, redact
from crypto_ai_swing.production.recovery import ExecutionRecoveryJournal

BAD_STATUS_PREFIXES = (
    "BLOCK", "DENIED", "REJECT", "FAILED", "ERROR", "UNHEALTHY"
)

def _production_config(settings) -> dict[str, Any]:
    path = Path(settings.project_root) / "config/production.yaml"
    if not path.is_file():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}

def _explicit_failures(payload: Mapping[str, Any] | None) -> list[str]:
    row = dict(payload or {})
    failures: list[str] = []
    for key in ("blockers", "failures", "reason_codes", "reasons"):
        values = row.get(key) or []
        if isinstance(values, str):
            values = [values]
        failures.extend(str(item) for item in values if str(item))
    for key in ("ready", "healthy", "accepted", "approved", "eligible", "allowed"):
        if key in row and row.get(key) is False:
            failures.append(f"{key.upper()}_FALSE")
    status = str(row.get("status") or "").upper()
    if status.startswith(BAD_STATUS_PREFIXES):
        failures.append(f"STATUS_{status}")
    return sorted(set(failures))

def _canary_active(payload: Mapping[str, Any] | None) -> tuple[bool, str]:
    row = dict(payload or {})
    for key in ("canary_active", "active", "authority_active", "approved", "armed"):
        if key in row:
            return bool(row.get(key)), key
    nested = row.get("canary")
    if isinstance(nested, Mapping):
        return _canary_active(nested)
    return False, "UNKNOWN"

class ProductionReadinessEngine:
    SCHEMA = "crypto_ai_swing_production_readiness_v1"

    def __init__(self, settings, *, authority=None) -> None:
        self.settings = settings
        self.cfg = _production_config(settings)
        self.authority = authority or CryptoAuthorityAdapter(settings.crypto_repo_root)
        self.root = Path(settings.project_root) / "output/crypto_ai_swing/production"
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.root / "readiness.json"
        self.audit = ProductionAuditLog(self.root / "audit.jsonl")
        self.journal = ExecutionRecoveryJournal(
            self.root / "execution_recovery.sqlite"
        )

    def _static(self) -> tuple[dict[str, bool], list[str]]:
        execution = dict(self.settings.execution or {})
        risk = dict(self.settings.risk or {})
        authority = dict(execution.get("authority", {}) or {})
        canary = dict(execution.get("swing_canary", {}) or {})
        market = dict(risk.get("market_policy", {}) or {})
        safe = dict(self.cfg.get("safe_canary_caps", {}) or {})

        checks = {
            "spot_only": bool(market.get("spot_only", False)),
            "long_only": bool(market.get("long_only", False)),
            "shorting_disabled": not bool(market.get("shorting_allowed", True)),
            "leverage_disabled": not bool(market.get("leverage_allowed", True)),
            "derivatives_disabled": not bool(market.get("derivatives_allowed", True)),
            "withdrawals_disabled": not bool(authority.get("withdrawals_enabled", True)),
            "automatic_live_promotion_disabled": not bool(
                authority.get("automatic_live_promotion", True)
            ),
            "strategy_direct_exchange_calls_disabled": not bool(
                authority.get("strategy_direct_exchange_calls", True)
            ),
            "native_crypto_authority": bool(canary.get("native_crypto_authority", False)),
            "native_exchange_stop_required": bool(
                canary.get("native_exchange_stop_required", False)
            ),
            "autoscale_disabled": not bool(canary.get("autoscale", True)),
            "automatic_authority_disabled": not bool(
                canary.get("automatic_authority", True)
            ),
            "order_cap_safe": float(canary.get("maximum_order_eur", 1e18))
            <= float(safe.get("maximum_order_eur", 10.0)),
            "exposure_cap_safe": float(
                canary.get("maximum_total_exposure_eur", 1e18)
            ) <= float(safe.get("maximum_total_exposure_eur", 10.0)),
            "position_cap_safe": int(canary.get("maximum_positions", 999))
            <= int(safe.get("maximum_positions", 1)),
            "daily_order_cap_safe": int(
                canary.get("maximum_new_orders_per_day", 999)
            ) <= int(safe.get("maximum_new_orders_per_day", 1)),
            "risk_cap_safe": float(
                canary.get("maximum_risk_per_trade_eur", 1e18)
            ) <= float(safe.get("maximum_risk_per_trade_eur", 1.0)),
            "slippage_cap_safe": float(
                canary.get("maximum_slippage_bps", 1e18)
            ) <= float(safe.get("maximum_slippage_bps", 25.0)),
        }
        blockers = [
            f"STATIC_{name.upper()}_FAILED"
            for name, passed in checks.items()
            if not passed
        ]
        return checks, blockers

    def _canonical_runtime(
        self, markets: list[str]
    ) -> tuple[dict[str, Any], list[str]]:
        result: dict[str, Any] = {}
        blockers: list[str] = []
        calls = {
            "authority_status": self.authority.authority_status,
            "reconciliation": lambda: self.authority.reconcile(markets),
            "account_snapshot": lambda: self.authority.account_snapshot(markets),
            "portfolio": self.authority.portfolio,
        }
        for name, fn in calls.items():
            try:
                raw = dict(fn() or {})
                result[name] = redact(raw)
                failures = _explicit_failures(raw)
                blockers.extend(
                    f"CANONICAL_{name.upper()}_{reason}"
                    for reason in failures
                )
            except Exception as exc:
                result[name] = {
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}:{str(exc)[:300]}",
                }
                blockers.append(f"CANONICAL_{name.upper()}_UNAVAILABLE")

        account = dict(result.get("account_snapshot") or {})
        if str(account.get("status") or "").upper() != "READY":
            blockers.append("CANONICAL_ACCOUNT_NOT_READY")

        reconcile = dict(result.get("reconciliation") or {})
        if _explicit_failures(reconcile):
            blockers.append("CANONICAL_RECONCILIATION_NOT_READY")

        authority_status = dict(result.get("authority_status") or {})
        active, source = _canary_active(authority_status)
        result["canary_authority"] = {"active": active, "source": source}
        require_active = bool(
            (self.cfg.get("live_guard", {}) or {}).get(
                "require_canary_authority_active", True
            )
        )
        if require_active and not active:
            blockers.append("CANARY_AUTHORITY_NOT_ACTIVE")

        if authority_status.get("execution_environment_ready") is False:
            blockers.append("CANONICAL_EXECUTION_ENVIRONMENT_NOT_READY")
        state_status = str(
            authority_status.get("state_status") or ""
        ).strip().upper()
        if state_status and state_status not in {"READY", "ACTIVE", "HEALTHY"}:
            blockers.append(
                f"CANONICAL_AUTHORITY_STATE_{state_status}"
            )

        account = dict(result.get("account_snapshot") or {})
        if account.get("entry_allowed") is False:
            blockers.append("CANONICAL_ACCOUNT_ENTRY_NOT_ALLOWED")

        return result, sorted(set(blockers))

    def assess(
        self, markets: list[str], *, network: bool = True, persist: bool = True
    ) -> dict[str, Any]:
        markets = [str(value).upper() for value in markets if str(value)]
        static_checks, blockers = self._static()
        runtime: dict[str, Any] = {"status": "NOT_CHECKED"}
        if network:
            runtime, runtime_blockers = self._canonical_runtime(markets)
            blockers.extend(runtime_blockers)

        recovery = self.journal.status()
        if recovery["unresolved_count"] > 0:
            blockers.append("UNRESOLVED_EXECUTION_OPERATION")

        require_network = bool(
            (self.cfg.get("live_guard", {}) or {}).get(
                "require_remote_runtime_checks", True
            )
        )
        if require_network and not network:
            blockers.append("REMOTE_RUNTIME_NOT_CHECKED")
        network_ok = network or not require_network
        static_ok = all(static_checks.values())
        live_ready = (
            static_ok
            and network_ok
            and not blockers
            and recovery["unresolved_count"] == 0
        )

        exit_blockers = [
            value
            for value in blockers
            if value not in {
                "CANARY_AUTHORITY_NOT_ACTIVE",
                "UNRESOLVED_EXECUTION_OPERATION",
                "CANONICAL_ACCOUNT_ENTRY_NOT_ALLOWED",
            }
        ]
        account_runtime = dict(
            (runtime.get("account_snapshot") or {})
            if isinstance(runtime, dict)
            else {}
        )
        if account_runtime.get("risk_reduction_allowed") is False:
            exit_blockers.append("CANONICAL_RISK_REDUCTION_NOT_ALLOWED")
        exit_blockers = sorted(set(exit_blockers))
        exit_ready = static_ok and network_ok and not exit_blockers

        payload = {
            "schema_version": self.SCHEMA,
            "status": "READY" if live_ready else "BLOCKED",
            "markets": markets,
            "static_checks": static_checks,
            "canonical_runtime": runtime,
            "recovery": recovery,
            "live_canary_ready": live_ready,
            "exit_ready": exit_ready,
            "blockers": sorted(set(blockers)),
            "exit_blockers": sorted(set(exit_blockers)),
            "strategy_scaling_authorized": False,
            "execution_validation_canary_only": True,
            "automatic_live_promotion": False,
            "autoscale_authorized": False,
            "orders_submitted": 0,
            "secrets_serialized": False,
        }
        if persist:
            self.latest_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            self.audit.append("READINESS_ASSESSMENT", payload)
        return payload

    def preflight_intent(self, intent) -> dict[str, Any]:
        try:
            raw = dict(self.authority.preflight(intent) or {})
        except Exception as exc:
            return {
                "status": "BLOCKED",
                "allowed": False,
                "blockers": [
                    f"CANONICAL_PREFLIGHT_UNAVAILABLE:{type(exc).__name__}"
                ],
            }
        denied, reasons = canonical_preflight_explicitly_denied(raw)
        return {
            "status": "READY" if not denied else "BLOCKED",
            "allowed": not denied,
            "blockers": reasons,
            "canonical_preflight": redact(raw),
        }
