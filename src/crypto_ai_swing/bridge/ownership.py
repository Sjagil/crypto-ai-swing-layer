from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CANONICAL_REPOSITORY = "Sjagil/crypto"
APPLICATION_REPOSITORY = "Sjagil/crypto-ai-swing-layer"
MANDATORY_CRYPTO_DOMAINS = frozenset({
    "market_data_transport", "canonical_candles", "provider_normalization",
    "account_state", "reconciliation", "exchange_transport", "trading_math",
    "kelly_math", "transaction_cost_model", "portfolio_risk", "correlation_risk",
    "position_sizing", "shariah_market_eligibility", "live_execution",
    "protective_exits", "kill_switch",
})
MANDATORY_SWING_DOMAINS = frozenset({
    "feature_engineering", "model_training", "model_calibration", "model_registry",
    "agent_development", "agent_orchestration", "strategy_research", "quant_research",
    "prospective_validation", "market_intelligence_fusion", "decision_generation",
    "performance_attribution", "retraining", "champion_challenger_management",
})
LEGACY_EXCHANGE_MARKERS = (
    "crypto_ai_swing.execution.bitvavo", "class BitvavoREST", "api.bitvavo.com",
)
CANONICAL_DEFINITION_MARKERS = (
    "def kelly_fraction(", "def fractional_kelly(", "def calculate_position_size(",
    "def calculate_position_size_from_stop_fraction(", "class RiskManager(",
    "class PortfolioSnapshot(", "class PositionExposure(", "class KillSwitch(",
    "class BitvavoSpotClient(",
)

@dataclass(frozen=True)
class OwnershipContract:
    canonical_repository: str
    application_repository: str
    fail_closed: bool
    crypto_owned: frozenset[str]
    swing_owned: frozenset[str]
    legacy_migration_exceptions: frozenset[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_repository": self.canonical_repository,
            "application_repository": self.application_repository,
            "fail_closed": self.fail_closed,
            "crypto_owned": sorted(self.crypto_owned),
            "swing_owned": sorted(self.swing_owned),
            "legacy_migration_exceptions": sorted(self.legacy_migration_exceptions),
        }


def load_ownership_contract(project_root: Path) -> OwnershipContract:
    root = Path(project_root).expanduser().resolve()
    path = root / "config/canonical_ownership.yaml"
    if not path.is_file():
        raise RuntimeError("ROUND39_OWNERSHIP_CONTRACT_MISSING")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("ROUND39_OWNERSHIP_CONTRACT_INVALID")
    canonical = str(payload.get("canonical_repository") or "")
    application = str(payload.get("application_repository") or "")
    fail_closed = bool(payload.get("fail_closed", False))
    crypto_owned = frozenset(str(v) for v in payload.get("crypto_owned") or ())
    swing_owned = frozenset(str(v) for v in payload.get("swing_owned") or ())
    exceptions = frozenset(str(v) for v in payload.get("legacy_migration_exceptions") or ())
    violations: list[str] = []
    if canonical != CANONICAL_REPOSITORY:
        violations.append("CANONICAL_REPOSITORY_MISMATCH")
    if application != APPLICATION_REPOSITORY:
        violations.append("APPLICATION_REPOSITORY_MISMATCH")
    if not fail_closed:
        violations.append("FAIL_CLOSED_REQUIRED")
    missing = MANDATORY_CRYPTO_DOMAINS - crypto_owned
    if missing:
        violations.append("MISSING_CRYPTO_DOMAINS:" + ",".join(sorted(missing)))
    missing = MANDATORY_SWING_DOMAINS - swing_owned
    if missing:
        violations.append("MISSING_SWING_DOMAINS:" + ",".join(sorted(missing)))
    overlap = crypto_owned & swing_owned
    if overlap:
        violations.append("OWNERSHIP_OVERLAP:" + ",".join(sorted(overlap)))
    if exceptions:
        violations.append(
            "ROUND40_LEGACY_MIGRATION_EXCEPTIONS_FORBIDDEN:"
            + ",".join(sorted(exceptions))
        )
    for relative in sorted(exceptions):
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            violations.append(f"INVALID_EXCEPTION_PATH:{relative}")
        elif not (root / candidate).is_file():
            violations.append(f"MISSING_EXCEPTION_PATH:{relative}")
    if violations:
        raise RuntimeError("ROUND39_OWNERSHIP_INVALID:" + "|".join(violations))
    return OwnershipContract(canonical, application, fail_closed, crypto_owned, swing_owned, exceptions)


def audit_source_ownership(project_root: Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    contract = load_ownership_contract(root)
    violations: list[dict[str, str]] = []
    legacy_hits: list[dict[str, str]] = []
    for path in sorted((root / "src/crypto_ai_swing").rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if relative == "src/crypto_ai_swing/bridge/ownership.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        exception = relative in contract.legacy_migration_exceptions
        for marker in LEGACY_EXCHANGE_MARKERS:
            if marker in text:
                row = {"path": relative, "marker": marker}
                if exception:
                    legacy_hits.append(row)
                else:
                    violations.append({**row, "reason": "NEW_LOCAL_EXCHANGE_DEPENDENCY"})
        for marker in CANONICAL_DEFINITION_MARKERS:
            if marker in text:
                row = {"path": relative, "marker": marker}
                if exception:
                    legacy_hits.append(row)
                else:
                    violations.append({**row, "reason": "DUPLICATED_CANONICAL_IMPLEMENTATION"})
    return {
        "schema_version": "round40_canonical_ownership_audit_v1",
        "ready": not violations,
        "canonical_repository": contract.canonical_repository,
        "application_repository": contract.application_repository,
        "violations": violations,
        "legacy_migration_hits": legacy_hits,
        "legacy_migration_exceptions": sorted(contract.legacy_migration_exceptions),
        "round40_required": bool(legacy_hits),
    }
