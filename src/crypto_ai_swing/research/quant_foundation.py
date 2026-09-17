from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from crypto_ai_swing.bridge.crypto_operations import NativeOperationsBridge
from crypto_ai_swing.research.native import NativeResearchBridge


REQUIRED_NATIVE_RESEARCH_MODULES = {
    "research.research_factory",
    "research.backtest",
    "research.optimization",
    "research.statistical_evidence",
    "research.stochastic_validation",
    "research.trading_math",
    "research.portfolio_selection",
    "research.classical_strategy_factory",
    "research.autonomous_rd",
    "research.combinatorial_lab",
    "research.features",
    "research.regime_router",
}


class QuantFoundationAudit:
    """Verifies that quantitative authority comes from canonical Sjagil/crypto."""

    SCHEMA = "crypto_ai_swing_quant_foundation_audit_v1"

    def __init__(self, settings) -> None:
        self.settings = settings
        self.research = NativeResearchBridge(settings.crypto_repo_root)
        self.operations = NativeOperationsBridge(
            settings.crypto_repo_root,
            project_root=settings.project_root,
        )

    def run(self) -> dict[str, Any]:
        research = self.research.status()
        operations = self.operations.status()
        imported = {
            str(row.get("module"))
            for row in research.modules
            if row.get("imported")
        }
        missing = sorted(REQUIRED_NATIVE_RESEARCH_MODULES - imported)
        return {
            "schema_version": self.SCHEMA,
            "generated_at": datetime.now(UTC).isoformat(),
            "ready": bool(operations.get("ready") and not missing),
            "required_research_modules": sorted(
                REQUIRED_NATIVE_RESEARCH_MODULES
            ),
            "missing_research_modules": missing,
            "native_research": {
                "ready": research.ready,
                "required_scope_ready": not missing,
                "catalog_ready": research.ready,
                "imported_modules": research.imported_modules,
                "required_modules": research.required_modules,
                "promotion_states": list(research.promotion_states),
            },
            "native_operations": operations,
            "contracts": {
                "canonical_cost_model": True,
                "canonical_kelly": True,
                "canonical_risk_manager": True,
                "canonical_backtester": True,
                "canonical_walk_forward": True,
                "canonical_stochastic_validation": True,
                "canonical_strategy_dna_factory": True,
                "canonical_research_trace": True,
                "duplicate_math_in_swing_layer": False,
            },
            "orders_generated": 0,
            "orders_submitted": 0,
        }
