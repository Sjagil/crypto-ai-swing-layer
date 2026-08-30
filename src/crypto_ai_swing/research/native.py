from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import inspect
import json

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge


NATIVE_RESEARCH_MODULES = (
    "research.research_factory",
    "research.backtest",
    "research.optimization",
    "research.stochastic_validation",
    "research.classical_strategy_factory",
    "research.autonomous_rd",
    "research.combinatorial_lab",
    "research.alpha_discovery",
    "research.features",
)


@dataclass(frozen=True)
class NativeResearchStatus:
    ready: bool
    imported_modules: int
    required_modules: int
    modules: tuple[dict[str, Any], ...]
    promotion_states: tuple[str, ...]


class NativeResearchBridge:
    """Read-only/control adapter for the canonical Sjagil/crypto research stack."""

    def __init__(self, crypto_repo_root: Path):
        self.crypto = CryptoLibraryBridge(crypto_repo_root)

    @staticmethod
    def _interfaces(module: Any) -> dict[str, str]:
        result: dict[str, str] = {}
        for name, value in vars(module).items():
            if name.startswith("_"):
                continue
            if inspect.isfunction(value) or inspect.isclass(value):
                try:
                    result[name] = str(inspect.signature(value))
                except Exception:
                    result[name] = "<signature unavailable>"
        return result

    def status(self) -> NativeResearchStatus:
        rows = []
        for name in NATIVE_RESEARCH_MODULES:
            try:
                module = self.crypto.import_module(name)
                rows.append(
                    {
                        "module": name,
                        "imported": True,
                        "file": str(getattr(module, "__file__", "") or ""),
                        "interfaces": self._interfaces(module),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "module": name,
                        "imported": False,
                        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                        "interfaces": {},
                    }
                )
        states: tuple[str, ...] = ()
        try:
            factory = self.crypto.import_module("research.research_factory")
            enum = getattr(factory, "PromotionState")
            states = tuple(str(item.value) for item in enum)
        except Exception:
            pass
        return NativeResearchStatus(
            ready=all(row["imported"] for row in rows),
            imported_modules=sum(bool(row["imported"]) for row in rows),
            required_modules=len(rows),
            modules=tuple(rows),
            promotion_states=states,
        )

    def classical_factory_plan(self, trial_count: int = 2000) -> dict[str, Any]:
        module = self.crypto.import_module("research.classical_strategy_factory")
        fn = getattr(module, "classical_factory_plan", None)
        if not callable(fn):
            raise RuntimeError("classical_factory_plan unavailable")
        value = fn(trial_count=int(trial_count))
        if not isinstance(value, dict):
            raise RuntimeError("classical_factory_plan returned non-dict")
        return value

    def generated_dna(self, trial_count: int = 2000) -> list[dict[str, Any]]:
        module = self.crypto.import_module("research.classical_strategy_factory")
        fn = getattr(module, "generate_classical_strategy_dna", None)
        if not callable(fn):
            raise RuntimeError("generate_classical_strategy_dna unavailable")
        rows = fn(trial_count=int(trial_count))
        result = []
        for row in rows:
            to_dict = getattr(row, "to_dict", None)
            result.append(to_dict() if callable(to_dict) else {"repr": repr(row)})
        return result


    def run_factory_campaign(
        self,
        *,
        maximum_rows: int = 20_000,
        execute_exact: bool = False,
    ) -> dict[str, Any]:
        """Run the canonical bounded research factory in Sjagil/crypto.

        This calls the native factory and never grants paper/live authority.
        Exact validation is opt-in because it can be materially more expensive.
        """
        module = self.crypto.import_module("research.research_factory")
        fn = getattr(module, "build_research_factory_artifact", None)
        if not callable(fn):
            raise RuntimeError("build_research_factory_artifact unavailable")
        settings = self.crypto.settings()
        value = fn(
            settings,
            maximum_rows=int(maximum_rows),
            execute_exact=bool(execute_exact),
        )
        if not isinstance(value, dict):
            raise RuntimeError("build_research_factory_artifact returned non-dict")
        return value

    @staticmethod
    def factory_summary(payload: dict[str, Any]) -> dict[str, Any]:
        stage0 = dict(payload.get("stage0") or {})
        exact = dict(payload.get("exact_validation") or {})
        promotion = list(payload.get("promotion_table") or [])
        first_promotion = dict(promotion[0]) if promotion else {}
        return {
            "schema_version": payload.get("schema_version"),
            "run_id": payload.get("run_id"),
            "created_at": payload.get("created_at"),
            "stage0_tested_variants": stage0.get("tested_variant_count"),
            "stage0_survivors": len(stage0.get("survivors") or []),
            "stage0_status_counts": stage0.get("status_counts") or {},
            "exact_status": exact.get("status"),
            "promotion_state": first_promotion.get("promotion_state"),
            "forward_candidate_count": len(payload.get("forward_candidates") or []),
            "missing_dataset_sources": payload.get("missing_dataset_sources") or [],
            "automatic_live_promotion": False,
            "orders_submitted": 0,
        }

    def write_status(self, path: Path) -> dict[str, Any]:
        status = self.status()
        payload = {
            "ready": status.ready,
            "imported_modules": status.imported_modules,
            "required_modules": status.required_modules,
            "promotion_states": list(status.promotion_states),
            "modules": list(status.modules),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload
