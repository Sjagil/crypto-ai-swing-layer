from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.bridge.crypto_operations import NativeOperationsBridge
from crypto_ai_swing.research.bootstrap import ColdStartResearchRunner
from crypto_ai_swing.research.native import NativeResearchBridge
from crypto_ai_swing.research.native_tournament import (
    run_native_alpha_tournament,
)
from crypto_ai_swing.research.performance_attribution import (
    PerformanceAttributionEngine,
)
from crypto_ai_swing.research.promotion import ResearchPromotionRegistry
from crypto_ai_swing.research.strategy_challenger import (
    StrategyChallengerLab,
)


def _now() -> datetime:
    return datetime.now(UTC)


class AutonomousStrategyDirector:
    """Continuously builds, tests and retires strategy challengers.

    Canonical Sjagil/crypto remains the owner of research mathematics, exact
    backtesting, costs, stochastic validation, strategy DNA and promotion
    states. This director schedules those capabilities and converts observed
    weaknesses into preregistered experiments.
    """

    SCHEMA = "crypto_ai_swing_autonomous_strategy_director_v1"

    def __init__(self, settings, *, mode: str = "shadow") -> None:
        self.settings = settings
        self.mode = str(mode).lower()
        self.cfg = dict(
            (getattr(settings, "autonomy", {}) or {}).get(
                "strategy_director", {}
            )
            or {}
        )
        self.crypto = CryptoLibraryBridge(settings.crypto_repo_root)
        self.native = NativeResearchBridge(settings.crypto_repo_root)
        self.bootstrap_research = ColdStartResearchRunner(settings)
        self.operations = NativeOperationsBridge(
            settings.crypto_repo_root,
            project_root=settings.project_root,
        )
        research_mode = "shadow" if self.mode == "live" else self.mode
        self.attribution = PerformanceAttributionEngine(
            settings, mode=research_mode
        )
        self.strategy_lab = StrategyChallengerLab(
            settings, mode=research_mode
        )
        self.registry = ResearchPromotionRegistry(settings)
        self.root = (
            Path(settings.project_root)
            / "output/crypto_ai_swing/research/strategy_director"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.latest_path = self.root / "latest.json"
        self.experiments_path = self.root / "experiments.jsonl"
        self.state = self._load(self.state_path)

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return dict(value) if isinstance(value, dict) else {}
        except (OSError, TypeError, ValueError):
            return {}

    @staticmethod
    def _parse(raw: Any) -> datetime | None:
        if not raw:
            return None
        try:
            value = datetime.fromisoformat(str(raw))
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        except (TypeError, ValueError):
            return None

    def _due(self, key: str, seconds: int, force: bool) -> bool:
        if force:
            return True
        previous = self._parse(self.state.get(key))
        return previous is None or (
            _now() - previous.astimezone(UTC)
        ).total_seconds() >= max(60, seconds)

    def _record_hypotheses(
        self,
        priorities: list[Mapping[str, Any]],
        *,
        markets: list[str],
    ) -> list[dict[str, Any]]:
        module = self.crypto.import_module("research.autonomous_rd")
        external_path = (
            Path(self.settings.project_root)
            / "output/crypto_ai_swing/intelligence/external_research/latest.json"
        )
        external = self._load(external_path)
        external_records = list(
            external.get("evidence_records") or []
        )
        external_dataset_ids = tuple(
            f"EXTERNAL_EVIDENCE:{row.get('raw_hash')}"
            for row in external_records[:10]
            if row.get("raw_hash")
        )
        output: list[dict[str, Any]] = []
        commit = "UNKNOWN"
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.settings.project_root,
                text=True,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
        costs = self.operations.canonical_cost_inputs()

        for priority in priorities[
            : int(self.cfg.get("maximum_new_hypotheses_per_cycle", 6))
        ]:
            code = str(priority.get("code") or "IMPROVE_EDGE")
            created_at = _now()
            hypothesis = module.ResearchHypothesis.create(
                statement=(
                    f"Improve robust prospective net performance for {code} "
                    "without weakening canonical risk, cost or causality gates."
                ),
                rationale=str(priority.get("reason") or code),
                falsification_criteria=(
                    "normal_net_after_costs_not_positive",
                    "stressed_net_after_costs_not_positive",
                    "no_oos_improvement_over_current_champion",
                    "stochastic_robustness_gate_failed",
                    "point_in_time_or_leakage_gate_failed",
                ),
                evidence_inputs=(
                    "canonical_ohlcv",
                    "canonical_cost_model",
                    "forward_evidence",
                    "market_regime",
                    "execution_attribution",
                    "structured_external_browser_evidence",
                ),
                created_at=created_at,
            )
            experiment = module.PreregisteredExperiment.create(
                hypothesis_id=hypothesis.hypothesis_id,
                code_commit=commit,
                dataset_ids=(
                    tuple(
                        f"FORWARD_EVIDENCE:{market}"
                        for market in markets[:25]
                    )
                    + external_dataset_ids
                )
                or ("FORWARD_EVIDENCE:UNIVERSE",),
                metrics=(
                    "net_return_after_costs",
                    "profit_factor",
                    "sortino",
                    "maximum_drawdown",
                    "stochastic_validation",
                    "benchmark_relative_return",
                ),
                acceptance_thresholds={
                    "normal_net_positive": 1,
                    "stressed_net_positive": 1,
                    "stochastic_validation_pass": 1,
                    "must_improve_champion": 1,
                },
                leakage_controls=(
                    "closed_candles_only",
                    "next_open_execution",
                    "purged_walk_forward",
                    "prospective_freeze",
                    "point_in_time_universe",
                ),
                cost_model_version=str(
                    costs.get("cost_model_version") or "UNKNOWN"
                ),
                preregistered_at=created_at,
            )
            trace = module.ResearchTrace.empty(
                namespace="round42_strategy_director"
            )
            trace = trace.append(hypothesis)
            trace = trace.append(experiment)
            row = {
                "generated_at": created_at.isoformat(),
                "priority": dict(priority),
                "hypothesis": hypothesis.model_dump(mode="json"),
                "experiment": experiment.model_dump(mode="json"),
                "trace": trace.model_dump(mode="json"),
                "automatic_live_promotion": False,
                "orders_submitted": 0,
            }
            output.append(row)
            with self.experiments_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(row, sort_keys=True, default=str) + "\n"
                )
        return output

    def cycle(
        self,
        *,
        markets: list[str],
        priorities: list[Mapping[str, Any]] | None = None,
        force: bool = False,
        force_exact: bool = False,
    ) -> dict[str, Any]:
        if not bool(self.cfg.get("enabled", True)):
            return {
                "schema_version": self.SCHEMA,
                "status": "DISABLED",
                "orders_generated": 0,
                "orders_submitted": 0,
            }

        tasks: dict[str, Any] = {}
        errors: list[dict[str, str]] = []
        now = _now().isoformat()

        if self._due(
            "last_attribution_at",
            int(self.cfg.get("attribution_seconds", 900)),
            force,
        ):
            try:
                tasks["attribution"] = self.attribution.refresh()
                self.state["last_attribution_at"] = now
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "task": "attribution",
                        "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                    }
                )

        if self._due(
            "last_strategy_lab_at",
            int(self.cfg.get("strategy_lab_seconds", 3600)),
            force,
        ):
            try:
                tasks["strategy_lab"] = self.strategy_lab.refresh()
                self.state["last_strategy_lab_at"] = now
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "task": "strategy_lab",
                        "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                    }
                )

        if self._due(
            "last_dna_factory_at",
            int(self.cfg.get("dna_factory_seconds", 21600)),
            force,
        ):
            try:
                plan = self.native.classical_factory_plan(
                    trial_count=int(
                        self.cfg.get("strategy_dna_trials", 2000)
                    )
                )
                tasks["canonical_strategy_dna_factory"] = {
                    "status": plan.get("status"),
                    "trial_count": plan.get("trial_count"),
                    "economic_family_count": plan.get(
                        "economic_family_count"
                    ),
                    "research_universe": plan.get("research_universe"),
                    "timeframe_routes": plan.get("timeframe_routes"),
                    "execution_policies": plan.get("execution_policies"),
                    "evaluation_contract": plan.get("evaluation_contract"),
                    "search_space_hash": plan.get("search_space_hash"),
                    "orders_generated": 0,
                    "orders_submitted": 0,
                }
                self.state["last_dna_factory_at"] = now
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "task": "canonical_strategy_dna_factory",
                        "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                    }
                )

        if self._due(
            "last_native_tournament_at",
            int(self.cfg.get("native_tournament_seconds", 21600)),
            force,
        ):
            try:
                tasks["native_alpha_tournament"] = (
                    run_native_alpha_tournament(
                        crypto_repo_root=self.settings.crypto_repo_root
                    )
                )
                self.state["last_native_tournament_at"] = now
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "task": "native_alpha_tournament",
                        "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                    }
                )

        factory_due = self._due(
            "last_factory_stage0_at",
            int(self.cfg.get("factory_stage0_seconds", 21600)),
            force,
        )
        exact_due = force_exact or self._due(
            "last_exact_research_at",
            int(self.cfg.get("exact_research_seconds", 86400)),
            False,
        )
        if factory_due or exact_due:
            try:
                raw = self.native.run_factory_campaign(
                    maximum_rows=int(
                        self.cfg.get("maximum_research_rows", 20000)
                    ),
                    execute_exact=bool(exact_due),
                )
                tasks["canonical_research_factory"] = {
                    **self.native.factory_summary(raw),
                    "exact_requested": bool(exact_due),
                }
                if str(raw.get("status") or "").startswith("COLD_START"):
                    tasks["cold_start_research"] = self.bootstrap_research.run(
                        markets=markets,
                        timeframe=str(
                            self.cfg.get(
                                "cold_start_timeframe",
                                "1h",
                            )
                        ),
                        final_holdout_shortlist_size=int(
                            self.cfg.get(
                                "cold_start_final_holdout_shortlist_size",
                                5,
                            )
                        ),
                    )
                    tasks["cold_start_research"][
                        "automatic_live_promotion"
                    ] = False
                    tasks["cold_start_research"][
                        "orders_submitted"
                    ] = 0
                self.state["last_factory_stage0_at"] = now
                if exact_due:
                    self.state["last_exact_research_at"] = now
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "task": "canonical_research_factory",
                        "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                    }
                )

        if priorities:
            try:
                tasks["adaptive_preregistration"] = (
                    self._record_hypotheses(
                        list(priorities),
                        markets=markets,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "task": "adaptive_preregistration",
                        "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                    }
                )

        try:
            tasks["promotion_registry"] = self.registry.refresh()
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "task": "promotion_registry",
                    "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                }
            )

        self.state["research_generation"] = int(
            self.state.get("research_generation", 0)
        ) + 1
        self.state["last_cycle_at"] = now
        self.operations.atomic_write_json(self.state_path, self.state)

        payload = {
            "schema_version": self.SCHEMA,
            "generated_at": now,
            "status": "DEGRADED" if errors else "HEALTHY",
            "markets": markets,
            "tasks": tasks,
            "errors": errors,
            "strategy_champion": self.strategy_lab.champion(),
            "research_generation": self.state["research_generation"],
            "continuous_candidate_generation": True,
            "canonical_math_and_backtest_authority": True,
            "strategies_are_available_to_agents": True,
            "promotion_policy": {
                "new_strategy_must_beat_current_champion_oos": True,
                "normal_and_stressed_costs": True,
                "walk_forward_and_prospective_evidence": True,
                "multiple_testing_controls": True,
                "stochastic_robustness_required": True,
                "automatic_live_promotion": False,
            },
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        self.operations.atomic_write_json(self.latest_path, payload)
        return payload

    def strategy_status(self) -> dict[str, Any]:
        champion = self.strategy_lab.champion()
        return {
            "status": "RESEARCH_CHAMPION" if champion else "NOT_BUILT",
            "champion": champion,
            "live_decision_influence": False,
            "automatic_live_promotion": False,
        }

    def status(self) -> dict[str, Any]:
        return self._load(self.latest_path) or {
            "schema_version": self.SCHEMA,
            "status": "NOT_BUILT",
            "strategy_champion": self.strategy_lab.champion(),
            "orders_generated": 0,
            "orders_submitted": 0,
        }
