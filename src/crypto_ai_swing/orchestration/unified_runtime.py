from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.orchestration.supervisor import AutonomousSupervisor
from crypto_ai_swing.research.entry_selector import ProspectiveSwingEntrySelector
from crypto_ai_swing.research.forward import ForwardEvidenceLedger
from crypto_ai_swing.research.performance_attribution import PerformanceAttributionEngine
from crypto_ai_swing.research.promotion import ResearchPromotionRegistry
from crypto_ai_swing.research.strategy_challenger import StrategyChallengerLab
from crypto_ai_swing.research.swing_geometry import SwingGeometryEngine
from crypto_ai_swing.universe.runtime import UniverseManager


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    every_seconds: int
    fn: Callable[[], dict[str, Any]]


class UnifiedAutonomyRuntime:
    """One process for trading, evidence, attribution and challenger research."""

    SCHEMA = "crypto_ai_swing_unified_autonomy_v4"

    def __init__(self, settings, *, mode: str = "shadow") -> None:
        self.settings = settings
        self.mode = str(mode)
        self.cfg = dict(getattr(settings, "supervisor", {}) or {})
        self.autonomy = dict(getattr(settings, "autonomy", {}) or {})
        self.project_root = Path(settings.project_root)
        self.output_root = self.project_root / "output/crypto_ai_swing/autonomy"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_root / "state.json"
        self.latest_path = self.output_root / "latest.json"
        self.events_path = self.output_root / "events.jsonl"
        self.base = AutonomousSupervisor(settings, mode=self.mode)
        self.universe = UniverseManager(settings)
        self.crypto = CryptoLibraryBridge(settings.crypto_repo_root)
        self.attribution = PerformanceAttributionEngine(settings, mode=self.mode)
        self.strategy_lab = StrategyChallengerLab(settings, mode=self.mode)
        self.swing_geometry = SwingGeometryEngine(settings, mode=self.mode)
        self.entry_selector = ProspectiveSwingEntrySelector(
            settings, mode=self.mode
        )
        self.agent_manager = self.base.agent_manager
        self.edge = self.agent_manager.edge_manager
        self.registry = ResearchPromotionRegistry(settings)
        self.state = self._load_state()
        self._cycle_id: str | None = None

    def close(self) -> None:
        self.base.close()

    def _load_state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _parse_time(raw: Any) -> datetime | None:
        if not raw:
            return None
        try:
            value = datetime.fromisoformat(str(raw))
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        except Exception:
            return None

    def _due(self, task: str, seconds: int) -> bool:
        previous = self._parse_time(self.state.get("tasks", {}).get(task, {}).get("completed_at"))
        return previous is None or (_now() - previous.astimezone(UTC)).total_seconds() >= seconds

    def _atomic_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(tmp, path)

    def _event(self, kind: str, payload: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": _now().isoformat(), "cycle_id": self._cycle_id, "kind": kind, **payload}, sort_keys=True, default=str) + "\n")

    def _forward_config(self) -> dict[str, Any]:
        autonomy = getattr(
            self,
            "autonomy",
            getattr(getattr(self, "settings", None), "autonomy", {}),
        )
        return dict((autonomy or {}).get("forward_evidence", {}) or {})

    def _execution_timeframe(self) -> str:
        cfg = self._forward_config()
        proactive = dict(getattr(self.settings, "proactive", {}) or {})
        return str(cfg.get("execution_timeframe", proactive.get("execution_timeframe", "15m")))

    def _forward_ledger(self) -> ForwardEvidenceLedger:
        cfg = self._forward_config()
        rel = cfg.get("path", "output/crypto_ai_swing/forward/forward.sqlite")
        return ForwardEvidenceLedger(
            self.project_root / rel,
            decision_bucket_minutes=int(cfg.get("decision_bucket_minutes", 15)),
        )

    def _mature_forward(self) -> dict[str, Any]:
        cfg = self._forward_config()
        execution_tf = self._execution_timeframe()
        ledger = self._forward_ledger()
        try:
            if bool(cfg.get("mature_on_cycle", True)):
                return {
                    "status": "DELEGATED_TO_PROACTIVE",
                    "execution_timeframe": execution_tf,
                    "causal_reference": "FIRST_EXECUTION_TIMEFRAME_OPEN_AT_OR_AFTER_OBSERVED_AT",
                    **ledger.outcome_status(),
                    "orders_submitted": 0,
                }
            markets = list(ledger.observation_markets())
            if not markets:
                return {"status": "NO_OBSERVATIONS", "markets": 0, "execution_timeframe": execution_tf, "orders_submitted": 0}
            frames = self.crypto.ohlcv_many(markets, execution_tf, persist=False, concurrency=4)
            result = ledger.mature_from_frames(
                frames,
                horizons_hours=tuple(int(x) for x in cfg.get("horizons_hours", (1, 4, 24)) if int(x) > 0),
                execution_timeframe=execution_tf,
            )
            result["status"] = "COMPLETE"
            return result
        finally:
            ledger.close()

    def _refresh_attribution(self) -> dict[str, Any]:
        return self.attribution.refresh()

    def _refresh_strategy_lab(self) -> dict[str, Any]:
        return self.strategy_lab.refresh()

    def _refresh_swing_geometry(self) -> dict[str, Any]:
        return self.swing_geometry.refresh(force=True)

    def _refresh_entry_selector(self) -> dict[str, Any]:
        return self.entry_selector.refresh(force=True)

    def _refresh_edge(self) -> dict[str, Any]:
        ledger = self._forward_ledger()
        try:
            result = self.agent_manager.cycle(
                markets=[
                    str(value).upper()
                    for value in self.universe.current().get("markets", [])
                    if str(value).strip()
                ],
                forward_database_path=ledger.path,
            )
            return {
                "status": "DELEGATED_TO_AGENT_MANAGER",
                "agent_manager_status": result.get("status"),
                "edge_manager": result.get("tasks", {}).get(
                    "edge_manager", {}
                ),
                "orders_submitted": 0,
            }
        finally:
            ledger.close()

    def _rl_tournament(self) -> dict[str, Any]:
        return {
            "status": "DELEGATED_TO_AGENT_MANAGER",
            "agent_manager": self.agent_manager.status(),
            "live_decision_influence": False,
            "automatic_live_promotion": False,
            "orders_submitted": 0,
        }

    def _registry_refresh(self) -> dict[str, Any]:
        return self.registry.refresh()

    def _task_specs(self) -> tuple[TaskSpec, ...]:
        # Backward-compatible core contract retained from v0.23.
        return (
            TaskSpec("forward_maturation", int(self.cfg.get("forward_mature_seconds", 900)), self._mature_forward),
            TaskSpec("edge_policy", int(self.cfg.get("edge_refresh_seconds", 900)), self._refresh_edge),
            TaskSpec("registry", int(self.cfg.get("promotion_registry_seconds", 3600)), self._registry_refresh),
            TaskSpec("rl_tournament", int(self.cfg.get("rl_retrain_seconds", 86400)), self._rl_tournament),
        )

    def _v024_task_specs(self) -> tuple[TaskSpec, ...]:
        attr = dict(getattr(self, "autonomy", {}).get("performance_attribution", {}) or {})
        lab = dict(getattr(self, "autonomy", {}).get("strategy_lab", {}) or {})
        return (
            TaskSpec("performance_attribution", int(attr.get("refresh_seconds", 900)), self._refresh_attribution),
            TaskSpec("strategy_lab", int(lab.get("refresh_seconds", 3600)), self._refresh_strategy_lab),
        )

    def _v026_task_specs(self) -> tuple[TaskSpec, ...]:
        geometry = dict(getattr(self, "autonomy", {}).get("swing_geometry", {}) or {})
        return (
            TaskSpec(
                "swing_geometry",
                int(geometry.get("refresh_seconds", 900)),
                self._refresh_swing_geometry,
            ),
        )

    def _v027_task_specs(self) -> tuple[TaskSpec, ...]:
        selector = dict(
            getattr(self, "autonomy", {}).get("entry_selector", {}) or {}
        )
        return (
            TaskSpec(
                "entry_selector",
                int(selector.get("refresh_seconds", 900)),
                self._refresh_entry_selector,
            ),
        )

    def _record_task(self, name: str, result: dict[str, Any] | None, error: Exception | None = None) -> None:
        row: dict[str, Any] = {
            "completed_at": _now().isoformat(),
            "status": "ERROR" if error else str((result or {}).get("status") or "COMPLETE"),
        }
        if result is not None:
            row["summary"] = {
                key: result.get(key)
                for key in (
                    "status", "qualified", "observations", "outcomes_inserted", "outcomes",
                    "markets", "market_count", "selected_seed", "execution_timeframe",
                    "shadow_influence", "live_decision_influence", "automatic_live_promotion",
                    "known_trial_count", "qualified_research_candidates",
                )
                if key in result
            }
        if error is not None:
            row["error"] = f"{type(error).__name__}:{str(error)[:500]}"
        self.state.setdefault("tasks", {})[name] = row

    def run_once(self, *, one_shot: bool = False) -> dict[str, Any]:
        self._cycle_id = uuid.uuid4().hex
        started = _now()
        results: dict[str, Any] = {}
        errors: list[dict[str, str]] = []
        try:
            results["trading_and_core_supervisor"] = self.base.run_once(one_shot=one_shot)
            core = results["trading_and_core_supervisor"]
            if self.mode in {"paper", "shadow"} and isinstance(core, dict):
                proactive = dict(
                    (core.get("tasks", {}) or {}).get("proactive", {}) or {}
                )
                paper_accounting = dict(
                    proactive.get("paper_accounting", {}) or {}
                )
                if paper_accounting.get("status") == "RECONCILIATION_REQUIRED":
                    errors.append({
                        "task": "paper_accounting",
                        "error": "PAPER_ACCOUNT_RECONCILIATION_REQUIRED",
                    })
        except Exception as exc:
            errors.append({"task": "trading_and_core_supervisor", "error": f"{type(exc).__name__}:{str(exc)[:500]}"})

        for spec in (
            *self._v024_task_specs(),
            *self._v026_task_specs(),
            *self._v027_task_specs(),
            *self._task_specs(),
        ):
            if not self._due(spec.name, spec.every_seconds):
                results[spec.name] = {"status": "NOT_DUE", "every_seconds": spec.every_seconds}
                continue
            self._event("TASK_START", {"task": spec.name})
            try:
                value = spec.fn()
                results[spec.name] = value
                self._record_task(spec.name, value)
                self._event("TASK_COMPLETE", {"task": spec.name, "status": value.get("status")})
            except Exception as exc:
                errors.append({"task": spec.name, "error": f"{type(exc).__name__}:{str(exc)[:500]}"})
                self._record_task(spec.name, None, exc)
                self._event("TASK_ERROR", {"task": spec.name, "error": f"{type(exc).__name__}:{str(exc)[:500]}"})

        registry = results.get("registry")
        if not isinstance(registry, dict) or registry.get("status") == "NOT_DUE":
            try:
                registry = self.registry.status()
            except Exception:
                registry = {"status": "UNAVAILABLE"}
        self.state["last_cycle_at"] = _now().isoformat()
        self.state["last_cycle_id"] = self._cycle_id
        self.state["mode"] = self.mode
        self._atomic_json(self.state_path, self.state)

        payload = {
            "schema_version": self.SCHEMA,
            "cycle_id": self._cycle_id,
            "mode": self.mode,
            "pid": os.getpid(),
            "started_at": started.isoformat(),
            "completed_at": _now().isoformat(),
            "state": "DEGRADED" if errors else "ONESHOT_COMPLETE" if one_shot else "HEALTHY",
            "components": results,
            "promotion_registry": registry,
            "errors": errors,
            "continuous_research": True,
            "performance_attribution_enabled": True,
            "stateful_champion_challenger_enabled": True,
            "agent_manager_enabled": True,
            "multi_horizon_swing_geometry_enabled": True,
            "advanced_linear_algebra_enabled": True,
            "prospective_swing_entry_selector_enabled": True,
            "abstention_is_first_class": True,
            "production_rc_enabled": True,
            "durable_execution_recovery_enabled": True,
            "automatic_crash_resubmission": False,
            "champion_is_immutable_during_execution": True,
            "challengers_research_only": True,
            "live_authority_granted_by_runtime": False,
            "automatic_model_live_promotion": False,
            "orders_submitted_by_autonomy_control_plane": 0,
        }
        self._atomic_json(self.latest_path, payload)
        return payload

    def run_forever(self) -> None:
        interval = max(10, int(self.cfg.get("cycle_interval_seconds", 60)))
        while True:
            started = time.monotonic()
            self.run_once(one_shot=False)
            time.sleep(max(1.0, interval - (time.monotonic() - started)))
