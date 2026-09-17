from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crypto_ai_swing.monitoring.research_drift import ResearchDriftMonitor
from crypto_ai_swing.orchestration.unified_runtime import UnifiedAutonomyRuntime
from crypto_ai_swing.research.economic_optimizer import (
    EconomicImprovementController,
)
from crypto_ai_swing.research.evidence_maturation import (
    ProspectiveEvidenceMaturation,
)
from crypto_ai_swing.research.optimization_controller import (
    OptimizationController,
)
from crypto_ai_swing.research.paper_economics import (
    PaperEconomicEvaluator,
)
from crypto_ai_swing.research.selector_optuna import SelectorOptunaTuner


def _now() -> datetime:
    return datetime.now(UTC)


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class Round43AutomationController:
    """Persistent shadow/paper automation around the canonical Round42 runtime."""

    SCHEMA = "crypto_ai_swing_round43_automation_controller_v1"
    ALLOWED_MODES = frozenset({"shadow", "paper"})

    def __init__(
        self,
        settings,
        *,
        mode: str = "shadow",
        runtime=None,
        evidence=None,
        drift=None,
        optimizer=None,
    ) -> None:
        selected_mode = str(mode).lower()
        if selected_mode not in self.ALLOWED_MODES:
            raise ValueError(
                "ROUND43_AUTOMATION_LIVE_MODE_FORBIDDEN:"
                "use shadow or paper; live authority remains separate"
            )

        self.settings = settings
        self.mode = selected_mode
        self.project_root = Path(settings.project_root)
        self.autonomy = dict(getattr(settings, "autonomy", {}) or {})
        self.cfg = dict(self.autonomy.get("round43_automation", {}) or {})
        self.runtime = runtime or UnifiedAutonomyRuntime(
            settings,
            mode=self.mode,
        )
        self.evidence = evidence or ProspectiveEvidenceMaturation(settings)
        self.drift = drift or ResearchDriftMonitor(settings)
        self.optimizer = optimizer or OptimizationController(settings)
        self.paper_economics = PaperEconomicEvaluator(
            settings,
            mode=self.mode,
        )
        self.economic_optimizer = EconomicImprovementController(
            settings
        )
        self.root = (
            self.project_root
            / "output/crypto_ai_swing/autonomy/round43"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.latest_path = self.root / "latest.json"
        self.heartbeat_path = self.root / "heartbeat.json"
        self.events_path = self.root / "events.jsonl"
        self.errors_path = self.root / "runtime_errors.jsonl"
        self.lock_path = self.root / "runtime.lock"
        self.state = self._read(self.state_path)
        self._cycle_id: str | None = None
        self._current_task = "IDLE"
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return dict(value) if isinstance(value, dict) else {}
        except (OSError, TypeError, ValueError):
            return {}

    @staticmethod
    def _write(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        os.replace(temp, path)

    def _event(self, kind: str, payload: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "at": _now().isoformat(),
                        "cycle_id": self._cycle_id,
                        "kind": kind,
                        **payload,
                    },
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @contextmanager
    def _lease(self):
        self.root.mkdir(parents=True, exist_ok=True)
        stale_seconds = int(
            self.cfg.get("runtime_lock_stale_seconds", 1800)
        )
        if self.lock_path.exists():
            current = self._read(self.lock_path)
            pid = int(current.get("pid") or 0)
            age = time.time() - self.lock_path.stat().st_mtime
            if pid > 0 and self._pid_alive(pid):
                raise RuntimeError(
                    f"ROUND43_RUNTIME_ALREADY_RUNNING:pid={pid}"
                )
            if pid <= 0 and age < max(60, stale_seconds):
                raise RuntimeError(
                    "ROUND43_RECENT_UNOWNED_LOCK_REQUIRES_RECHECK"
                )
            self.lock_path.unlink(missing_ok=True)

        fd = os.open(
            self.lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(
                fd,
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "created_at": _now().isoformat(),
                        "mode": self.mode,
                    }
                ).encode("utf-8"),
            )
            os.close(fd)
            fd = -1
            yield
        finally:
            if fd >= 0:
                os.close(fd)
            self.lock_path.unlink(missing_ok=True)

    def _due(self, key: str, seconds: int) -> bool:
        previous = _parse(self.state.get(key))
        if previous is None:
            return True
        return (
            _now() - previous.astimezone(UTC)
        ).total_seconds() >= max(10, int(seconds))

    def _heartbeat_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "crypto_ai_swing_round43_heartbeat_v1",
            "heartbeat_at": _now().isoformat(),
            "pid": os.getpid(),
            "mode": self.mode,
            "cycle_id": self._cycle_id,
            "current_task": self._current_task,
            "state": (
                "IDLE" if self._current_task == "IDLE" else "BUSY"
            ),
            "automatic_live_authority": False,
            "automatic_live_promotion": False,
        }

    def _heartbeat_pump(self) -> None:
        interval = max(
            5.0,
            float(self.cfg.get("heartbeat_seconds", 30)),
        )
        while not self._heartbeat_stop.wait(interval):
            self._write(
                self.heartbeat_path,
                self._heartbeat_payload(),
            )

    def _start_heartbeat(self) -> None:
        self._heartbeat_stop.clear()
        self._write(self.heartbeat_path, self._heartbeat_payload())
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_pump,
            name="crypto-swing-round43-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
        self._heartbeat_thread = None
        self._current_task = "IDLE"
        self._write(self.heartbeat_path, self._heartbeat_payload())

    def _markets(self) -> list[str]:
        try:
            payload = self.runtime.universe.current()
            return [
                str(value).upper()
                for value in payload.get("markets", [])
                if str(value).strip()
            ]
        except Exception:  # noqa: BLE001 - diagnostics fallback
            return []

    def _strategy_status(self) -> dict[str, Any]:
        try:
            return (
                self.runtime.base.chief_agent.strategy_director
                .strategy_status()
            )
        except Exception:  # noqa: BLE001 - status fallback
            return {}

    def _agent_status(self) -> dict[str, Any]:
        try:
            return self.runtime.agent_manager.status()
        except Exception:  # noqa: BLE001 - status fallback
            return {}

    def _run_forced_chief(
        self,
        plan: dict[str, Any],
        markets: list[str],
    ) -> dict[str, Any]:
        directives = dict(plan.get("directives") or {})
        if not any(
            directives.get(key)
            for key in (
                "force_train",
                "force_research",
                "force_exact_research",
                "force_external_research",
            )
        ):
            return {"status": "NOT_TRIGGERED", "orders_submitted": 0}

        return self.runtime.base.chief_agent.cycle(
            markets=markets,
            forward_database_path=self.evidence.path,
            force_train=bool(directives.get("force_train")),
            force_research=bool(directives.get("force_research")),
            force_exact_research=bool(
                directives.get("force_exact_research")
            ),
            browser_research=(
                True
                if directives.get("force_external_research")
                else None
            ),
        )

    def _run_selector_optuna(
        self,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        directives = dict(plan.get("directives") or {})
        if not directives.get("run_selector_optuna"):
            return {"status": "NOT_TRIGGERED", "orders_submitted": 0}
        tuner = SelectorOptunaTuner(self.runtime.entry_selector)
        return tuner.run(
            trials=int(plan.get("selector_optuna_trials") or 40),
            study_name=str(
                self.optimizer.cfg.get(
                    "selector_optuna_study_name",
                    "prospective-swing-entry-selector-round43",
                )
            ),
            seed=int(self.optimizer.cfg.get("selector_optuna_seed", 43)),
            fold_count=int(
                self.optimizer.cfg.get(
                    "selector_optuna_inner_folds",
                    3,
                )
            ),
        )

    def _notify_transition(
        self,
        *,
        degraded: bool,
        errors: list[dict[str, str]],
        drift: dict[str, Any],
        markets: list[str],
    ) -> None:
        previous = bool(self.state.get("previous_cycle_degraded", False))
        if previous == degraded:
            return
        try:
            self.runtime.base.operations.notify_system_event(
                (
                    "OPERATIONAL_DEGRADATION"
                    if degraded
                    else "OPERATIONAL_RECOVERY"
                ),
                {
                    "status": "DEGRADED" if degraded else "HEALTHY",
                    "reason": (
                        ";".join(
                            row.get("error", "")
                            for row in errors[:3]
                        )
                        or ",".join(drift.get("issue_codes", [])[:5])
                        or "round43 automation recovered"
                    ),
                    "mode": self.mode,
                    "component": "ROUND43_AUTOMATION",
                },
                allowed_markets=markets,
            )
        except Exception:  # noqa: BLE001 - notification is non-authoritative
            return

    def _cycle(self, *, one_shot: bool) -> dict[str, Any]:
        self._cycle_id = uuid.uuid4().hex
        started = _now()
        errors: list[dict[str, str]] = []
        actions: dict[str, Any] = {}

        self._current_task = "UNIFIED_RUNTIME"
        self._event("TASK_START", {"task": self._current_task})
        try:
            base = self.runtime.run_once(one_shot=one_shot)
        except Exception as exc:  # noqa: BLE001 - daemon isolation boundary
            base = {"state": "ERROR", "errors": []}
            errors.append(
                {
                    "task": "unified_runtime",
                    "error": f"{type(exc).__name__}:{str(exc)[:600]}",
                }
            )
        self._event(
            "TASK_COMPLETE",
            {
                "task": self._current_task,
                "status": base.get("state"),
            },
        )

        self._current_task = "EVIDENCE_MATURATION"
        maturation_seconds = int(
            self.cfg.get("evidence_maturation_seconds", 900)
        )
        if self._due("last_evidence_maturation_at", maturation_seconds):
            evidence = self.evidence.mature()
            self.state["last_evidence_maturation_at"] = _now().isoformat()
        else:
            evidence = self.evidence.snapshot()
            evidence["maturation_status"] = "NOT_DUE"
        if evidence.get("maturation_status") == "ERROR":
            errors.append(
                {
                    "task": "evidence_maturation",
                    "error": str(
                        (evidence.get("maturation") or {}).get(
                            "error",
                            "UNKNOWN_EVIDENCE_MATURATION_ERROR",
                        )
                    ),
                }
            )

        self._current_task = "PAPER_ECONOMICS"
        try:
            paper_economics = self.paper_economics.refresh()
            economic_plan = self.economic_optimizer.plan(
                paper_economics
            )
        except Exception as exc:  # noqa: BLE001
            paper_economics = {
                "status": "ERROR",
                "error": (
                    f"{type(exc).__name__}:"
                    f"{str(exc)[:600]}"
                ),
                "automatic_live_authority": False,
                "automatic_live_promotion": False,
                "orders_submitted": 0,
            }
            economic_plan = {
                "force_train": False,
                "force_research": False,
                "research_priorities": [],
            }
            errors.append(
                {
                    "task": "paper_economics",
                    "error": paper_economics["error"],
                }
            )

        agent_status = self._agent_status()
        strategy_status = self._strategy_status()

        self._current_task = "DRIFT_DIAGNOSTICS"
        drift = self.drift.evaluate(
            evidence=evidence,
            agent_status=agent_status,
            strategy_status=strategy_status,
        )

        self._current_task = "OPTIMIZATION_PLAN"
        plan = self.optimizer.plan(
            evidence=evidence,
            drift=drift,
            agent_status=agent_status,
            strategy_status=strategy_status,
        )

        merged_plan = dict(plan)
        directives = dict(
            merged_plan.get("directives") or {}
        )
        directives["force_train"] = bool(
            directives.get("force_train")
            or economic_plan.get("force_train")
        )
        directives["force_research"] = bool(
            directives.get("force_research")
            or economic_plan.get("force_research")
        )
        merged_plan["directives"] = directives
        merged_plan["economic_improvement"] = economic_plan
        merged_plan["research_priorities"] = list(
            economic_plan.get("research_priorities") or []
        )
        plan = merged_plan

        markets = self._markets()
        self._current_task = "BOUNDED_OPTIMIZATION"
        try:
            actions["chief_optimization"] = self._run_forced_chief(
                plan,
                markets,
            )
        except Exception as exc:  # noqa: BLE001 - one action must not kill daemon
            actions["chief_optimization"] = {
                "status": "ERROR",
                "error": f"{type(exc).__name__}:{str(exc)[:600]}",
            }
            errors.append(
                {
                    "task": "chief_optimization",
                    "error": actions["chief_optimization"]["error"],
                }
            )

        try:
            actions["selector_optuna"] = self._run_selector_optuna(plan)
        except Exception as exc:  # noqa: BLE001 - research optimizer fails closed
            actions["selector_optuna"] = {
                "status": "ERROR",
                "error": f"{type(exc).__name__}:{str(exc)[:600]}",
                "authority": "RESEARCH_ONLY",
                "orders_submitted": 0,
            }
            errors.append(
                {
                    "task": "selector_optuna",
                    "error": actions["selector_optuna"]["error"],
                }
            )

        self.optimizer.record_execution(
            plan=plan,
            evidence=evidence,
            actions=actions,
        )

        refreshed_strategy = self._strategy_status()
        stage = str(evidence.get("research_stage") or "NO_EVIDENCE")
        if refreshed_strategy.get("champion"):
            stage = "RESEARCH_CHAMPION"
        optuna = dict(actions.get("selector_optuna") or {})
        if bool(optuna.get("qualified")):
            stage = "OPTUNA_QUALIFIED_RESEARCH"
        if bool(
            paper_economics.get("economic_qualification")
        ):
            stage = "PAPER_ECONOMIC_QUALIFIED_RESEARCH"

        base_errors = list(base.get("errors") or [])
        for row in base_errors:
            if isinstance(row, dict):
                errors.append(
                    {
                        "task": str(row.get("task") or "base"),
                        "error": str(row.get("error") or "UNKNOWN"),
                    }
                )

        degraded = bool(errors) or str(drift.get("status")) == "DEGRADED"
        self._notify_transition(
            degraded=degraded,
            errors=errors,
            drift=drift,
            markets=markets,
        )
        self.state["previous_cycle_degraded"] = degraded
        self.state["last_cycle_at"] = _now().isoformat()
        self.state["last_cycle_id"] = self._cycle_id
        self.state["mode"] = self.mode
        self._write(self.state_path, self.state)

        payload = {
            "schema_version": self.SCHEMA,
            "cycle_id": self._cycle_id,
            "mode": self.mode,
            "pid": os.getpid(),
            "started_at": started.isoformat(),
            "completed_at": _now().isoformat(),
            "state": (
                "DEGRADED"
                if degraded
                else "ONESHOT_COMPLETE"
                if one_shot
                else "HEALTHY"
            ),
            "maturation_stage": stage,
            "base_runtime": {
                "schema_version": base.get("schema_version"),
                "state": base.get("state"),
                "cycle_id": base.get("cycle_id"),
            },
            "evidence": evidence,
            "paper_economics": paper_economics,
            "drift": drift,
            "optimization_plan": plan,
            "optimization_actions": actions,
            "strategy_status": refreshed_strategy,
            "agent_status": self._agent_status(),
            "errors": errors,
            "automation": {
                "persistent_runtime": True,
                "continuous_market_cycles": True,
                "continuous_forward_evidence": True,
                "continuous_evidence_maturation": True,
                "continuous_attribution": True,
                "continuous_paper_economic_attribution": True,
                "realized_pnl_feedback": True,
                "continuous_strategy_generation": True,
                "continuous_strategy_testing": True,
                "continuous_agent_training": True,
                "evidence_triggered_exact_research": True,
                "evidence_triggered_selector_optuna": True,
                "drift_detection": True,
                "browser_and_rss_research": True,
                "crash_recovery": True,
                "single_process_lease": True,
            },
            "safety": {
                "allowed_modes": sorted(self.ALLOWED_MODES),
                "live_mode_allowed": False,
                "automatic_live_authority": False,
                "automatic_live_promotion": False,
                "threshold_relaxation_allowed": False,
                "risk_widening_allowed": False,
                "direct_exchange_submission_by_round43": False,
            },
            "orders_generated_by_round43_control_plane": 0,
            "orders_submitted_by_round43_control_plane": 0,
            "performance_improvement_guaranteed": False,
        }
        self._write(self.latest_path, payload)
        return payload

    def run_once(self) -> dict[str, Any]:
        with self._lease():
            self._start_heartbeat()
            try:
                return self._cycle(one_shot=True)
            finally:
                self._stop_heartbeat()

    def run_forever(self) -> None:
        interval = max(
            10,
            int(self.cfg.get("cycle_seconds", 60)),
        )
        backoff = max(
            5,
            int(self.cfg.get("crash_backoff_seconds", 15)),
        )
        with self._lease():
            while True:
                started = time.monotonic()
                self._start_heartbeat()
                try:
                    self._cycle(one_shot=False)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001 - persistent boundary
                    row = {
                        "at": _now().isoformat(),
                        "error": type(exc).__name__,
                        "detail": str(exc)[:1200],
                        "mode": self.mode,
                    }
                    with self.errors_path.open(
                        "a",
                        encoding="utf-8",
                    ) as handle:
                        handle.write(json.dumps(row) + "\n")
                    self._event("CYCLE_CRASH", row)
                    time.sleep(backoff)
                finally:
                    self._stop_heartbeat()
                elapsed = time.monotonic() - started
                time.sleep(max(1.0, interval - elapsed))

    def close(self) -> None:
        self.runtime.close()

    def status(self) -> dict[str, Any]:
        return self._read(self.latest_path) or {
            "schema_version": self.SCHEMA,
            "state": "NOT_BUILT",
            "mode": self.mode,
            "automatic_live_authority": False,
            "automatic_live_promotion": False,
        }
