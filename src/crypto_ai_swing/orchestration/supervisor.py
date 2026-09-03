
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import threading
import time
import uuid
from typing import Any

from crypto_ai_swing.agents.training import AgentTrainer
from crypto_ai_swing.bridge.crypto_operations import NativeOperationsBridge
from crypto_ai_swing.orchestration.proactive import ProactiveTrader
from crypto_ai_swing.research.bootstrap import ColdStartResearchRunner
from crypto_ai_swing.research.native import NativeResearchBridge
from crypto_ai_swing.research.forward import ForwardEvidenceLedger
from crypto_ai_swing.universe.runtime import UniverseManager


def _now():
    return datetime.now(timezone.utc)


class AutonomousSupervisor:
    """Schedules decisions, universe, retraining, economics and research.

    It cannot arm live authority and cannot promote models to live.
    """

    def __init__(self, settings, *, mode: str = "shadow") -> None:
        self.settings = settings
        self.mode = mode
        self.cfg = dict(getattr(settings, "supervisor", {}) or {})
        self.state_path = settings.project_root / self.cfg.get(
            "state_path", "output/crypto_ai_swing/supervisor/state.json"
        )
        self.heartbeat_path = settings.project_root / self.cfg.get(
            "heartbeat_path", "output/crypto_ai_swing/supervisor/heartbeat.json"
        )
        self.universe = UniverseManager(settings)
        self.trader = ProactiveTrader(settings, mode=mode)
        self.trainer = AgentTrainer(settings)
        self.research = NativeResearchBridge(settings.crypto_repo_root)
        self.bootstrap_research = ColdStartResearchRunner(settings)
        self.operations = NativeOperationsBridge(
            settings.crypto_repo_root, project_root=settings.project_root
        )
        self.state = self._load()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = None
        self._heartbeat_lock = threading.Lock()
        self._cycle_id = None
        self._cycle_started_at = None
        self._current_task = 'IDLE'
        self._last_completed_task = None

    def _load(self):
        try:
            return json.loads(self.state_path.read_text()) if self.state_path.is_file() else {}
        except Exception:
            return {}

    def close(self):
        self.trader.close()

    def _due(self, key: str, every: int) -> bool:
        raw = self.state.get(key)
        if not raw:
            return True
        try:
            previous = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if previous.tzinfo is None:
                previous = previous.replace(tzinfo=timezone.utc)
        except ValueError:
            return True
        return (_now() - previous.astimezone(timezone.utc)).total_seconds() >= every

    def _atomic_write(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.operations.atomic_write_json(path, payload)
        except Exception:
            path.write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )

    def _save(self, payload):
        self._atomic_write(self.state_path, self.state)
        self._atomic_write(self.heartbeat_path, payload)

    def _progress_payload(self):
        now = _now().isoformat()
        return {
            "schema_version": "crypto_ai_swing_supervisor_heartbeat_v4",
            "mode": self.mode,
            "pid": os.getpid(),
            "cycle_id": self._cycle_id,
            "cycle_started_at": self._cycle_started_at,
            "state": "BUSY",
            "current_task": self._current_task,
            "task_status": "RUNNING",
            "last_completed_task": self._last_completed_task,
            "last_progress_at": now,
            "heartbeat_at": now,
            "live_authority_armed_by_supervisor": False,
            "automatic_model_live_promotion": False,
        }

    def _heartbeat_pump(self):
        interval = max(5.0, float(self.cfg.get("task_heartbeat_seconds", 30)))
        while not self._heartbeat_stop.wait(interval):
            with self._heartbeat_lock:
                self._atomic_write(self.heartbeat_path, self._progress_payload())

    def _start_progress(self):
        self._heartbeat_stop.clear()
        with self._heartbeat_lock:
            self._atomic_write(self.heartbeat_path, self._progress_payload())
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_pump,
            name="crypto-swing-progress-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_progress(self):
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)
        self._heartbeat_thread = None

    def _task(self, name: str):
        self._current_task = name

    def _prospective_canary_readiness(self) -> dict[str, Any]:
        forward_cfg = dict(
            (getattr(self.settings, "autonomy", {}) or {}).get(
                "forward_evidence", {}
            )
        )
        rel = forward_cfg.get(
            "path", "output/crypto_ai_swing/forward/forward.sqlite"
        )
        ledger = ForwardEvidenceLedger(
            self.settings.project_root / rel,
            decision_bucket_minutes=int(
                forward_cfg.get("decision_bucket_minutes", 15)
            ),
        )
        try:
            cfg = dict(forward_cfg.get("canary_readiness", {}) or {})
            return ledger.canary_readiness(
                primary_horizon_hours=int(
                    cfg.get("primary_horizon_hours", 4)
                ),
                minimum_unblocked_buy_outcomes=int(
                    cfg.get("minimum_unblocked_buy_outcomes", 30)
                ),
                minimum_distinct_markets=int(
                    cfg.get("minimum_distinct_markets", 5)
                ),
                minimum_observation_span_hours=float(
                    cfg.get("minimum_observation_span_hours", 72)
                ),
                minimum_mean_return_bps=float(
                    cfg.get("minimum_mean_return_bps", 0.0)
                ),
                minimum_positive_return_rate=float(
                    cfg.get("minimum_positive_return_rate", 0.50)
                ),
            )
        finally:
            ledger.close()

    def run_once(self, *, one_shot: bool = False) -> dict[str, Any]:
        started = _now()
        self._cycle_id = uuid.uuid4().hex
        self._cycle_started_at = started.isoformat()
        self._current_task = "STARTING"
        self._start_progress()
        tasks: dict[str, Any] = {}
        errors: list[dict[str, str]] = []
        try:
            self._task("universe")
            universe = self.universe.current()
            self._last_completed_task = "universe"
            tasks["universe"] = {
                "selected_size": universe.get("selected_size"),
                "markets": universe.get("markets"),
                "generated_at": universe.get("generated_at"),
                "liquidity_degraded": universe.get("liquidity_degraded"),
            }
        except Exception as exc:
            errors.append({"task": "universe", "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
            universe = {"markets": []}

        try:
            self._task("proactive")
            tasks["proactive"] = self.trader.cycle()
            self._last_completed_task = "proactive"
        except Exception as exc:
            errors.append({"task": "proactive", "error": f"{type(exc).__name__}: {str(exc)[:500]}"})

        try:
            self._task("prospective_canary_readiness")
            tasks["prospective_canary_readiness"] = (
                self._prospective_canary_readiness()
            )
            self._last_completed_task = "prospective_canary_readiness"
        except Exception as exc:
            errors.append({
                "task": "prospective_canary_readiness",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            })

        acfg = dict(getattr(self.settings, "agents", {}) or {})
        if bool(acfg.get("enabled", True)) and self._due(
            "last_agent_train_at", int(self.cfg.get("agent_retrain_seconds", 21600))
        ):
            try:
                self._task("agent_training")
                configured = [str(x).upper() for x in acfg.get("training_markets", []) if str(x).strip()]
                training_markets = configured or list(universe.get("markets") or [])
                result = self.trainer.train(
                    markets=training_markets,
                    timeframe=str(acfg.get("timeframe", "1h")),
                    horizon_bars=int(acfg.get("horizon_bars", 4)),
                    minimum_rows=int(acfg.get("minimum_rows", 1200)),
                    minimum_net_move_bps=float(acfg.get("minimum_net_move_bps", 65)),
                )
                tasks["agent_training"] = {
                    "status": result.status,
                    "artifact": str(result.artifact_path),
                    "rows": result.row_count,
                    "market_count": len(result.markets),
                    "metrics": result.metrics,
                }
                self.state["last_agent_train_at"] = _now().isoformat()
                self._last_completed_task = "agent_training"
            except Exception as exc:
                errors.append({"task": "agent_training", "error": f"{type(exc).__name__}: {str(exc)[:500]}"})

        if self._due("last_economics_at", int(self.cfg.get("economics_refresh_seconds", 21600))):
            try:
                self._task("economics")
                tasks["economics"] = self.research.bootstrap_economics()
                self.state["last_economics_at"] = _now().isoformat()
                self._last_completed_task = "economics"
            except Exception as exc:
                errors.append({"task": "economics", "error": f"{type(exc).__name__}: {str(exc)[:500]}"})

        if bool(self.cfg.get("research_enabled", True)) and self._due(
            "last_research_at", int(self.cfg.get("research_run_seconds", 86400))
        ):
            try:
                self._task("research")
                raw = self.research.run_factory_campaign(
                    maximum_rows=int(self.cfg.get("research_maximum_rows", 20000)),
                    execute_exact=bool(self.cfg.get("research_exact", False)),
                )
                tasks["research_native"] = self.research.factory_summary(raw)
                if str(raw.get("status") or "").startswith("COLD_START"):
                    tasks["research_bootstrap"] = self.bootstrap_research.run(
                        markets=list(universe.get("markets") or []),
                        timeframe="1h",
                    )
                self.state["last_research_at"] = _now().isoformat()
                self._last_completed_task = "research"
            except Exception as exc:
                errors.append({"task": "research", "error": f"{type(exc).__name__}: {str(exc)[:500]}"})

        warning_codes = tuple(
            f"SWING_{str(row['task']).upper()}_ERROR" for row in errors
        )
        try:
            tasks["native_operational_degradation"] = self.operations.degradation(
                warning=warning_codes
            )
        except Exception as exc:
            errors.append({
                "task": "native_operational_degradation",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            })

        previous_errors = bool(self.state.get("previous_cycle_had_errors", False))
        current_errors = bool(errors)
        markets = list(universe.get("markets") or [])
        if current_errors:
            self.operations.notify_system_event(
                "OPERATIONAL_DEGRADATION",
                {
                    "status": "DEGRADED",
                    "reason": ";".join(str(row["error"]) for row in errors[:3]),
                    "mode": self.mode,
                },
                allowed_markets=markets,
            )
        elif previous_errors:
            self.operations.notify_system_event(
                "OPERATIONAL_RECOVERY",
                {
                    "status": "HEALTHY",
                    "reason": "swing supervisor recovered",
                    "mode": self.mode,
                },
                allowed_markets=markets,
            )

        self.state["previous_cycle_had_errors"] = current_errors
        self.state["last_cycle_at"] = _now().isoformat()
        self._stop_progress()
        payload = {
            "schema_version": "crypto_ai_swing_supervisor_v4",
            "mode": self.mode,
            "pid": os.getpid(),
            "cycle_id": self._cycle_id,
            "started_at": started.isoformat(),
            "completed_at": _now().isoformat(),
            "last_progress_at": _now().isoformat(),
            "state": (
                "DEGRADED"
                if errors
                else "ONESHOT_COMPLETE"
                if one_shot
                else "HEALTHY"
            ),
            "task_status": (
                "COMPLETE"
                if one_shot and not errors
                else None
            ),
            "current_task": "IDLE",
            "last_completed_task": self._last_completed_task,
            "tasks": tasks,
            "native_reuse": self.operations.status(),
            "errors": errors,
            "live_authority_armed_by_supervisor": False,
            "automatic_model_live_promotion": False,
        }
        self._save(payload)
        return payload

    def run_forever(self):
        interval = max(10, int(self.cfg.get("cycle_interval_seconds", 60)))
        while True:
            started = time.time()
            try:
                self.run_once(one_shot=False)
            except KeyboardInterrupt:
                raise
            except Exception:
                pass
            time.sleep(max(1.0, interval - (time.time() - started)))
