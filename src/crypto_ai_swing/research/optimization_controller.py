from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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


class OptimizationController:
    """Evidence-triggered optimization scheduler with immutable safety gates."""

    SCHEMA = "crypto_ai_swing_round43_optimization_controller_v1"

    def __init__(self, settings) -> None:
        self.settings = settings
        self.project_root = Path(settings.project_root)
        autonomy = dict(getattr(settings, "autonomy", {}) or {})
        self.cfg = dict(autonomy.get("optimization_controller", {}) or {})
        self.selector_cfg = dict(autonomy.get("entry_selector", {}) or {})
        self.root = (
            self.project_root
            / "output/crypto_ai_swing/research/optimization_controller"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.latest_path = self.root / "latest.json"
        self.history_path = self.root / "history.jsonl"
        self.state = self._read(self.state_path)

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
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        temp.replace(path)

    def _due(self, key: str, seconds: int) -> bool:
        previous = _parse(self.state.get(key))
        if previous is None:
            return True
        return (
            _now() - previous.astimezone(UTC)
        ).total_seconds() >= max(60, int(seconds))

    def _complete_minimum(self) -> int:
        configured = self.cfg.get(
            "minimum_complete_horizon_observations_for_optuna"
        )
        if configured is not None:
            return max(1, int(configured))
        minimum_train = int(
            self.selector_cfg.get("minimum_train_observations", 24)
        )
        return max(60, minimum_train * 3)

    @staticmethod
    def _agent_state(agent_status: dict[str, Any], key: str) -> str:
        payload = dict(agent_status.get(key) or {})
        return str(
            payload.get("status")
            or payload.get("state")
            or "UNKNOWN"
        ).upper()

    def plan(
        self,
        *,
        evidence: dict[str, Any],
        drift: dict[str, Any],
        agent_status: dict[str, Any],
        strategy_status: dict[str, Any],
    ) -> dict[str, Any]:
        enabled = bool(self.cfg.get("enabled", True))
        primary = int(evidence.get("primary_horizon_outcomes") or 0)
        complete = int(
            evidence.get("complete_required_horizon_observations") or 0
        )
        previous_research_primary = int(
            self.state.get("last_research_primary_outcomes", 0)
        )
        previous_exact_complete = int(
            self.state.get("last_exact_complete_observations", 0)
        )
        previous_optuna_complete = int(
            self.state.get("last_optuna_complete_observations", 0)
        )
        new_primary = max(0, primary - previous_research_primary)
        new_exact_complete = max(0, complete - previous_exact_complete)
        new_optuna_complete = max(0, complete - previous_optuna_complete)

        issue_codes = {
            str(value)
            for value in drift.get("issue_codes", [])
        }
        agent_states = {
            key: self._agent_state(agent_status, key)
            for key in ("supervised", "rl")
        }
        agent_failure = any(
            value in {"ERROR", "EXPIRED", "FAILED", "NOT_TRAINED"}
            for value in agent_states.values()
        )

        discovery_minimum = int(
            evidence.get("minimum_discovery_observations") or 80
        )
        complete_minimum = self._complete_minimum()
        champion = strategy_status.get("champion")

        force_train = bool(
            enabled
            and agent_failure
            and self._due(
                "last_forced_train_at",
                int(self.cfg.get("force_train_cooldown_seconds", 14400)),
            )
        )

        evidence_delta_trigger = (
            primary >= discovery_minimum
            and new_primary
            >= int(
                self.cfg.get(
                    "minimum_new_primary_outcomes_for_research",
                    20,
                )
            )
        )
        missing_champion_trigger = (
            primary >= discovery_minimum
            and not champion
            and "NO_CHAMPION_WITH_DISCOVERY_EVIDENCE" in issue_codes
        )
        force_research = bool(
            enabled
            and (evidence_delta_trigger or missing_champion_trigger)
            and self._due(
                "last_forced_research_at",
                int(self.cfg.get("research_cooldown_seconds", 21600)),
            )
        )

        exact_minimum = max(
            complete_minimum,
            int(
                self.cfg.get(
                    "minimum_complete_horizon_observations_for_exact",
                    complete_minimum,
                )
            ),
        )
        force_exact = bool(
            enabled
            and complete >= exact_minimum
            and new_exact_complete
            >= int(
                self.cfg.get(
                    "minimum_new_complete_horizon_observations_for_exact",
                    20,
                )
            )
            and self._due(
                "last_forced_exact_research_at",
                int(self.cfg.get("exact_research_cooldown_seconds", 86400)),
            )
        )

        run_optuna = bool(
            enabled
            and complete >= complete_minimum
            and new_optuna_complete
            >= int(
                self.cfg.get(
                    "minimum_new_complete_horizon_observations_for_optuna",
                    12,
                )
            )
            and self._due(
                "last_selector_optuna_at",
                int(self.cfg.get("selector_optuna_cooldown_seconds", 86400)),
            )
        )

        force_external = bool(
            enabled
            and "EXTERNAL_RESEARCH_STALE" in issue_codes
            and self._due(
                "last_forced_external_research_at",
                int(
                    self.cfg.get(
                        "external_research_cooldown_seconds",
                        900,
                    )
                ),
            )
        )

        directives = {
            "force_train": force_train,
            "force_research": force_research,
            "force_exact_research": force_exact,
            "run_selector_optuna": run_optuna,
            "force_external_research": force_external,
        }
        raw = json.dumps(
            {
                "directives": directives,
                "primary": primary,
                "complete": complete,
                "new_primary": new_primary,
                "new_exact_complete": new_exact_complete,
                "new_optuna_complete": new_optuna_complete,
                "issue_codes": sorted(issue_codes),
            },
            sort_keys=True,
        ).encode()
        payload = {
            "schema_version": self.SCHEMA,
            "generated_at": _now().isoformat(),
            "status": "READY" if enabled else "DISABLED",
            "plan_id": hashlib.sha256(raw).hexdigest()[:24],
            "directives": directives,
            "evidence": {
                "primary_horizon_outcomes": primary,
                "complete_required_horizon_observations": complete,
                "new_primary_since_last_forced_research": new_primary,
                "new_complete_since_last_exact": new_exact_complete,
                "new_complete_since_last_optuna": new_optuna_complete,
            },
            "agent_states": agent_states,
            "strategy_champion_present": bool(champion),
            "issue_codes": sorted(issue_codes),
            "selector_optuna_trials": int(
                self.cfg.get("selector_optuna_trials", 40)
            ),
            "bounded_compute": True,
            "heavy_work_is_evidence_triggered": True,
            "threshold_relaxation_allowed": False,
            "risk_widening_allowed": False,
            "automatic_live_authority": False,
            "automatic_live_promotion": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        self._write(self.latest_path, payload)
        return payload

    def record_execution(
        self,
        *,
        plan: dict[str, Any],
        evidence: dict[str, Any],
        actions: dict[str, Any],
    ) -> dict[str, Any]:
        directives = dict(plan.get("directives") or {})
        now = _now().isoformat()
        primary = int(evidence.get("primary_horizon_outcomes") or 0)
        complete = int(
            evidence.get("complete_required_horizon_observations") or 0
        )

        if directives.get("force_train"):
            self.state["last_forced_train_at"] = now
        if directives.get("force_research"):
            self.state["last_forced_research_at"] = now
            self.state["last_research_primary_outcomes"] = primary
        if directives.get("force_exact_research"):
            self.state["last_forced_exact_research_at"] = now
            self.state["last_exact_complete_observations"] = complete
        if directives.get("run_selector_optuna"):
            self.state["last_selector_optuna_at"] = now
            self.state["last_optuna_complete_observations"] = complete
        if directives.get("force_external_research"):
            self.state["last_forced_external_research_at"] = now

        self.state["last_plan_id"] = plan.get("plan_id")
        self.state["last_execution_at"] = now
        self.state["last_actions"] = {
            key: (
                value.get("status")
                if isinstance(value, dict)
                else type(value).__name__
            )
            for key, value in actions.items()
        }
        self._write(self.state_path, self.state)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "recorded_at": now,
                        "plan": plan,
                        "actions": actions,
                    },
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA,
            "status": "READY",
            "state": dict(self.state),
            "latest_plan": self._read(self.latest_path),
            "threshold_relaxation_allowed": False,
            "automatic_live_authority": False,
            "automatic_live_promotion": False,
        }
