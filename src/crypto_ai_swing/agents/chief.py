from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crypto_ai_swing.agents.manager import AgentManager
from crypto_ai_swing.agents.performance_governor import (
    PerformanceGovernor,
)
from crypto_ai_swing.bridge.crypto_operations import NativeOperationsBridge
from crypto_ai_swing.intelligence.external_research import (
    ExternalResearchPlane,
)
from crypto_ai_swing.research.autonomous_strategy_director import (
    AutonomousStrategyDirector,
)
from crypto_ai_swing.research.quant_foundation import QuantFoundationAudit


class ChiefAgent:
    """Top-level agent-of-agents and continuous performance research governor."""

    SCHEMA = "crypto_ai_swing_chief_agent_v1"

    def __init__(
        self,
        settings,
        *,
        mode: str = "shadow",
        agent_manager=None,
        governor=None,
        strategy_director=None,
        external_research=None,
        quant_audit=None,
    ) -> None:
        self.settings = settings
        self.mode = str(mode).lower()
        self.cfg = dict(
            (getattr(settings, "autonomy", {}) or {}).get(
                "chief_agent", {}
            )
            or {}
        )
        self.agent_manager = agent_manager or AgentManager(
            settings, mode=self.mode
        )
        self.governor = governor or PerformanceGovernor(settings)
        self.strategy_director = (
            strategy_director
            or AutonomousStrategyDirector(
                settings, mode=self.mode
            )
        )
        self.external_research = (
            external_research or ExternalResearchPlane(settings)
        )
        self.quant_audit = quant_audit or QuantFoundationAudit(settings)
        self.operations = NativeOperationsBridge(
            settings.crypto_repo_root,
            project_root=settings.project_root,
        )
        # Existing edge manager consumes the same prospective strategy champion.
        self.agent_manager.edge_manager.strategy_lab = (
            self.strategy_director.strategy_lab
        )
        self.root = (
            Path(settings.project_root)
            / "output/crypto_ai_swing/agents/chief"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.root / "latest.json"
        self.history_path = self.root / "history.jsonl"
        self.state_path = self.root / "state.json"

    def _read(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return dict(value) if isinstance(value, dict) else {}
        except (OSError, TypeError, ValueError):
            return {}

    def _browser_due(self) -> bool:
        state = self._read(self.state_path)
        raw = state.get("last_external_research_at")
        if not raw:
            return True
        try:
            previous = datetime.fromisoformat(str(raw))
            if previous.tzinfo is None:
                previous = previous.replace(tzinfo=UTC)
        except (TypeError, ValueError):
            return True
        seconds = int(
            (getattr(self.settings, "autonomy", {}) or {})
            .get("external_research", {})
            .get("refresh_seconds", 900)
        )
        return (
            datetime.now(UTC) - previous.astimezone(UTC)
        ).total_seconds() >= max(60, seconds)

    def cycle(
        self,
        *,
        markets: list[str],
        forward_database_path: Path,
        force_train: bool = False,
        force_research: bool = False,
        force_exact_research: bool = False,
        browser_research: bool | None = None,
    ) -> dict[str, Any]:
        tasks: dict[str, Any] = {}
        errors: list[dict[str, str]] = []

        try:
            tasks["quant_foundation"] = self.quant_audit.run()
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "task": "quant_foundation",
                    "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                }
            )

        try:
            tasks["agent_manager"] = self.agent_manager.cycle(
                markets=markets,
                forward_database_path=forward_database_path,
                force_train=force_train,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "task": "agent_manager",
                    "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                }
            )
            tasks["agent_manager"] = self.agent_manager.status()

        agent_status = self.agent_manager.status()
        try:
            preliminary = self.governor.evaluate(
                agent_status=agent_status,
                attribution=self.strategy_director.attribution.status()
                if hasattr(
                    self.strategy_director.attribution, "status"
                )
                else {},
                strategy=self.strategy_director.strategy_status(),
                native_research=self.strategy_director.status(),
            )
            tasks["performance_governor_before_research"] = preliminary
        except Exception as exc:  # noqa: BLE001
            preliminary = {"priorities": []}
            errors.append(
                {
                    "task": "performance_governor_before_research",
                    "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                }
            )

        try:
            tasks["strategy_director"] = self.strategy_director.cycle(
                markets=markets,
                priorities=list(preliminary.get("priorities") or []),
                force=force_research,
                force_exact=force_exact_research,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "task": "strategy_director",
                    "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                }
            )
            tasks["strategy_director"] = self.strategy_director.status()

        external_cfg = dict(
            (getattr(self.settings, "autonomy", {}) or {}).get(
                "external_research", {}
            )
            or {}
        )
        should_browser = (
            bool(browser_research)
            if browser_research is not None
            else bool(external_cfg.get("enabled", True))
            and self._browser_due()
        )
        if should_browser:
            try:
                tasks["external_research"] = (
                    self.external_research.collect()
                )
                state = self._read(self.state_path)
                state["last_external_research_at"] = (
                    datetime.now(UTC).isoformat()
                )
                self.operations.atomic_write_json(
                    self.state_path, state
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "task": "external_research",
                        "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                    }
                )
        else:
            tasks["external_research"] = {
                "status": "NOT_DUE",
                "orders_submitted": 0,
            }

        try:
            final_governor = self.governor.evaluate(
                agent_status=self.agent_manager.status(),
                attribution=(
                    tasks.get("strategy_director", {})
                    .get("tasks", {})
                    .get("attribution", {})
                ),
                strategy=(
                    tasks.get("strategy_director", {})
                    .get("tasks", {})
                    .get(
                        "strategy_lab",
                        self.strategy_director.strategy_status(),
                    )
                ),
                native_research=(
                    tasks.get("strategy_director", {})
                    .get("tasks", {})
                    .get("canonical_research_factory", {})
                ),
            )
            tasks["performance_governor"] = final_governor
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "task": "performance_governor",
                    "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                }
            )
            final_governor = self.governor.status()

        payload = {
            "schema_version": self.SCHEMA,
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "DEGRADED" if errors else "HEALTHY",
            "mode": self.mode,
            "markets": markets,
            "tasks": tasks,
            "errors": errors,
            "continuous_improvement_loop": True,
            "agent_of_agents": True,
            "agents_consume_strategy_champion": True,
            "continuous_strategy_generation": True,
            "continuous_strategy_testing": True,
            "continuous_model_training": True,
            "browser_and_rss_research": True,
            "performance_objective": (
                "robust prospective net performance after full costs"
            ),
            "improvement_is_not_guaranteed": True,
            "only_evidence_improving_challengers_may_gain_influence": True,
            "automatic_live_authority": False,
            "automatic_live_promotion": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        self.operations.atomic_write_json(self.latest_path, payload)
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(payload, sort_keys=True, default=str) + "\n"
            )
        return payload

    def status(self) -> dict[str, Any]:
        return self._read(self.latest_path) or {
            "schema_version": self.SCHEMA,
            "status": "NOT_BUILT",
            "agent_of_agents": True,
            "automatic_live_authority": False,
            "automatic_live_promotion": False,
        }
