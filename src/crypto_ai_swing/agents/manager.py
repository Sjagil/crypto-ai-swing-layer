from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crypto_ai_swing.agents.edge_manager import ResearchEdgeManager
from crypto_ai_swing.agents.rl_multi_market import MultiMarketRLTrainer
from crypto_ai_swing.agents.rl_runtime import RLRuntime
from crypto_ai_swing.agents.runtime import AgentRuntime
from crypto_ai_swing.agents.training import AgentTrainer
from crypto_ai_swing.agents.prospective_context import ProspectiveContextAgent
from crypto_ai_swing.research.feature_attribution import Round44FeatureAttribution
from crypto_ai_swing.research.live_readiness import Round44LiveReadiness
from crypto_ai_swing.bridge.crypto_operations import NativeOperationsBridge
from crypto_ai_swing.research.promotion import ResearchPromotionRegistry
from crypto_ai_swing.universe.runtime import UniverseManager


def _now() -> datetime:
    return datetime.now(UTC)


class AgentManager:
    """Agent-of-agents for bounded continuous training and evidence management."""

    SCHEMA = "crypto_ai_swing_agent_manager_v1"

    def __init__(
        self,
        settings,
        *,
        mode: str = "shadow",
        trainer=None,
        runtime=None,
        rl_trainer=None,
        rl_runtime=None,
        edge_manager=None,
        registry=None,
        universe=None,
    ) -> None:
        self.settings = settings
        self.mode = str(mode).lower()
        self.cfg = dict(
            (getattr(settings, "agents", {}) or {}).get("manager", {}) or {}
        )
        self.root = (
            settings.project_root / "output/crypto_ai_swing/agents/manager"
        )
        self.state_path = self.root / "state.json"
        self.history_path = self.root / "history.jsonl"
        self.lock_path = self.root / "training.lock"
        self.trainer = trainer or AgentTrainer(settings)
        self.runtime = runtime or AgentRuntime(settings, mode=self.mode)
        self.rl_trainer = rl_trainer or MultiMarketRLTrainer(settings)
        self.rl_runtime = rl_runtime or RLRuntime(settings)
        self.round44_context_agent = ProspectiveContextAgent(settings)
        self.round44_attribution = Round44FeatureAttribution(settings)
        self.round44_readiness = Round44LiveReadiness(settings)
        self.edge_manager = edge_manager or ResearchEdgeManager(
            settings,
            mode="shadow" if self.mode == "live" else self.mode,
        )
        self.registry = registry or ResearchPromotionRegistry(settings)
        self.universe = universe or UniverseManager(settings)
        self.operations = NativeOperationsBridge(
            settings.crypto_repo_root,
            project_root=settings.project_root,
        )
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            payload = json.loads(
                self.state_path.read_text(encoding="utf-8")
            )
            return dict(payload) if isinstance(payload, dict) else {}
        except (OSError, TypeError, ValueError):
            return {}

    def _write(self, payload: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.operations.atomic_write_json(self.state_path, payload)
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(payload, sort_keys=True, default=str) + "\n"
            )

    @staticmethod
    def _parse_time(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(
                str(value)
            )
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except (TypeError, ValueError):
            return None

    def _due(
        self,
        key: str,
        seconds: int,
        *,
        immediate_statuses: set[str] | None = None,
        current_status: str | None = None,
    ) -> bool:
        if (
            current_status
            and immediate_statuses
            and current_status.upper() in immediate_statuses
        ):
            return True
        previous = self._parse_time(self.state.get(key))
        return previous is None or (
            _now() - previous.astimezone(UTC)
        ).total_seconds() >= max(60, int(seconds))

    @contextmanager
    def _training_lease(self):
        self.root.mkdir(parents=True, exist_ok=True)
        stale_after = int(self.cfg.get("stale_lock_seconds", 21600))
        if self.lock_path.exists():
            age = time.time() - self.lock_path.stat().st_mtime
            if age > stale_after:
                self.lock_path.unlink(missing_ok=True)
            else:
                raise RuntimeError("AGENT_MANAGER_TRAINING_ALREADY_RUNNING")
        fd = os.open(
            self.lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(
                fd,
                json.dumps(
                    {"pid": os.getpid(), "created_at": _now().isoformat()}
                ).encode("utf-8"),
            )
            os.close(fd)
            fd = -1
            yield
        finally:
            if fd >= 0:
                os.close(fd)
            self.lock_path.unlink(missing_ok=True)

    def status(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA,
            "mode": self.mode,
            "supervised": self.runtime.status(),
            "rl": self.rl_runtime.status(),
            "last_supervised_train_at": self.state.get(
                "last_supervised_train_at"
            ),
            "last_rl_train_at": self.state.get("last_rl_train_at"),
            "last_edge_refresh_at": self.state.get("last_edge_refresh_at"),
            "last_registry_refresh_at": self.state.get(
                "last_registry_refresh_at"
            ),
            "continuous_monitoring": True,
            "automatic_live_authority": False,
            "automatic_model_live_promotion": False,
        }

    def cycle(
        self,
        *,
        markets: list[str] | None = None,
        forward_database_path: Path | None = None,
        force_train: bool = False,
    ) -> dict[str, Any]:
        if not bool(self.cfg.get("enabled", True)):
            return {
                **self.status(),
                "status": "DISABLED",
                "orders_submitted": 0,
            }

        selected = [
            str(value).upper()
            for value in (markets or [])
            if str(value).strip()
        ]
        if not selected:
            selected = [
                str(value).upper()
                for value in self.universe.current().get("markets", [])
                if str(value).strip()
            ]

        errors: list[dict[str, str]] = []
        tasks: dict[str, Any] = {}
        supervised_status = str(
            self.runtime.status().get("status") or "NOT_TRAINED"
        ).upper()
        rl_status = str(
            self.rl_runtime.status().get("status") or "NOT_TRAINED"
        ).upper()

        supervised_due = force_train or self._due(
            "last_supervised_train_at",
            int(self.cfg.get("supervised_retrain_seconds", 14400)),
            immediate_statuses={"NOT_TRAINED", "ERROR", "EXPIRED"},
            current_status=supervised_status,
        )
        rl_enabled = bool(
            ((getattr(self.settings, "agents", {}) or {}).get("rl", {}) or {})
            .get("enabled", False)
        )
        rl_due = rl_enabled and (
            force_train
            or self._due(
                "last_rl_train_at",
                int(self.cfg.get("rl_retrain_seconds", 86400)),
                immediate_statuses={"NOT_TRAINED", "ERROR", "EXPIRED"},
                current_status=rl_status,
            )
        )

        if supervised_due:
            try:
                with self._training_lease():
                    acfg = dict(getattr(self.settings, "agents", {}) or {})
                    result = self.trainer.train(
                        markets=selected,
                        timeframe=str(acfg.get("timeframe", "1h")),
                        horizon_bars=int(acfg.get("horizon_bars", 4)),
                        minimum_rows=int(acfg.get("minimum_rows", 8000)),
                        minimum_net_move_bps=float(
                            acfg.get("minimum_net_move_bps", 65.0)
                        ),
                    )
                supervised_markets = [
                    str(value).upper() for value in result.markets
                ]
                tasks["supervised_training"] = {
                    "status": result.status,
                    "artifact": str(result.artifact_path),
                    "dataset_id": result.dataset_id,
                    "rows": result.row_count,
                    "markets": supervised_markets,
                    "requested_markets": list(selected),
                    "requested_market_count": len(selected),
                    "trained_market_count": len(supervised_markets),
                    "training_coverage_fraction": (
                        len(set(supervised_markets) & set(selected))
                        / max(1, len(selected))
                    ),
                    "metrics": result.metrics,
                }
                self.state["last_supervised_train_at"] = _now().isoformat()
            except Exception as exc:
                errors.append(
                    {
                        "task": "supervised_training",
                        "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                    }
                )
        else:
            tasks["supervised_training"] = {"status": "NOT_DUE"}

        if rl_due:
            try:
                with self._training_lease():
                    acfg = dict(getattr(self.settings, "agents", {}) or {})
                    rcfg = dict(acfg.get("rl", {}) or {})
                    result = self.rl_trainer.train(
                        markets=selected,
                        timeframe=str(rcfg.get("timeframe", "1h")),
                        total_timesteps=int(
                            rcfg.get("total_timesteps", 50000)
                        ),
                        minimum_rows_per_market=int(
                            rcfg.get("minimum_rows_per_market", 900)
                        ),
                    )
                rl_payload = dict(result)
                rl_markets = [
                    str(value).upper()
                    for value in (rl_payload.get("markets") or [])
                ]
                rl_payload["requested_markets"] = list(selected)
                rl_payload["requested_market_count"] = len(selected)
                rl_payload["trained_market_count"] = len(rl_markets)
                rl_payload["training_coverage_fraction"] = (
                    len(set(rl_markets) & set(selected))
                    / max(1, len(selected))
                )
                tasks["rl_training"] = rl_payload
                self.state["last_rl_train_at"] = _now().isoformat()
            except Exception as exc:
                errors.append(
                    {
                        "task": "rl_training",
                        "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                    }
                )
        else:
            tasks["rl_training"] = {"status": "NOT_DUE"}

        context_due = (
            forward_database_path is not None
            and self._due(
                "last_round44_context_train_at",
                int(self.cfg.get("round44_context_retrain_seconds", 21600)),
            )
        )
        if context_due:
            try:
                tasks["round44_context_training"] = self.round44_context_agent.train(
                    Path(forward_database_path),
                    horizon_hours=4,
                    minimum_rows=int(self.cfg.get("round44_context_minimum_rows", 150)),
                    minimum_markets=5,
                    maximum_features=72,
                )
                if tasks["round44_context_training"].get("status") != "NO_NEW_EVIDENCE":
                    self.state["last_round44_context_train_at"] = _now().isoformat()
            except Exception as exc:
                errors.append(
                    {
                        "task": "round44_context_training",
                        "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                    }
                )
        else:
            tasks["round44_context_training"] = {"status": "NOT_DUE_OR_NO_LEDGER"}

        attribution_due = (
            forward_database_path is not None
            and self._due(
                "last_round44_attribution_at",
                int(self.cfg.get("round44_attribution_seconds", 900)),
            )
        )
        if attribution_due:
            try:
                tasks["round44_attribution"] = self.round44_attribution.evaluate(
                    Path(forward_database_path),
                    horizons_hours=(1, 4, 24, 72, 168),
                    minimum_observations=int(
                        self.cfg.get("round44_attribution_minimum_observations", 60)
                    ),
                    maximum_features=80,
                )
                self.state["last_round44_attribution_at"] = _now().isoformat()
            except Exception as exc:
                errors.append(
                    {
                        "task": "round44_attribution",
                        "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                    }
                )
        else:
            tasks["round44_attribution"] = {"status": "NOT_DUE_OR_NO_LEDGER"}

        if forward_database_path is not None:
            try:
                tasks["round44_readiness"] = self.round44_readiness.evaluate(
                    Path(forward_database_path)
                )
            except Exception as exc:
                errors.append(
                    {
                        "task": "round44_readiness",
                        "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                    }
                )
        else:
            tasks["round44_readiness"] = {"status": "NO_FORWARD_LEDGER"}

        edge_due = self._due(
            "last_edge_refresh_at",
            int(self.cfg.get("edge_refresh_seconds", 900)),
        )
        if forward_database_path is not None and edge_due:
            try:
                tasks["edge_manager"] = self.edge_manager.refresh_policy(
                    Path(forward_database_path),
                    force=True,
                )
                self.state["last_edge_refresh_at"] = _now().isoformat()
            except Exception as exc:
                errors.append(
                    {
                        "task": "edge_manager",
                        "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                    }
                )
        else:
            tasks["edge_manager"] = {"status": "NOT_DUE_OR_NO_LEDGER"}

        registry_due = self._due(
            "last_registry_refresh_at",
            int(self.cfg.get("registry_refresh_seconds", 3600)),
        )
        if registry_due:
            try:
                tasks["promotion_registry"] = self.registry.refresh()
                self.state["last_registry_refresh_at"] = _now().isoformat()
            except Exception as exc:
                errors.append(
                    {
                        "task": "promotion_registry",
                        "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                    }
                )
        else:
            tasks["promotion_registry"] = {"status": "NOT_DUE"}

        payload = {
            "schema_version": self.SCHEMA,
            "status": "DEGRADED" if errors else "HEALTHY",
            "checked_at": _now().isoformat(),
            "mode": self.mode,
            "markets": selected,
            "tasks": tasks,
            "runtime_after": self.runtime.status(),
            "rl_runtime_after": self.rl_runtime.status(),
            "errors": errors,
            "continuous_monitoring": True,
            "training_is_bounded_and_due_driven": True,
            "runtime_universe_training_markets": list(selected),
            "runtime_universe_training_market_count": len(selected),
            "full_25_market_training_requested": len(selected) == 25,
            "retrain_missing_expired_or_error_immediately": True,
            "challengers_research_only": True,
            "live_decision_influence": False,
            "automatic_live_authority": False,
            "automatic_model_live_promotion": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        self.state.update(
            {
                key: value
                for key, value in payload.items()
                if key in {"status", "checked_at", "mode", "markets", "errors"}
            }
        )
        self._write({**self.state, **payload})
        return payload
