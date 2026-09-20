from __future__ import annotations

import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crypto_ai_swing.agents.chief import ChiefAgent
from crypto_ai_swing.agents.hpo import HPOService
from crypto_ai_swing.agents.live_promotion import LiveModelGovernor
from crypto_ai_swing.universe.runtime import UniverseManager


def _now() -> datetime:
    return datetime.now(UTC)


class ContinuousLearningWorker:
    """Heavy learning/HPO/research process kept separate from live trading."""

    def __init__(self, settings, *, mode: str = "live") -> None:
        self.settings = settings
        self.mode = str(mode).lower()
        self.cfg = dict(settings.agents.get("continuous_learning", {}) or {})
        self.root = settings.project_root / "output/crypto_ai_swing/learning_worker"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.heartbeat_path = self.root / "heartbeat.json"
        self.errors_path = self.root / "errors.jsonl"
        self.state = self._read(self.state_path)
        os.environ["CRYPTO_SWING_LEARNING_PROCESS"] = "1"
        self.universe = UniverseManager(settings)
        self.hpo = HPOService(settings)
        self.chief = ChiefAgent(settings, mode=self.mode)
        self.governor = LiveModelGovernor(settings)

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return dict(value) if isinstance(value, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _parse(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)
        except Exception:
            return None

    def _due(self, key: str, seconds: int) -> bool:
        previous = self._parse(self.state.get(key))
        return previous is None or (
            _now() - previous.astimezone(UTC)
        ).total_seconds() >= max(60, int(seconds))

    @staticmethod
    def _atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)

    def run_once(self) -> dict[str, Any]:
        cycle_id = uuid.uuid4().hex
        started = _now()
        markets = [
            str(v).upper()
            for v in self.universe.current().get("markets", [])
            if str(v).strip()
        ]
        tasks: dict[str, Any] = {}
        errors: list[dict[str, str]] = []
        force_train = False

        if self._due("last_hpo_at", int(self.cfg.get("hpo_refresh_seconds", 43200))):
            try:
                result = self.hpo.run_supervised(
                    markets=markets,
                    timeframe=str(self.settings.agents.get("timeframe", "1h")),
                    horizon_bars=int(self.settings.agents.get("horizon_bars", 4)),
                    minimum_rows=int(self.settings.agents.get("minimum_rows", 8000)),
                )
                tasks["supervised_hpo"] = result.payload
                self.state["last_hpo_at"] = _now().isoformat()
                force_train = result.status == "READY"
            except Exception as exc:
                errors.append({
                    "task": "supervised_hpo",
                    "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                })

        if self._due(
            "last_rl_hpo_at",
            int(self.cfg.get("rl_hpo_refresh_seconds", 86400)),
        ):
            try:
                result = self.hpo.run_rl(
                    markets=markets,
                    timeframe=str(
                        (self.settings.agents.get("rl", {}) or {}).get(
                            "timeframe", "1h"
                        )
                    ),
                    minimum_rows_per_market=int(
                        (self.settings.agents.get("rl", {}) or {}).get(
                            "minimum_rows_per_market", 900
                        )
                    ),
                )
                tasks["rl_hpo"] = result.payload.get("rl") or result.payload
                self.state["last_rl_hpo_at"] = _now().isoformat()
                force_train = force_train or result.status == "READY"
            except Exception as exc:
                errors.append({
                    "task": "rl_hpo",
                    "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                })

        try:
            forward_cfg = dict(self.settings.autonomy.get("forward_evidence", {}) or {})
            forward_path = self.settings.project_root / forward_cfg.get(
                "path",
                "output/crypto_ai_swing/forward/forward_v3.sqlite",
            )
            tasks["chief_agent"] = self.chief.cycle(
                markets=markets,
                forward_database_path=forward_path,
                force_train=force_train,
                force_research=False,
                force_exact_research=False,
            )
        except Exception as exc:
            errors.append({
                "task": "chief_agent",
                "error": f"{type(exc).__name__}:{str(exc)[:500]}",
            })

        try:
            tasks["live_model_governor"] = self.governor.cycle()
        except Exception as exc:
            errors.append({
                "task": "live_model_governor",
                "error": f"{type(exc).__name__}:{str(exc)[:500]}",
            })

        self.state["last_cycle_at"] = _now().isoformat()
        self.state["last_cycle_had_errors"] = bool(errors)
        self._atomic(self.state_path, self.state)

        payload = {
            "schema_version": "crypto_ai_swing_learning_worker_v1",
            "cycle_id": cycle_id,
            "pid": os.getpid(),
            "mode": self.mode,
            "started_at": started.isoformat(),
            "completed_at": _now().isoformat(),
            "status": "DEGRADED" if errors else "HEALTHY",
            "markets": markets,
            "tasks": tasks,
            "errors": errors,
            "trading_process_separate": True,
            "continuous_data_consumption": True,
            "continuous_supervised_training": True,
            "continuous_rl_backprop": True,
            "continuous_hpo": True,
            "continuous_backtesting": True,
            "qualified_models_have_live_influence": True,
            "execution_authority_owned_by_pi": False,
            "execution_authority_owner": "Sjagil/crypto canonical execution gate",
        }
        self._atomic(self.heartbeat_path, payload)
        if errors:
            with self.errors_path.open("a", encoding="utf-8") as handle:
                for row in errors:
                    handle.write(json.dumps(
                        {"at": _now().isoformat(), **row},
                        sort_keys=True,
                    ) + "\n")
        return payload

    def run_forever(self) -> None:
        interval = max(30, int(self.cfg.get("cycle_seconds", 60)))
        while True:
            started = time.monotonic()
            try:
                self.run_once()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                with self.errors_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "at": _now().isoformat(),
                        "task": "worker",
                        "error": f"{type(exc).__name__}:{str(exc)[:1000]}",
                    }) + "\n")
            time.sleep(max(1.0, interval - (time.monotonic() - started)))
