"""Unified 24/7 Raspberry Pi learning/research worker.

Heavy model learning is intentionally separated from the Mac trading supervisor.
The worker may promote qualified model *influence* through existing governors,
but it never owns exchange execution, capital, order or risk authority.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crypto_ai_swing.agents.tcn_gru_live_governor import TCNGRULiveGovernor
from crypto_ai_swing.intelligence.cmc_startup_bridge import CMCStartupBridge
from crypto_ai_swing.models.tcn_gru import TCNGRUChallengerTrainer, default_tcn_gru_config
from crypto_ai_swing.orchestration.learning_worker import ContinuousLearningWorker
from crypto_ai_swing.research.cmc_feature_research import cmc_feature_research
from crypto_ai_swing.research.native import NativeResearchBridge
from crypto_ai_swing.universe.runtime import UniverseManager


def _now() -> datetime:
    return datetime.now(UTC)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, dict) else {}
    except Exception:
        return {}


def _atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class PiUnifiedLearningWorker:
    SCHEMA = "crypto_ai_swing_pi_learning_worker_v2"

    def __init__(self, settings, *, mode: str = "shadow", factory_exact: bool = False) -> None:
        os.environ["CRYPTO_SWING_LEARNING_PROCESS"] = "1"
        self.settings = settings
        self.mode = str(mode).lower()
        if self.mode not in {"shadow", "paper", "live"}:
            raise ValueError("mode must be shadow, paper, or live")
        self.factory_exact = bool(factory_exact)
        self.base = ContinuousLearningWorker(settings, mode=self.mode)
        self.universe = UniverseManager(settings)
        self.native_research = NativeResearchBridge(settings.crypto_repo_root)
        self.tcn_governor = TCNGRULiveGovernor(settings)
        self.cmc = CMCStartupBridge(settings)
        self.root = Path(settings.project_root) / "output/crypto_ai_swing/pi_learning_worker"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.heartbeat_path = self.root / "heartbeat.json"
        self.errors_path = self.root / "errors.jsonl"
        self.state = _read(self.state_path)
        self.agents = dict(settings.agents or {})
        self.tcn_cfg = dict(self.agents.get("tcn_gru") or {})
        self.continuous_cfg = dict(self.agents.get("continuous_learning") or {})

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
        prior = self._parse(self.state.get(key))
        if prior is None:
            return True
        return (_now() - prior.astimezone(UTC)).total_seconds() >= max(60, int(seconds))

    def _markets(self, base: dict[str, Any]) -> list[str]:
        rows = [str(v).upper() for v in (base.get("markets") or []) if str(v).strip()]
        if not rows:
            rows = [
                str(v).upper()
                for v in (self.universe.current().get("markets") or [])
                if str(v).strip()
            ]
        return list(dict.fromkeys(rows))

    def _run_tcn(self, markets: list[str]) -> dict[str, Any]:
        if not bool(self.tcn_cfg.get("enabled", True)):
            return {"status": "DISABLED"}
        pointer = Path(self.settings.project_root) / "output/crypto_ai_swing/agents/tcn_gru/latest.pointer.json"
        cadence = int(self.tcn_cfg.get("retrain_seconds", 21600))
        if pointer.is_file() and not self._due("last_tcn_train_at", cadence):
            return {"status": "NOT_DUE", "retrain_seconds": cadence}
        timeframe = str(self.tcn_cfg.get("timeframe", "1h"))
        horizon = int(self.tcn_cfg.get("horizon_bars", 4))
        minimum_rows = int(self.tcn_cfg.get("minimum_rows", 8000))
        trainer = TCNGRUChallengerTrainer(
            self.settings,
            config=default_tcn_gru_config(horizon_bars=horizon),
        )
        result = trainer.train(
            markets=markets,
            timeframe=timeframe,
            horizon_bars=horizon,
            minimum_rows=minimum_rows,
            minimum_net_move_bps=float(self.agents.get("minimum_net_move_bps", 65.0)),
            expires_days=30,
        )
        self.state["last_tcn_train_at"] = _now().isoformat()
        return {
            "status": result.status,
            "rows": result.row_count,
            "markets": list(result.markets),
            "dataset_id": result.dataset_id,
            "artifact_path": str(result.artifact_path),
            "manifest_path": str(result.manifest_path),
            "metrics": result.metrics,
            "retrain_seconds": cadence,
        }

    def _run_factory(self) -> dict[str, Any]:
        cadence = int(self.continuous_cfg.get("factory_refresh_seconds", 21600))
        if not self._due("last_factory_at", cadence):
            return {"status": "NOT_DUE", "refresh_seconds": cadence}
        raw = self.native_research.run_factory_campaign(
            maximum_rows=int(self.continuous_cfg.get("factory_maximum_rows", 50000)),
            execute_exact=self.factory_exact,
        )
        self.state["last_factory_at"] = _now().isoformat()
        return {
            **self.native_research.factory_summary(raw),
            "exact_execution": self.factory_exact,
            "refresh_seconds": cadence,
        }

    def _run_cmc_research(self, markets: list[str]) -> dict[str, Any]:
        cadence = int(self.continuous_cfg.get("cmc_feature_research_seconds", 21600))
        if not self._due("last_cmc_research_at", cadence):
            return {"status": "NOT_DUE", "refresh_seconds": cadence}
        result = cmc_feature_research(
            self.settings,
            markets,
            timeframe=str(self.continuous_cfg.get("cmc_feature_research_timeframe", "1d")),
            horizon_bars=int(self.continuous_cfg.get("cmc_feature_research_horizon_bars", 7)),
        )
        self.state["last_cmc_research_at"] = _now().isoformat()
        return result

    def smoke(self) -> dict[str, Any]:
        markets = [
            str(v).upper()
            for v in (self.universe.current().get("markets") or [])
            if str(v).strip()
        ]
        native = self.native_research.status()
        payload = {
            "schema_version": "crypto_ai_swing_pi_learning_smoke_v1",
            "generated_at": _now().isoformat(),
            "market_count": len(markets),
            "markets": markets,
            "cmc": self.cmc.status(),
            "native_research_ready": native.ready,
            "native_research_imported_modules": native.imported_modules,
            "tcn_enabled": bool(self.tcn_cfg.get("enabled", True)),
            "supervised_retrain_seconds": int((self.agents.get("manager") or {}).get("supervised_retrain_seconds", 14400)),
            "rl_retrain_seconds": int((self.agents.get("manager") or {}).get("rl_retrain_seconds", 86400)),
            "execution_authority_owned_by_pi": False,
            "orders_submitted": 0,
        }
        payload["status"] = (
            "PASS"
            if payload["market_count"] > 0 and native.ready
            else "DEGRADED"
        )
        return payload

    def run_once(self) -> dict[str, Any]:
        cycle_id = uuid.uuid4().hex
        started = _now()
        tasks: dict[str, Any] = {}
        errors: list[dict[str, str]] = []
        try:
            base = self.base.run_once()
            tasks["continuous_learning"] = base
        except Exception as exc:
            base = {}
            errors.append({"task":"continuous_learning","error":f"{type(exc).__name__}:{str(exc)[:800]}"})
        markets = self._markets(base)
        jobs = (
            ("cmc_status", lambda: self.cmc.status()),
            ("cmc_feature_research", lambda: self._run_cmc_research(markets)),
            ("tcn_gru_training", lambda: self._run_tcn(markets)),
            ("tcn_gru_governor", self.tcn_governor.cycle),
            ("strategy_factory", self._run_factory),
        )
        for name, fn in jobs:
            try:
                tasks[name] = fn()
            except Exception as exc:
                errors.append({"task":name,"error":f"{type(exc).__name__}:{str(exc)[:800]}"})
        self.state["last_cycle_at"] = _now().isoformat()
        self.state["last_cycle_had_errors"] = bool(errors)
        _atomic(self.state_path, self.state)
        payload = {
            "schema_version": self.SCHEMA,
            "cycle_id": cycle_id,
            "pid": os.getpid(),
            "mode": self.mode,
            "started_at": started.isoformat(),
            "completed_at": _now().isoformat(),
            "status": "DEGRADED" if errors else "HEALTHY",
            "markets": markets,
            "market_count": len(markets),
            "tasks": tasks,
            "errors": errors,
            "continuous_supervised": True,
            "continuous_rl": True,
            "continuous_hpo": True,
            "continuous_tcn_gru": True,
            "continuous_cmc_research": True,
            "continuous_strategy_research": True,
            "execution_authority_owned_by_pi": False,
            "execution_authority": False,
            "capital_authority": False,
            "risk_limit_authority": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        _atomic(self.heartbeat_path, payload)
        if errors:
            with self.errors_path.open("a", encoding="utf-8") as handle:
                for row in errors:
                    handle.write(json.dumps({"at":_now().isoformat(),**row}, sort_keys=True) + "\n")
        return payload

    def run_forever(self, *, poll_seconds: int = 60) -> None:
        interval = max(60, int(poll_seconds))
        while True:
            started = time.monotonic()
            try:
                print(json.dumps(self.run_once(), indent=2, default=str), flush=True)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                row={"at":_now().isoformat(),"task":"worker","error":f"{type(exc).__name__}:{str(exc)[:1000]}"}
                print(json.dumps(row, indent=2), flush=True)
                with self.errors_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True)+"\n")
            time.sleep(max(1.0, interval-(time.monotonic()-started)))


__all__ = ["PiUnifiedLearningWorker"]
