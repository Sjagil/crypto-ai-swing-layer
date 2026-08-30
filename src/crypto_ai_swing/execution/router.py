from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess

from crypto_ai_swing.contracts import Authority, TradeIntent
from crypto_ai_swing.execution.bitvavo import live_gate_status


@dataclass(frozen=True)
class RouteResult:
    accepted: bool
    mode: str
    path: Path | None = None
    return_code: int | None = None
    blocker: str | None = None
    response: dict | None = None


class ExecutionRouter:
    def __init__(self, project_root: Path, crypto_repo_root: Path, config: dict):
        self.project_root = project_root
        self.crypto_repo_root = crypto_repo_root
        self.config = config

    def _intent_dir(self) -> Path:
        rel = self.config.get("execution_adapter", {}).get("intent_directory", "output/crypto_ai_swing/trade_intents")
        return self.crypto_repo_root / rel

    def write_intent(self, intent: TradeIntent) -> Path:
        target = self._intent_dir()
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{intent.intent_id}.json"
        path.write_text(json.dumps(intent.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def route(self, intent: TradeIntent, mode: str = "shadow") -> RouteResult:
        now = datetime.now(timezone.utc)
        if intent.expires_at <= now:
            return RouteResult(False, mode, blocker="INTENT_EXPIRED")
        path = self.write_intent(intent)
        if mode == "shadow":
            return RouteResult(True, mode, path=path)

        execution = self.config.get("execution", {})
        adapter = self.config.get("execution_adapter", {})
        if mode == "paper":
            if not execution.get("paper_enabled", True):
                return RouteResult(False, mode, path=path, blocker="PAPER_DISABLED")
            command = adapter.get("command", [])
        elif mode == "live":
            if intent.authority != Authority.LIVE:
                return RouteResult(False, mode, path=path, blocker="INTENT_NOT_LIVE")
            if not execution.get("live_enabled", False):
                return RouteResult(False, mode, path=path, blocker="LIVE_DISABLED")
            gate = live_gate_status(self.config)
            if not gate.ready:
                return RouteResult(False, mode, path=path, blocker=";".join(gate.blockers))
            if str(adapter.get("mode", "file_contract")) == "direct_bitvavo":
                return RouteResult(
                    False,
                    mode,
                    path=path,
                    blocker="DIRECT_BITVAVO_EXECUTION_DISABLED_USE_CRYPTO_AUTHORITY",
                )
            if str(adapter.get("mode", "file_contract")) == "crypto_execution_authority":
                return RouteResult(
                    False,
                    mode,
                    path=path,
                    blocker="CRYPTO_EXECUTION_AUTHORITY_SUBMISSION_NOT_MAPPED",
                )
            command = adapter.get("live_command", [])
        else:
            return RouteResult(False, mode, path=path, blocker="UNKNOWN_MODE")

        if not command:
            return RouteResult(False, mode, path=path, blocker="EXTERNAL_ADAPTER_NOT_CONFIGURED")
        rendered = [str(token).replace("{intent}", str(path)) for token in command]
        proc = subprocess.run(rendered, cwd=self.crypto_repo_root, shell=False, check=False, timeout=120)
        return RouteResult(proc.returncode == 0, mode, path=path, return_code=proc.returncode)
