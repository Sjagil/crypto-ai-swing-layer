from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from crypto_ai_swing.contracts import Authority, TradeIntent
from crypto_ai_swing.execution.crypto_authority import CryptoAuthorityAdapter


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
        rel = self.config.get("execution_adapter", {}).get(
            "intent_directory", "output/crypto_ai_swing/trade_intents"
        )
        return self.crypto_repo_root / rel

    def write_intent(self, intent: TradeIntent) -> Path:
        target = self._intent_dir()
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{intent.intent_id}.json"
        path.write_text(json.dumps(intent.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def route(self, intent: TradeIntent, mode: str = "shadow") -> RouteResult:
        if intent.expires_at <= datetime.now(UTC):
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
            if not command:
                return RouteResult(False, mode, path=path, blocker="EXTERNAL_ADAPTER_NOT_CONFIGURED")
            rendered = [str(token).replace("{intent}", str(path)) for token in command]
            proc = subprocess.run(rendered, cwd=self.crypto_repo_root, shell=False, check=False, timeout=120)
            return RouteResult(proc.returncode == 0, mode, path=path, return_code=proc.returncode)

        if mode != "live":
            return RouteResult(False, mode, path=path, blocker="UNKNOWN_MODE")
        if intent.authority != Authority.LIVE:
            return RouteResult(False, mode, path=path, blocker="INTENT_NOT_LIVE")
        if not execution.get("live_enabled", False):
            return RouteResult(False, mode, path=path, blocker="LIVE_DISABLED")

        adapter_mode = str(adapter.get("mode", "file_contract"))
        if adapter_mode == "direct_bitvavo":
            return RouteResult(False, mode, path=path, blocker="DIRECT_BITVAVO_EXECUTION_REMOVED_USE_CRYPTO_AUTHORITY")
        if adapter_mode != "crypto_execution_authority":
            return RouteResult(False, mode, path=path, blocker="CANONICAL_CRYPTO_EXECUTION_AUTHORITY_REQUIRED")

        authority = CryptoAuthorityAdapter(self.crypto_repo_root)
        try:
            preflight = authority.preflight(intent)
        except Exception as exc:
            return RouteResult(False, mode, path=path, blocker=f"CRYPTO_EXECUTION_PREFLIGHT_UNAVAILABLE:{type(exc).__name__}:{str(exc)[:300]}")
        if preflight.get("accepted") is not True:
            blockers = [str(v) for v in preflight.get("blockers", []) if str(v)]
            return RouteResult(False, mode, path=path, blocker=";".join(blockers) or "CRYPTO_AUTHORITY_PREFLIGHT_BLOCKED", response={"canonical_preflight": preflight})
        try:
            result = authority.submit_buy(intent)
        except Exception as exc:
            return RouteResult(False, mode, path=path, blocker=f"CRYPTO_EXECUTION_AUTHORITY_UNAVAILABLE:{type(exc).__name__}:{str(exc)[:300]}", response={"canonical_preflight": preflight})
        blockers = [str(v) for v in result.payload.get("blockers", []) if str(v)]
        return RouteResult(
            result.accepted,
            mode,
            path=path,
            blocker=None if result.accepted else ";".join(blockers) or str(result.payload.get("status") or result.payload.get("reason_code") or "CRYPTO_AUTHORITY_BLOCKED"),
            response={"canonical_preflight": preflight, "canonical_submission": result.payload},
        )
