from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import subprocess

from .discovery import discover


class CryptoRepoBridge:
    """Narrow bridge to the existing crypto repository.

    The default bridge reads artifacts and executes explicitly configured commands.
    It does not import exchange modules into this process and never guesses a private
    order API.
    """

    def __init__(self, root: Path):
        self.root = root.resolve()

    def inventory(self) -> list[dict[str, Any]]:
        return [
            {"capability": x.capability, "found": x.found, "paths": list(x.paths)}
            for x in discover(self.root)
        ]

    def read_json(self, relative_path: str) -> dict[str, Any] | None:
        path = (self.root / relative_path).resolve()
        if self.root not in path.parents and path != self.root:
            raise ValueError("Path escapes crypto repository")
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def realtime_market_state(self) -> dict[str, Any] | None:
        candidates = (
            "output/trading_runtime_v1/realtime_market_state/latest.json",
            "data/normalized/realtime_market_state/latest.json",
        )
        for rel in candidates:
            data = self.read_json(rel)
            if data is not None:
                return data
        return None

    def websocket_health(self) -> dict[str, Any] | None:
        return self.read_json("output/trading_runtime_v1/websocket_supervisor/status.json")

    def cmc_provider_status(self) -> dict[str, Any] | None:
        return self.read_json("output/historical_data_v7/provider_status.json")

    def run_explicit(self, command: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
        if not command:
            raise ValueError("Explicit command is required")
        return subprocess.run(
            command,
            cwd=self.root,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
