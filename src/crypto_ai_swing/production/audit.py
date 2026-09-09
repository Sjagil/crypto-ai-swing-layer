from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

SENSITIVE_FRAGMENTS = (
    "api_key", "apikey", "api_secret", "secret", "password", "passphrase",
    "credential", "authorization", "private_key", "access_token", "refresh_token",
)

def now_iso() -> str:
    return datetime.now(UTC).isoformat()

def redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            lower = name.lower()
            result[name] = (
                "[REDACTED]"
                if any(fragment in lower for fragment in SENSITIVE_FRAGMENTS)
                else redact(item)
            )
        return result
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value

class ProductionAuditLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        row = {"at": now_iso(), "event": str(event), "payload": redact(dict(payload or {}))}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
