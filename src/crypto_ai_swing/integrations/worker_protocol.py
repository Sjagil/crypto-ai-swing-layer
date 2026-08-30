from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import json


@dataclass(frozen=True)
class WorkerRequest:
    protocol_version: str
    request_id: str
    action: str
    payload: dict[str, Any]

    def dumps(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass(frozen=True)
class WorkerResponse:
    protocol_version: str
    request_id: str
    ok: bool
    payload: dict[str, Any]
    error: str | None = None

    def dumps(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)
