from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import time


@dataclass
class HealthState:
    market_data_ok: bool
    reconciliation_ok: bool
    stale: bool
    blockers: list[str]


def derive_health(bridge, max_age_seconds: float = 20.0) -> HealthState:
    blockers: list[str] = []
    ws = bridge.websocket_health() or {}
    state = bridge.realtime_market_state() or {}

    if ws and str(ws.get("supervisor_status", "")).upper() not in {"OK", "HEALTHY", "RUNNING"}:
        blockers.append("WEBSOCKET_HEALTH")
    generated = state.get("generated_at")
    stale = False
    if generated:
        try:
            ts = datetime.fromisoformat(str(generated).replace("Z", "+00:00"))
            stale = (datetime.now(timezone.utc) - ts).total_seconds() > max_age_seconds
        except Exception:
            stale = True
    if stale:
        blockers.append("MARKET_STATE_STALE")

    reconciliation_ok = bool(state.get("reconciliation_ok", True))
    if not reconciliation_ok:
        blockers.append("RECONCILIATION")

    return HealthState(not any(x.startswith("WEBSOCKET") for x in blockers), reconciliation_ok, stale, blockers)


def write_heartbeat(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    record["generated_at"] = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
