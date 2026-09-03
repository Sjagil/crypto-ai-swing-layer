from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


TERMINAL_SUCCESS_STATES = {"ONESHOT_COMPLETE", "STOPPED_CLEAN"}


def parse_time(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def runtime_health(
    heartbeat: Path,
    *,
    max_age: float = 900.0,
) -> dict[str, Any]:
    failures: list[str] = []
    payload: dict[str, Any] = {}

    if not heartbeat.is_file():
        failures.append("HEARTBEAT_MISSING")
    else:
        try:
            payload = json.loads(
                heartbeat.read_text(encoding="utf-8")
            )
        except Exception as exc:
            failures.append(
                f"HEARTBEAT_INVALID:{type(exc).__name__}"
            )

    state = str(payload.get("state") or "UNKNOWN").upper()
    terminal_success = state in TERMINAL_SUCCESS_STATES

    observed = (
        payload.get("last_progress_at")
        or payload.get("heartbeat_at")
        or payload.get("completed_at")
    )
    age = None
    if observed:
        try:
            age = (
                datetime.now(timezone.utc)
                - parse_time(str(observed))
            ).total_seconds()
            if age > float(max_age) and not terminal_success:
                failures.append("HEARTBEAT_STALE")
        except Exception:
            failures.append("HEARTBEAT_TIMESTAMP_INVALID")
    elif payload:
        failures.append("HEARTBEAT_TIME_MISSING")

    pid = int(payload.get("pid") or 0) if payload else 0
    alive = pid_alive(pid) if pid else None
    if pid and not alive and not terminal_success:
        failures.append("SUPERVISOR_PID_NOT_ALIVE")

    task_status = str(
        payload.get("task_status") or ""
    ).upper()
    busy = state == "BUSY" or task_status == "RUNNING"
    runtime_errors = list(payload.get("errors") or [])
    if runtime_errors and not busy:
        failures.append("SUPERVISOR_REPORTED_ERRORS")

    if failures:
        status = "UNHEALTHY"
    elif state == "ONESHOT_COMPLETE":
        status = "ONESHOT_COMPLETE"
    elif state == "STOPPED_CLEAN":
        status = "STOPPED_CLEAN"
    elif busy:
        status = "HEALTHY_BUSY"
    else:
        status = "HEALTHY"

    return {
        "healthy": not failures,
        "status": status,
        "heartbeat": str(heartbeat),
        "age_seconds": age,
        "mode": payload.get("mode"),
        "schema_version": payload.get("schema_version"),
        "state": state,
        "pid": pid or None,
        "pid_alive": alive,
        "cycle_id": payload.get("cycle_id"),
        "current_task": payload.get("current_task"),
        "task_status": payload.get("task_status"),
        "last_completed_task": payload.get(
            "last_completed_task"
        ),
        "runtime_error_count": len(runtime_errors),
        "failures": failures,
    }
