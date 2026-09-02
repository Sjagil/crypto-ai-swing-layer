#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--max-age", type=float, default=900.0)
    args = parser.parse_args()

    failures: list[str] = []
    payload: dict = {}
    if not args.heartbeat.is_file():
        failures.append("HEARTBEAT_MISSING")
    else:
        try:
            payload = json.loads(args.heartbeat.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"HEARTBEAT_INVALID:{type(exc).__name__}")

    observed = (
        payload.get("last_progress_at")
        or payload.get("heartbeat_at")
        or payload.get("completed_at")
    )
    age = None
    if observed:
        try:
            age = (datetime.now(timezone.utc) - parse_time(str(observed))).total_seconds()
            if age > float(args.max_age):
                failures.append("HEARTBEAT_STALE")
        except Exception:
            failures.append("HEARTBEAT_TIMESTAMP_INVALID")
    elif payload:
        failures.append("HEARTBEAT_TIME_MISSING")

    pid = int(payload.get("pid") or 0) if payload else 0
    alive = pid_alive(pid) if pid else None
    if pid and not alive:
        failures.append("SUPERVISOR_PID_NOT_ALIVE")

    state = str(payload.get("state") or "UNKNOWN").upper()
    task_status = str(payload.get("task_status") or "").upper()
    busy = state == "BUSY" or task_status == "RUNNING"
    runtime_errors = list(payload.get("errors") or [])
    if runtime_errors and not busy:
        failures.append("SUPERVISOR_REPORTED_ERRORS")

    status = "HEALTHY_BUSY" if not failures and busy else "HEALTHY" if not failures else "UNHEALTHY"
    result = {
        "healthy": not failures,
        "status": status,
        "heartbeat": str(args.heartbeat),
        "age_seconds": age,
        "mode": payload.get("mode"),
        "schema_version": payload.get("schema_version"),
        "state": state,
        "pid": pid or None,
        "pid_alive": alive,
        "cycle_id": payload.get("cycle_id"),
        "current_task": payload.get("current_task"),
        "task_status": payload.get("task_status"),
        "last_completed_task": payload.get("last_completed_task"),
        "runtime_error_count": len(runtime_errors),
        "failures": failures,
    }
    print(json.dumps(result, indent=2, default=str))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
