#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def parse_time(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--max-age", type=float, default=180.0)
    args = parser.parse_args()

    failures: list[str] = []
    payload = {}
    if not args.heartbeat.is_file():
        failures.append("HEARTBEAT_MISSING")
    else:
        try:
            payload = json.loads(args.heartbeat.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"HEARTBEAT_INVALID:{type(exc).__name__}")

    completed = payload.get("completed_at")
    age = None
    if completed:
        try:
            age = (
                datetime.now(timezone.utc) - parse_time(str(completed))
            ).total_seconds()
            if age > float(args.max_age):
                failures.append("HEARTBEAT_STALE")
        except Exception:
            failures.append("HEARTBEAT_TIMESTAMP_INVALID")
    elif payload:
        failures.append("HEARTBEAT_COMPLETED_AT_MISSING")

    runtime_errors = list(payload.get("errors") or []) if payload else []
    if runtime_errors:
        failures.append("SUPERVISOR_REPORTED_ERRORS")

    result = {
        "healthy": not failures,
        "heartbeat": str(args.heartbeat),
        "age_seconds": age,
        "mode": payload.get("mode"),
        "schema_version": payload.get("schema_version"),
        "runtime_error_count": len(runtime_errors),
        "failures": failures,
    }
    print(json.dumps(result, indent=2, default=str))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
