from __future__ import annotations

import json
from pathlib import Path

from crypto_ai_swing.orchestration.health import runtime_health


def test_completed_oneshot_is_not_dead_pid_failure(
    tmp_path: Path,
):
    heartbeat = tmp_path / "heartbeat.json"
    heartbeat.write_text(
        json.dumps(
            {
                "schema_version": (
                    "crypto_ai_swing_supervisor_v5"
                ),
                "mode": "paper",
                "state": "ONESHOT_COMPLETE",
                "pid": 99999999,
                "completed_at": (
                    "2020-01-01T00:00:00+00:00"
                ),
                "last_progress_at": (
                    "2020-01-01T00:00:00+00:00"
                ),
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    report = runtime_health(
        heartbeat,
        max_age=1,
    )
    assert report["healthy"] is True
    assert report["status"] == "ONESHOT_COMPLETE"
    assert (
        "SUPERVISOR_PID_NOT_ALIVE"
        not in report["failures"]
    )
    assert "HEARTBEAT_STALE" not in report["failures"]
