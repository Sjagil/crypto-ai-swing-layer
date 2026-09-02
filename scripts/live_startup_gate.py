#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def run_json(root: Path, python: str, args: list[str]) -> dict:
    proc = subprocess.run(
        [python, "-m", "crypto_ai_swing.cli", *args],
        cwd=root,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"{' '.join(args)} failed with code {proc.returncode}:\n{proc.stderr[-2000:]}"
        )
    text = proc.stdout.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"{' '.join(args)} did not return JSON:\n{text[-2000:]}"
        ) from exc


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    python = os.environ.get(
        "CRYPTO_SWING_PYTHON",
        str(root / ".venv" / "bin" / "python"),
    )
    evidence = run_json(root, python, ["edge-calibration-status"])
    readiness = dict(evidence.get("readiness") or {})
    canary = run_json(root, python, ["live-canary-preflight"])

    failures: list[str] = []
    if readiness.get("eligible") is not True:
        failures.extend(
            f"EVIDENCE:{x}"
            for x in readiness.get("blockers") or ["NOT_ELIGIBLE"]
        )
    if canary.get("ready") is not True:
        failures.extend(
            f"CANARY:{x}"
            for x in canary.get("blockers") or [canary.get("status") or "NOT_READY"]
        )

    if os.environ.get("CRYPTO_SWING_CANARY_EXECUTE") != "YES":
        failures.append("EXECUTION_ENV:CRYPTO_SWING_CANARY_EXECUTE_NOT_YES")

    payload = {
        "ready": not failures,
        "mode": "live",
        "failures": failures,
        "prospective_readiness": readiness,
        "canary_preflight": canary,
        "orders_submitted": 0,
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0 if not failures else 4


if __name__ == "__main__":
    raise SystemExit(main())
