#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_ai_swing.bridge.crypto_library import (
    CryptoLibraryBridge,
)
from crypto_ai_swing.bridge.reference_stack import (
    ReferenceStack,
)
from crypto_ai_swing.research.entry_selector import (
    ProspectiveSwingEntrySelector,
)
from crypto_ai_swing.settings import Settings


def selector_status(settings) -> dict:
    selector = ProspectiveSwingEntrySelector(
        settings,
        mode="paper",
    )
    rows = selector._load_rows()
    required = tuple(
        int(value) for value in selector.horizons
    )
    by_observation = {}
    for row in rows:
        oid = str(row["observation_id"])
        by_observation.setdefault(oid, set()).add(
            int(row["horizon_hours"])
        )
    complete = sum(
        set(required).issubset(values)
        for values in by_observation.values()
    )
    minimum = max(selector.minimum_train * 3, 60)
    return {
        "status": (
            "READY"
            if complete >= minimum
            else "COLLECTING"
        ),
        "complete_horizons_required": list(required),
        "observations_any_required_label": len(
            by_observation
        ),
        "complete_horizon_observations": complete,
        "minimum_complete_calibration_observations": (
            minimum
        ),
        "ready": complete >= minimum,
    }


def preflight(live: bool) -> dict:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "v1_rc_preflight.py"),
        "--markets",
        "BTC-EUR,ETH-EUR,SOL-EUR,LINK-EUR",
    ]
    if not live:
        command.append("--offline")
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {
            "status": "ERROR",
            "blockers": [
                "V1_RC_PREFLIGHT_INVALID_OUTPUT"
            ],
            "stderr": proc.stderr[-1000:],
        }
    payload["returncode"] = proc.returncode
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "also perform canonical private account/"
            "reconciliation checks"
        ),
    )
    args = parser.parse_args()

    settings = Settings.load(ROOT)
    bridge = CryptoLibraryBridge(
        settings.crypto_repo_root
    )
    crypto_modules = bridge.integration_status()
    references = ReferenceStack(ROOT).status()
    selector = selector_status(settings)

    crypto_reference_health = {
        "status": "NOT_RUN",
        "ready": False,
    }
    try:
        module = bridge.import_module(
            "reporting.reference_integration_health"
        )
        result = module.build_reference_integration_health(
            settings.crypto_repo_root
        )
        health = dict(result.get("payload") or result)
        crypto_reference_health = {
            "status": health.get("status"),
            "live_readiness": health.get(
                "live_readiness"
            ),
            "ready": health.get("status") == "PASSED",
            "acceptance": health.get("acceptance"),
            "artifact_hash": health.get(
                "artifact_hash"
            ),
        }
    except Exception as exc:
        crypto_reference_health = {
            "status": "ERROR",
            "ready": False,
            "error": (
                f"{type(exc).__name__}:"
                f"{str(exc)[:500]}"
            ),
        }

    execution = preflight(args.live)
    live_ready = bool(
        execution.get("live_canary_ready") is True
    )
    blockers = []
    if not crypto_modules.get("ready"):
        blockers.append(
            "CRYPTO_CANONICAL_LIBRARY_NOT_READY"
        )
    if not references.get("ready"):
        blockers.append("REFERENCE_STACK_PARTIAL")
    if not selector.get("ready"):
        blockers.append(
            "PROSPECTIVE_SELECTOR_COLLECTING"
        )
    if not live_ready:
        blockers.extend(
            str(value)
            for value in execution.get("blockers") or []
        )

    paper_ready = bool(
        crypto_modules.get("ready")
        and references.get("ready")
    )
    payload = {
        "schema_version": (
            "crypto_ai_swing_full_stack_readiness_v1"
        ),
        "status": (
            "READY_FOR_PAPER_RESEARCH"
            if paper_ready
            else "ACTION_REQUIRED"
        ),
        "code_ready": bool(
            crypto_modules.get("ready")
        ),
        "research_stack_ready": bool(
            references.get("ready")
        ),
        "prospective_selector_ready": bool(
            selector.get("ready")
        ),
        "live_execution_ready": live_ready,
        "live_network_checked": bool(args.live),
        "crypto_canonical_library": crypto_modules,
        "crypto_native_reference_integration": (
            crypto_reference_health
        ),
        "reference_stack": references,
        "prospective_selector": selector,
        "execution_preflight": execution,
        "blockers": list(dict.fromkeys(blockers)),
        "orders_generated": 0,
        "orders_submitted": 0,
        "automatic_live_promotion": False,
        "autoscale_authorized": False,
    }
    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0 if paper_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
