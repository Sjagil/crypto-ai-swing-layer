#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_ai_swing.bridge.engine_contract import validate_engine_contract
from crypto_ai_swing.settings import Settings


def main() -> int:
    settings = Settings.load(ROOT)
    try:
        contract = validate_engine_contract(
            ROOT,
            settings=settings,
            require_clean=True,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema_version": "round36_1_engine_contract_audit_v1",
                    "status": "FAILED",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "schema_version": "round36_1_engine_contract_audit_v1",
                "status": "PASSED",
                "crypto_root": str(contract.root),
                "branch": contract.branch,
                "expected_head": contract.expected_head,
                "actual_head": contract.actual_head,
                "clean": contract.clean,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
