#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_ai_swing.bridge.data_source_fabric import CanonicalDataSourceFabric
from crypto_ai_swing.bridge.runtime import activate_canonical_crypto
from crypto_ai_swing.settings import Settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collect-seconds",
        type=float,
        default=0.0,
        help="Run the canonical public/read-only multi-source collector for N seconds.",
    )
    parser.add_argument(
        "--skip-intelligence",
        action="store_true",
        help="Do not run the canonical web/RSS intelligence pipeline during collection.",
    )
    parser.add_argument(
        "--strict-config",
        action="store_true",
        help="Also require every keyed canonical provider configured in the environment.",
    )
    args = parser.parse_args()

    settings = Settings.load(ROOT)
    bridge = activate_canonical_crypto(settings)
    fabric = CanonicalDataSourceFabric(bridge, swing_root=ROOT)

    collection = None
    collection_error = None
    if args.collect_seconds > 0:
        try:
            collection = fabric.collect_public_once_sync(
                duration_seconds=args.collect_seconds,
                include_intelligence=not args.skip_intelligence,
            )
        except Exception as exc:
            collection_error = f"{type(exc).__name__}: {str(exc)[:500]}"

    audit = fabric.audit()
    keyed = [
        row
        for row in audit["providers"]
        if row["env_aliases"]
        and row["canonical_settings_bound"]
    ]
    missing_keyed = sorted(
        row["source_id"] for row in keyed if not row["configured"]
    )
    strict_ready = not missing_keyed if args.strict_config else True
    status = (
        "READY"
        if audit["code_ready"]
        and not audit["blockers"]
        and collection_error is None
        and strict_ready
        else "ACTION_REQUIRED"
    )
    payload = {
        **audit,
        "status": status,
        "strict_config": bool(args.strict_config),
        "missing_keyed_canonical_providers": missing_keyed,
        "collection_requested": args.collect_seconds > 0,
        "collection": collection,
        "collection_error": collection_error,
        "orders_generated": 0,
        "orders_submitted": 0,
        "private_exchange_requests": 0,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if status == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
