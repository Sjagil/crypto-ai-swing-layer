from __future__ import annotations

import argparse
import json
from pathlib import Path

from crypto_ai_swing.bridge.reference_runtime import reference_environment_status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit reference repositories and isolated .venvs."
    )
    parser.add_argument(
        "--enabled-only",
        action="store_true",
        help="Only show references enabled in config/references.yaml.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    payload = reference_environment_status(
        root,
        include_disabled=not args.enabled_only,
    )
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
