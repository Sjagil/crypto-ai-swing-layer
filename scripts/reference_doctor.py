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
    parser.add_argument(
        "--venv-root",
        action="append",
        default=[],
        type=Path,
        help=(
            "Additional shared .venvs root. May be repeated. "
            "The parent workspace .venvs is auto-discovered."
        ),
    )
    parser.add_argument(
        "--repo-root",
        action="append",
        default=[],
        type=Path,
        help=(
            "Additional reference source root. May be repeated. "
            "The parent workspace is auto-discovered."
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    payload = reference_environment_status(
        root,
        include_disabled=not args.enabled_only,
        venv_roots=args.venv_root or None,
        repo_roots=args.repo_root or None,
    )
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
