#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crypto_ai_swing.orchestration.health import runtime_health


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--heartbeat",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--max-age",
        type=float,
        default=900.0,
    )
    args = parser.parse_args()
    result = runtime_health(
        args.heartbeat,
        max_age=args.max_age,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
