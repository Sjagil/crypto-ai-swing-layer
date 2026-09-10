#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crypto_ai_swing.orchestration.unified_runtime import (
    UnifiedAutonomyRuntime,
)
from crypto_ai_swing.settings import Settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("shadow", "paper", "live"),
        default="shadow",
    )
    parser.add_argument("--forever", action="store_true")
    args = parser.parse_args()

    settings = Settings.load(Path.cwd())
    runtime = UnifiedAutonomyRuntime(settings, mode=args.mode)
    try:
        if args.forever:
            runtime.run_forever()
            return 0
        payload = runtime.run_once(one_shot=True)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if not payload.get("errors") else 2
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
