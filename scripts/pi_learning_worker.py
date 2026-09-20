#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crypto_ai_swing.orchestration.pi_learning_worker import PiUnifiedLearningWorker
from crypto_ai_swing.settings import Settings


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("shadow","paper","live"), default="shadow")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--factory-exact", action="store_true", default=False)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    worker=PiUnifiedLearningWorker(Settings.load(root), mode=args.mode, factory_exact=args.factory_exact)
    if args.smoke:
        payload = worker.smoke()
        print(json.dumps(payload, indent=2, default=str))
        return 0 if payload.get("status") == "PASS" else 2
    if args.once:
        print(json.dumps(worker.run_once(), indent=2, default=str)); return 0
    worker.run_forever(poll_seconds=args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
