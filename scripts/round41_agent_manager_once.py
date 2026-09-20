#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crypto_ai_swing.agents.manager import AgentManager
from crypto_ai_swing.settings import Settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument(
        "--markets",
        default="",
        help="optional comma-separated market override",
    )
    args = parser.parse_args()

    settings = Settings.load(Path.cwd())
    markets = [
        value.strip().upper()
        for value in args.markets.split(",")
        if value.strip()
    ]
    forward_cfg = dict(
        settings.autonomy.get("forward_evidence", {}) or {}
    )
    ledger = settings.project_root / forward_cfg.get(
        "path",
        "output/crypto_ai_swing/forward/forward.sqlite",
    )
    manager = AgentManager(settings, mode="shadow")
    payload = manager.cycle(
        markets=markets or None,
        forward_database_path=ledger,
        force_train=bool(args.force_train),
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload.get("status") != "DEGRADED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
