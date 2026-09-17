#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crypto_ai_swing.production.dataflow_audit import Round41DataFlowAudit
from crypto_ai_swing.settings import Settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument(
        "--markets",
        default="BTC-EUR,ETH-EUR,SOL-EUR",
    )
    parser.add_argument(
        "--timeframes",
        default="15m,1h,4h",
    )
    parser.add_argument("--require-trained-agents", action="store_true")
    args = parser.parse_args()

    settings = Settings.load(Path.cwd())
    payload = Round41DataFlowAudit(settings).run(
        markets=[
            value.strip().upper()
            for value in args.markets.split(",")
            if value.strip()
        ],
        timeframes=[
            value.strip()
            for value in args.timeframes.split(",")
            if value.strip()
        ],
        network=bool(args.network),
        require_trained_agents=bool(args.require_trained_agents),
        deep_context=bool(args.deep),
    )
    output = (
        settings.project_root
        / "output/crypto_ai_swing/round41/dataflow_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(f"OUTPUT={output}")
    return 0 if payload.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
