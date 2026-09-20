#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crypto_ai_swing.production.live_lifecycle import (
    LiveLifecycleCoordinator,
)
from crypto_ai_swing.settings import Settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--markets",
        default="BTC-EUR,ETH-EUR,SOL-EUR,LINK-EUR",
    )
    args = parser.parse_args()
    markets = [
        item.strip().upper()
        for item in args.markets.split(",")
        if item.strip()
    ]
    settings = Settings.load(Path.cwd())
    payload = LiveLifecycleCoordinator(settings).audit(markets)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    # Audit itself is successful even when live authority is intentionally off.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
