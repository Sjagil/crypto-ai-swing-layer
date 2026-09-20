#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crypto_ai_swing.agents.chief import ChiefAgent
from crypto_ai_swing.settings import Settings
from crypto_ai_swing.universe.runtime import UniverseManager


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("shadow", "paper", "live"),
        default="shadow",
    )
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-research", action="store_true")
    parser.add_argument("--force-exact-research", action="store_true")
    parser.add_argument("--browser", action="store_true")
    args = parser.parse_args()

    settings = Settings.load(Path.cwd())
    markets = [
        str(value).upper()
        for value in UniverseManager(settings).current().get(
            "markets", []
        )
        if str(value).strip()
    ]
    forward_cfg = dict(
        (settings.autonomy.get("forward_evidence", {}) or {})
    )
    ledger = settings.project_root / forward_cfg.get(
        "path",
        "output/crypto_ai_swing/forward/forward.sqlite",
    )
    payload = ChiefAgent(settings, mode=args.mode).cycle(
        markets=markets,
        forward_database_path=ledger,
        force_train=bool(args.force_train),
        force_research=bool(args.force_research),
        force_exact_research=bool(args.force_exact_research),
        browser_research=True if args.browser else None,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload.get("status") != "DEGRADED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
