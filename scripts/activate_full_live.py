#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crypto-root",
        default=os.getenv("CRYPTO_REPO_PATH", "/home/pi/sjagil/crypto"),
    )
    parser.add_argument(
        "--approve",
        required=True,
        help='Must equal: I APPROVE FULL LIVE QUANT TRADING',
    )
    parser.add_argument(
        "--markets",
        default="",
        help="Optional comma-separated EUR spot authority universe",
    )
    args = parser.parse_args()

    root = Path(args.crypto_root).expanduser().resolve()
    sys.path.insert(0, str(root))
    os.environ["CRYPTO_FULL_LIVE"] = "YES"

    from config.settings import Settings
    from core.swing_layer_live import (
        approve_swing_layer_full_live,
        reconcile_swing_layer_live,
        swing_layer_authority_status,
    )

    settings = Settings.load(env_file=root / ".env", create_directories=True)
    symbols = (
        tuple(
            dict.fromkeys(
                value.strip().upper()
                for value in args.markets.split(",")
                if value.strip()
            )
        )
        if args.markets.strip()
        else tuple(str(x).upper() for x in settings.market_data.symbols)
    )
    eligible = tuple(
        market
        for market in symbols
        if settings.shariah.eligibility(market).status.value == "ALLOWED"
    )
    if not eligible:
        raise SystemExit("No canonical eligible EUR spot markets found")

    approval = approve_swing_layer_full_live(
        markets=eligible,
        approval=args.approve,
    )
    reconciliation = reconcile_swing_layer_live(eligible)
    authority = swing_layer_authority_status()

    payload = {
        "approval": approval,
        "eligible_markets": list(eligible),
        "reconciliation": reconciliation,
        "authority": authority,
    }
    print(json.dumps(payload, indent=2, default=str))

    if authority.get("active") is not True:
        return 4
    if str(authority.get("mode") or "").upper() != "FULL_LIVE":
        return 5
    if str(reconciliation.get("status") or "").upper() != "READY":
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
