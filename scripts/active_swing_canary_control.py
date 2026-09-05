from __future__ import annotations

import argparse
import json
from pathlib import Path

from crypto_ai_swing.execution.crypto_authority import CryptoAuthorityAdapter
from crypto_ai_swing.settings import Settings
from crypto_ai_swing.universe.runtime import UniverseManager


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manual control for the active-swing execution-validation canary."
    )
    parser.add_argument("action", choices=("status", "approve", "deactivate"))
    parser.add_argument("--markets", default="")
    parser.add_argument("--approval", default="")
    args = parser.parse_args()

    settings = Settings.load(Path.cwd())
    adapter = CryptoAuthorityAdapter(settings.crypto_repo_root)
    active = dict(settings.proactive.get("active_swing", {}) or {})
    canary = dict(active.get("execution_validation_canary", {}) or {})

    if args.action == "status":
        payload = {
            "style": "ACTIVE_SWING",
            "high_frequency_trading": False,
            "signal_timeframe": settings.proactive.get(
                "primary_signal_timeframe", "1h"
            ),
            "execution_timeframe": settings.proactive.get(
                "execution_timeframe", "15m"
            ),
            "execution_validation_canary": canary,
            "canonical_authority": adapter.authority_status(),
            "alpha_evidence_authorized": False,
            "automatic_live_promotion": False,
            "autoscale_authorized": False,
        }
    elif args.action == "deactivate":
        payload = adapter.deactivate()
    else:
        if not bool(canary.get("enabled", False)):
            raise SystemExit("EXECUTION_VALIDATION_CANARY_DISABLED")
        if not args.approval:
            raise SystemExit("--approval is required for approve")
        markets = [
            value.strip().upper()
            for value in args.markets.split(",")
            if value.strip()
        ]
        if not markets:
            markets = list(UniverseManager(settings).current()["markets"])
        payload = adapter.approve(
            markets=markets,
            approval=args.approval,
        )
        payload = {
            **dict(payload),
            "scope": "EXECUTION_VALIDATION_ONLY",
            "maximum_order_eur": float(canary.get("maximum_order_eur", 10.0)),
            "alpha_evidence_authorized": False,
            "automatic_live_promotion": False,
            "autoscale_authorized": False,
        }

    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
