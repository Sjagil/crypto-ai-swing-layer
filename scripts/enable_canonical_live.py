#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_ai_swing.bridge.runtime import activate_canonical_crypto
from crypto_ai_swing.settings import Settings as SwingSettings


async def run(approval: str) -> int:
    swing = SwingSettings.load(ROOT)
    bridge = activate_canonical_crypto(swing)
    settings_module = bridge.import_module("config.settings")
    live_module = bridge.import_module("core.autonomous_live")

    expected = str(live_module.APPROVAL_PHRASE)
    if approval.strip() != expected:
        raise PermissionError(
            "canonical autonomous-live approval phrase does not match"
        )

    crypto_settings = settings_module.Settings.load(create_directories=True)
    supervisor = live_module.AutonomousLiveSupervisor(crypto_settings)
    payload = await supervisor.enable(
        markets=supervisor.markets,
        approval=approval,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval", required=True)
    args = parser.parse_args()
    return asyncio.run(run(args.approval))


if __name__ == "__main__":
    raise SystemExit(main())
