#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crypto_ai_swing.agents.performance_governor import (
    PerformanceGovernor,
)
from crypto_ai_swing.agents.manager import AgentManager
from crypto_ai_swing.research.autonomous_strategy_director import (
    AutonomousStrategyDirector,
)
from crypto_ai_swing.settings import Settings
from crypto_ai_swing.universe.runtime import UniverseManager


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    settings = Settings.load(Path.cwd())
    markets = list(UniverseManager(settings).current().get("markets", []))
    manager = AgentManager(settings, mode="shadow")
    governor = PerformanceGovernor(settings)
    priorities = governor.evaluate(
        agent_status=manager.status()
    ).get("priorities", [])
    payload = AutonomousStrategyDirector(
        settings, mode="shadow"
    ).cycle(
        markets=[str(x).upper() for x in markets],
        priorities=list(priorities),
        force=bool(args.force),
        force_exact=bool(args.exact),
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload.get("status") != "DEGRADED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
