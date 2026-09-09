from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from crypto_ai_swing.production.recovery import ExecutionRecoveryJournal

class CanaryCertification:
    SCHEMA = "crypto_ai_swing_live_canary_certification_v1"

    def __init__(self, settings) -> None:
        self.settings = settings
        cfg_path = Path(settings.project_root) / "config/production.yaml"
        try:
            cfg = yaml.safe_load(
                cfg_path.read_text(encoding="utf-8")
            ) or {}
        except Exception:
            cfg = {}
        self.cfg = dict(cfg.get("certification", {}) or {})
        self.journal = ExecutionRecoveryJournal(
            Path(settings.project_root)
            / "output/crypto_ai_swing/production/execution_recovery.sqlite"
        )

    def status(self) -> dict[str, Any]:
        rows = list(reversed(self.journal.recent(limit=1000)))
        reconciled = [
            row for row in rows if row.get("state") == "RECONCILED"
        ]
        buys = defaultdict(int)
        sells = defaultdict(int)
        for row in reconciled:
            market = str(row.get("market") or "")
            side = str(row.get("side") or "").upper()
            if side == "BUY":
                buys[market] += 1
            elif side == "SELL":
                sells[market] += 1
        roundtrips = sum(
            min(buys[market], sells[market])
            for market in set(buys) | set(sells)
        )
        minimum = int(
            self.cfg.get("minimum_real_roundtrips", 3)
        )
        unresolved = self.journal.unresolved()
        certified = roundtrips >= minimum and not unresolved
        return {
            "schema_version": self.SCHEMA,
            "status": "CERTIFIED" if certified else "COLLECTING",
            "reconciled_operations": len(reconciled),
            "reconciled_buys": int(sum(buys.values())),
            "reconciled_sells": int(sum(sells.values())),
            "complete_roundtrips": int(roundtrips),
            "minimum_real_roundtrips": minimum,
            "unresolved_operations": len(unresolved),
            "execution_certified": certified,
            "strategy_profitability_certified": False,
            "scaling_authorized": False,
        }
