#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.research.forward import ForwardEvidenceLedger
from crypto_ai_swing.settings import Settings


def main() -> int:
    settings = Settings.load()
    cfg = dict(
        (getattr(settings, "autonomy", {}) or {}).get("forward_evidence", {})
        or {}
    )
    rel = cfg.get("path", "output/crypto_ai_swing/forward/forward.sqlite")
    db = Path(settings.project_root) / rel
    execution_tf = str(
        cfg.get(
            "execution_timeframe",
            (getattr(settings, "proactive", {}) or {}).get(
                "execution_timeframe", "15m"
            ),
        )
    )
    horizons = tuple(
        int(x)
        for x in cfg.get("horizons_hours", (1, 4, 24))
        if int(x) > 0
    )

    if not db.is_file():
        raise SystemExit(f"Forward ledger missing: {db}")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = db.with_name(f"{db.name}.pre_v0231_{stamp}.bak")
    shutil.copy2(db, backup)

    ledger = ForwardEvidenceLedger(
        db,
        decision_bucket_minutes=int(cfg.get("decision_bucket_minutes", 15)),
    )
    before = ledger.outcome_status()
    markets = list(ledger.observation_markets())
    ledger.close()

    bridge = CryptoLibraryBridge(settings.crypto_repo_root)
    frames = bridge.ohlcv_many(
        markets,
        execution_tf,
        persist=False,
        concurrency=4,
    )

    conn = sqlite3.connect(db)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM forward_outcomes_v2")
        conn.commit()
    finally:
        conn.close()

    ledger = ForwardEvidenceLedger(
        db,
        decision_bucket_minutes=int(cfg.get("decision_bucket_minutes", 15)),
    )
    try:
        result = ledger.mature_from_frames(
            frames,
            horizons_hours=horizons,
            execution_timeframe=execution_tf,
        )
        after = ledger.outcome_status()
    finally:
        ledger.close()

    before_count = int(before.get("outcomes") or 0)
    after_count = int(after.get("outcomes") or 0)
    if after_count < before_count:
        shutil.copy2(backup, db)
        raise SystemExit(
            "REBUILD_INCOMPLETE: rebuilt outcome count "
            f"{after_count} < previous {before_count}. "
            f"Original database restored from {backup}"
        )

    payload = {
        "schema_version": "forward_outcome_v0231_repair_v1",
        "status": "REBUILT",
        "database": str(db),
        "backup": str(backup),
        "execution_timeframe": execution_tf,
        "horizons_hours": list(horizons),
        "before": before,
        "after": after,
        "maturation": result,
        "orders_submitted": 0,
    }
    out = (
        Path(settings.project_root)
        / "output/crypto_ai_swing/autonomy/forward_repair_v0231.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
