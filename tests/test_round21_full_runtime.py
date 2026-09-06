from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from crypto_ai_swing.monitoring.account_state import collect_account_state
from crypto_ai_swing.quant.bayesian_forward import build_bayesian_forward_snapshot


def test_account_observer_does_not_fake_wallet(tmp_path: Path):
    class Adapter:
        def __init__(self, _root): pass
        def authority_status(self): return {"active": True, "state_status": "READY"}
        def portfolio(self): return {"positions": {}}
        def account_snapshot(self, _markets): raise RuntimeError("blocked")

    settings = SimpleNamespace(crypto_repo_root=tmp_path)
    result = collect_account_state(
        settings, ["BTC-EUR"], adapter_factory=Adapter
    )
    assert result["status"] == "BLOCKED"
    assert result["eur_available"] is None
    assert result["estimated_equity_eur"] is None


def test_bayesian_forward_uses_matured_unblocked_buys(tmp_path: Path):
    db = tmp_path / "forward.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE signal_observations(
            observation_id TEXT PRIMARY KEY,
            market TEXT,
            side TEXT,
            blocked INTEGER
        );
        CREATE TABLE forward_outcomes_v2(
            observation_id TEXT,
            horizon_hours INTEGER,
            matured_at TEXT,
            return_bps REAL
        );
        """
    )
    for i, value in enumerate((50.0, 20.0, -10.0, 30.0)):
        oid = f"x{i}"
        conn.execute(
            "INSERT INTO signal_observations VALUES (?,?,?,?)",
            (oid, "BTC-EUR", "BUY", 0),
        )
        conn.execute(
            "INSERT INTO forward_outcomes_v2 VALUES (?,?,?,?)",
            (oid, 4, f"2026-01-01T0{i}:00:00Z", value),
        )
    conn.commit()
    conn.close()
    result = build_bayesian_forward_snapshot(db, draws=500)
    assert result["aggregate"]["observations"] == 4
    assert result["live_decision_influence"] is False
