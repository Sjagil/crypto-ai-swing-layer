from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace


from crypto_ai_swing.agents.edge_manager import ResearchEdgeManager
from crypto_ai_swing.orchestration.mtf_challenger import evaluate_mtf_challenger


class FakeTF:
    def __init__(self):
        state = {
            "score": 0.5,
            "trend": 0.6,
            "momentum": 0.4,
            "breakout": 0.3,
            "rsi": 55.0,
        }
        self.states = {
            "15m": dict(state),
            "1h": dict(state),
            "2h": dict(state),
            "4h": dict(state),
            "1d": dict(state),
            "1w": dict(state),
        }
        self.entry_blocked = False


def test_mtf_challenger_is_research_only():
    result = evaluate_mtf_challenger(
        FakeTF(),
        microstructure={
            "spread_bps": 5,
            "book_imbalance": 0.2,
            "cvd_ratio": 0.1,
        },
        cmc_asset={
            "percent_change_24h": 4,
            "percent_change_7d": 8,
        },
        cmc_context={
            "breadth_top250": {"positive_fraction_24h": 0.65},
            "fear_and_greed": {"value": 60},
        },
    )
    assert 0 <= result.score <= 1
    assert result.authority == "RESEARCH_ONLY"
    assert result.live_decision_influence is False


def test_edge_manager_collects_until_enough_observations(tmp_path: Path):
    db = tmp_path / "forward.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE signal_observations(
            observation_id TEXT PRIMARY KEY,
            market TEXT,
            side TEXT,
            blocked INTEGER,
            context TEXT
        );
        CREATE TABLE forward_outcomes_v2(
            observation_id TEXT,
            horizon_hours INTEGER,
            matured_at TEXT,
            return_bps REAL
        );
        """
    )
    context = json.dumps(
        {
            "mtf_challenger": {"score": 0.7, "cmc_regime_score": 0.6},
            "agents": {"alpha_probability": 0.6, "execution_score": 0.7},
            "universe_screen": {"execution_adjusted_score": 0.5},
        }
    )
    for i in range(10):
        oid = f"x{i}"
        conn.execute(
            "INSERT INTO signal_observations VALUES (?,?,?,?,?)",
            (oid, "BTC-EUR", "BUY", 0, context),
        )
        conn.execute(
            "INSERT INTO forward_outcomes_v2 VALUES (?,?,?,?)",
            (oid, 4, f"2026-01-01T{i:02d}:00:00Z", 100.0),
        )
    conn.commit()
    conn.close()

    settings = SimpleNamespace(
        crypto_repo_root=tmp_path,
        project_root=tmp_path,
        execution={"costs": {"fee_bps_per_side": 25, "base_slippage_bps": 2}},
        agents={"edge_manager": {"minimum_observations": 80}},
    )
    manager = ResearchEdgeManager.__new__(ResearchEdgeManager)
    manager.settings = settings
    manager.mode = "shadow"
    manager.root = tmp_path / "meta"
    manager.latest_path = manager.root / "latest.json"
    manager.refresh_seconds = 900
    manager.minimum_observations = 80
    manager.minimum_holdout_selected = 30
    manager._last_refresh = 0
    manager._policy = {}
    manager.bridge = SimpleNamespace()
    result = manager.refresh_policy(db, force=True)
    assert result["status"] == "COLLECTING"
    assert result["shadow_influence"] is False
    assert result["live_decision_influence"] is False
