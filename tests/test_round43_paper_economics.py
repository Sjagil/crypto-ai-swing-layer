from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

from crypto_ai_swing.research.paper_economics import (
    PaperEconomicEvaluator,
)


def _settings(tmp_path):
    crypto = tmp_path / "crypto"
    crypto.mkdir()
    return SimpleNamespace(
        project_root=tmp_path,
        crypto_repo_root=crypto,
        proactive={"shadow_equity_eur": 10000},
        autonomy={
            "forward_evidence": {
                "path": "output/forward.sqlite",
            },
            "paper_economics": {
                "minimum_closed_trades_for_diagnostics": 2,
                "minimum_closed_trades_for_qualification": 2,
                "minimum_strategy_closed_trades": 2,
                "minimum_profit_factor": 1.05,
                "minimum_payoff_ratio": 1.0,
                "maximum_realized_drawdown_fraction": 0.20,
                "minimum_complete_long_horizon_observations": 1,
            },
        },
    )


def _paper_database(tmp_path):
    path = (
        tmp_path
        / "output/crypto_ai_swing/modes/shadow/proactive"
        / "paper_ledger.sqlite"
    )
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                side TEXT NOT NULL,
                market TEXT NOT NULL,
                quantity TEXT NOT NULL,
                price TEXT NOT NULL,
                gross_eur TEXT NOT NULL,
                fee_eur TEXT NOT NULL,
                realized_pnl_eur TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO meta VALUES("
            "'starting_equity_eur','10000')"
        )
        rows = [
            (
                "b1",
                "2026-01-01T00:00:00+00:00",
                "BUY",
                "BTC-EUR",
                "1",
                "100",
                "100",
                "0.25",
                "0",
                json.dumps(
                    {
                        "strategy": "swing_a",
                        "stop_pct": 0.02,
                        "estimated_round_trip_cost_bps": "50",
                    }
                ),
            ),
            (
                "s1",
                "2026-01-02T00:00:00+00:00",
                "SELL",
                "BTC-EUR",
                "1",
                "106",
                "106",
                "0.265",
                "5.485",
                json.dumps({"reason": "TAKE_PROFIT"}),
            ),
            (
                "b2",
                "2026-01-03T00:00:00+00:00",
                "BUY",
                "ETH-EUR",
                "1",
                "100",
                "100",
                "0.25",
                "0",
                json.dumps(
                    {
                        "strategy": "swing_a",
                        "stop_pct": 0.02,
                        "estimated_round_trip_cost_bps": "50",
                    }
                ),
            ),
            (
                "s2",
                "2026-01-04T00:00:00+00:00",
                "SELL",
                "ETH-EUR",
                "1",
                "102",
                "102",
                "0.255",
                "1.495",
                json.dumps({"reason": "TRAILING_STOP"}),
            ),
        ]
        conn.executemany(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
    return path


def _forward_database(tmp_path):
    path = tmp_path / "output/forward.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE signal_observations (
                observation_id TEXT PRIMARY KEY,
                market TEXT NOT NULL,
                side TEXT NOT NULL,
                blocked INTEGER NOT NULL
            );
            CREATE TABLE forward_outcomes_v2 (
                observation_id TEXT NOT NULL,
                horizon_hours INTEGER NOT NULL,
                return_bps REAL NOT NULL,
                mfe_bps REAL NOT NULL,
                mae_bps REAL NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO signal_observations "
            "VALUES('o1','BTC-EUR','BUY',0)"
        )
        for horizon in (4, 24, 72, 168):
            conn.execute(
                "INSERT INTO forward_outcomes_v2 "
                "VALUES(?,?,?,?,?)",
                ("o1", horizon, 200.0, 300.0, -80.0),
            )
    return path


def test_realized_paper_economics_and_long_horizon_gate(
    tmp_path,
):
    _paper_database(tmp_path)
    _forward_database(tmp_path)
    report = PaperEconomicEvaluator(
        _settings(tmp_path),
        mode="shadow",
        crypto=None,
    ).refresh()

    assert report["overall"]["closed_trades"] == 2
    assert report["overall"]["cost_adjusted_pnl_eur"] > 0
    assert report["paper_economic_leader"]["strategy"] == "swing_a"
    assert (
        report["forward_swing_horizons"][
            "complete_long_horizon_observations"
        ]
        == 1
    )
    assert report["economic_qualification"] is True
    assert report["automatic_live_authority"] is False
    assert report["orders_submitted"] == 0
