
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from crypto_ai_swing.agents.prospective_context import _dataset_hash, _purged_split
from crypto_ai_swing.intelligence.comprehensive import (
    L3_STATUS,
    PROVIDER_REQUIRED,
    TIMEFRAMES,
    _availability,
    _hash,
)
from crypto_ai_swing.research.feature_attribution import Round44FeatureAttribution
from crypto_ai_swing.research.live_readiness import Round44LiveReadiness


def _settings(tmp_path: Path):
    return SimpleNamespace(
        project_root=tmp_path,
        crypto_repo_root=tmp_path / "crypto",
        agents={"rl": {"enabled": False}},
    )


def test_round44_market_data_contract_is_real_l2_not_fake_l3():
    assert TIMEFRAMES == ("15m", "1h", "2h", "4h", "1d", "1w")
    assert L3_STATUS == "L3_UNSUPPORTED_BY_EXECUTION_VENUE"
    unavailable = _availability(
        ready=False,
        source="provider",
        reason="missing",
        provider_required=True,
    )
    assert unavailable["status"] == PROVIDER_REQUIRED
    assert unavailable["synthetic_data_used"] is False


def test_component_hash_is_deterministic():
    left = {"technical": {"rsi": 50.0}, "l2": {"ofi": 0.2}}
    right = {"l2": {"ofi": 0.2}, "technical": {"rsi": 50.0}}
    assert _hash(left) == _hash(right)


def test_purged_context_split_keeps_time_gaps():
    times = pd.date_range("2026-01-01", periods=120, freq="15min", tz="UTC")
    rows = []
    for index, ts in enumerate(times):
        rows.append(
            {
                "_observed_at": ts.isoformat(),
                "_observation_id": f"o{index}",
                "_market": "BTC-EUR" if index % 2 == 0 else "ETH-EUR",
                "_return_bps": float(index % 7 - 3),
                "_mae_bps": -1.0,
                "_mfe_bps": 2.0,
                "f1": float(index),
            }
        )
    frame = pd.DataFrame(rows)
    split = _purged_split(frame, purge_buckets=4, embargo_buckets=4)
    train_max = pd.to_datetime(split.train["_observed_at"], utc=True).max()
    validation_min = pd.to_datetime(split.validation["_observed_at"], utc=True).min()
    validation_max = pd.to_datetime(split.validation["_observed_at"], utc=True).max()
    test_min = pd.to_datetime(split.test["_observed_at"], utc=True).min()
    assert validation_min > train_max
    assert test_min > validation_max
    assert _dataset_hash(frame, 4) == _dataset_hash(frame.copy(), 4)


def test_multi_horizon_attribution_uses_only_matured_forward_rows(tmp_path):
    database = tmp_path / "forward.sqlite"
    conn = sqlite3.connect(database)
    conn.executescript(
        """
        CREATE TABLE signal_observations(
            observation_id TEXT PRIMARY KEY, observed_at TEXT, market TEXT,
            side TEXT, blocked INTEGER, context TEXT
        );
        CREATE TABLE forward_outcomes_v2(
            observation_id TEXT, horizon_hours INTEGER, return_bps REAL,
            mfe_bps REAL, mae_bps REAL
        );
        """
    )
    for index in range(80):
        observation_id = f"o{index}"
        vector = {"x": float(index), "noise": float((index * 17) % 11)}
        context = json.dumps({"round44": {"learning_vector": vector}})
        conn.execute(
            "INSERT INTO signal_observations VALUES (?,?,?,?,?,?)",
            (
                observation_id,
                f"2026-01-01T{index % 24:02d}:00:00+00:00",
                f"M{index % 8}-EUR",
                "BUY" if index % 3 else "NO_TRADE",
                0,
                context,
            ),
        )
        for horizon in (1, 4, 24, 72, 168):
            ret = float(index - 40) * (1.0 + horizon / 200.0)
            conn.execute(
                "INSERT INTO forward_outcomes_v2 VALUES (?,?,?,?,?)",
                (observation_id, horizon, ret, max(0.0, ret), min(0.0, ret)),
            )
    conn.commit()
    conn.close()
    report = Round44FeatureAttribution(_settings(tmp_path)).evaluate(
        database, minimum_observations=40, maximum_features=10
    )
    assert report["status"] == "READY"
    assert set(report["ready_horizons"]) == {1, 4, 24, 72, 168}
    assert report["long_horizons_ready"] is True
    assert "x" in report["stable_shortlist_features"]
    assert report["automatic_live_authority"] is False


def test_readiness_never_grants_execution_authority(tmp_path):
    settings = _settings(tmp_path)
    report = Round44LiveReadiness(settings).evaluate(tmp_path / "missing.sqlite")
    assert report["stage"] == "RESEARCH"
    assert report["execution_authority"] == "NONE"
    assert report["automatic_live_authority"] is False
    assert report["automatic_live_promotion"] is False
