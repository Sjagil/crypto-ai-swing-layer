from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from crypto_ai_swing.agents.edge_manager import ResearchEdgeManager
from crypto_ai_swing.research.swing_geometry import (
    SwingGeometryEngine,
    asymmetric_swing_quality_pass,
    fit_shrinkage_geometry,
    return_quality,
    score_geometry,
)


def test_shrinkage_covariance_is_psd_and_fisher_separates():
    rng = np.random.default_rng(26)
    latent = rng.normal(size=(240, 3))
    mix = rng.normal(size=(3, 8))
    x = latent @ mix + rng.normal(scale=0.25, size=(240, 8))
    signal = 1.2 * x[:, 0] - 0.8 * x[:, 1] + 0.35 * x[:, 3]
    y = (signal + rng.normal(scale=0.8, size=len(x)) > 0.0).astype(int)
    model = fit_shrinkage_geometry(
        x,
        y,
        feature_names=[f"f{i}" for i in range(x.shape[1])],
        minimum_feature_coverage=0.5,
        bootstrap_draws=80,
        seed=26,
    )
    assert model["status"] == "READY"
    eig = np.asarray(model["eigenvalues"], dtype=float)
    assert np.all(eig > 0.0)
    assert 1.0 <= model["effective_rank"] <= x.shape[1]
    assert model["rank_90pct_variance"] <= x.shape[1]
    assert model["fisher_separation"] > 0.0
    assert model["linear_algebra_contract"]["covariance"] == "LEDOIT_WOLF_SHRINKAGE"

    scores = []
    for row in x:
        scores.append(
            score_geometry(
                model,
                {f"f{i}": float(row[i]) for i in range(x.shape[1])},
            )["fisher_probability_like"]
        )
    scores = np.asarray(scores)
    assert float(scores[y > 0].mean()) > float(scores[y <= 0].mean())


def test_mahalanobis_flags_far_out_of_distribution_vector():
    rng = np.random.default_rng(2602)
    x = rng.normal(size=(180, 5))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    model = fit_shrinkage_geometry(
        x,
        y,
        feature_names=[f"f{i}" for i in range(5)],
        bootstrap_draws=40,
    )
    scored = score_geometry(
        model,
        {f"f{i}": 15.0 for i in range(5)},
    )
    assert scored["out_of_distribution"] is True
    assert scored["mahalanobis_distance"] > scored["ood_threshold"]


def test_cost_tolerance_is_relative_not_ignored():
    q = return_quality(
        [240.0, 180.0, -80.0, -60.0],
        [300.0, 240.0, -20.0, 0.0],
        [60.0, 60.0, 60.0, 60.0],
    )
    assert abs(q["cost_fraction_of_mean_winner"] - (60.0 / 270.0)) < 1e-12
    assert q["cost_fraction_of_mean_winner"] < 0.35
    weak = return_quality(
        [40.0, 20.0, -80.0],
        [100.0, 80.0, -20.0],
        [60.0, 60.0, 60.0],
    )
    assert weak["cost_fraction_of_mean_winner"] > 0.35


def test_asymmetric_swing_can_pass_with_40_percent_win_rate():
    values = np.asarray([300.0] * 4 + [-100.0] * 6)
    summary = return_quality(values)["normal_net"]
    assert summary["positive_fraction"] == 0.4
    assert summary["profit_factor"] == 2.0
    assert summary["payoff_ratio"] == 3.0
    assert asymmetric_swing_quality_pass(summary)


def test_delayed_label_horizon_selection_never_uses_current_future_label():
    engine = SwingGeometryEngine.__new__(SwingGeometryEngine)
    engine.swing_horizons = (24,)
    engine.minimum_horizon_history = 20
    engine.maximum_cost_fraction = 0.35
    engine.minimum_total = 999
    engine.minimum_oos = 999
    engine.minimum_positive_fraction = 0.35
    engine.minimum_profit_factor = 1.2
    engine.minimum_payoff_ratio = 1.1
    engine.minimum_probability_positive = 0.95
    engine.bootstrap_draws = 1000

    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(50):
        observed = start + timedelta(days=i)
        matured = observed + timedelta(hours=24, minutes=5)
        rows.append({
            "observation_id": f"o{i}",
            "observed_at": observed.isoformat(),
            "observed_dt": observed,
            "market": "BTC-EUR",
            "context": {},
            "horizon_hours": 24,
            "matured_at": matured.isoformat(),
            "matured_dt": matured,
            "gross_return_bps": 180.0 if i % 3 else 20.0,
            "normal_cost_bps": 60.0,
            "stressed_cost_bps": 85.0,
            "normal_net_bps": 120.0 if i % 3 else -40.0,
            "stressed_net_bps": 95.0 if i % 3 else -65.0,
            "mfe_bps": 220.0,
            "mae_bps": -70.0,
        })
    result = engine._sequential_oos(rows)
    assert result["oos_selected"] > 0
    assert all(row["lookahead_check"] for row in result["recent_audit"])
    first = result["recent_audit"][0]
    assert first["history_observations"] >= 20


def test_round22_new_style_collaborators_are_optional(tmp_path: Path):
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
    context = json.dumps({
        "mtf_challenger": {"score": 0.7, "cmc_regime_score": 0.6},
        "agents": {"alpha_probability": 0.6, "execution_score": 0.7},
        "universe_screen": {"execution_adjusted_score": 0.5},
    })
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
    assert result["net_edge_calibration"]["qualified"] is False
    assert result["swing_geometry"]["qualified"] is False
