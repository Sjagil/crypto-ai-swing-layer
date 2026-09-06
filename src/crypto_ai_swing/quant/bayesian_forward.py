from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def _posterior(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"status": "NO_DATA", "observations": 0}

    rng = np.random.default_rng(seed)
    wins = int(np.sum(values > 0.0))
    losses = int(len(values) - wins)
    positive = rng.beta(1 + wins, 1 + losses, size=draws)
    if len(values) == 1:
        means = np.full(draws, values[0])
    else:
        means = rng.dirichlet(np.ones(len(values)), size=draws) @ values

    return {
        "status": "READY",
        "observations": int(len(values)),
        "wins": wins,
        "sample_mean_return_bps": float(values.mean()),
        "positive_rate_posterior_mean": float(positive.mean()),
        "positive_rate_p05": float(np.quantile(positive, 0.05)),
        "positive_rate_p95": float(np.quantile(positive, 0.95)),
        "mean_return_p05_bps": float(np.quantile(means, 0.05)),
        "mean_return_p50_bps": float(np.quantile(means, 0.50)),
        "mean_return_p95_bps": float(np.quantile(means, 0.95)),
        "probability_mean_return_positive": float(np.mean(means > 0.0)),
    }


def build_bayesian_forward_snapshot(
    database_path: Path,
    *,
    horizon_hours: int = 4,
    draws: int = 5000,
) -> dict[str, Any]:
    path = Path(database_path)
    if not path.is_file():
        return {
            "status": "NO_LEDGER",
            "authority": "RESEARCH_ONLY",
            "live_decision_influence": False,
        }

    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            """
            SELECT s.market, o.return_bps
            FROM signal_observations s
            JOIN forward_outcomes_v2 o
              ON s.observation_id=o.observation_id
            WHERE s.side='BUY'
              AND s.blocked=0
              AND o.horizon_hours=?
            ORDER BY o.matured_at
            """,
            (int(horizon_hours),),
        ).fetchall()
    finally:
        conn.close()

    grouped: dict[str, list[float]] = {}
    all_values: list[float] = []
    for market, value in rows:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        grouped.setdefault(str(market), []).append(value)
        all_values.append(value)

    aggregate = _posterior(np.asarray(all_values), draws, 17)
    by_market = {
        market: _posterior(np.asarray(values), draws, 18 + i)
        for i, (market, values) in enumerate(sorted(grouped.items()))
    }
    return {
        "schema_version": "bayesian_forward_edge_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": (
            "READY"
            if int(aggregate.get("observations") or 0) >= 30
            else "COLLECTING"
        ),
        "horizon_hours": int(horizon_hours),
        "draws": int(draws),
        "aggregate": aggregate,
        "markets": by_market,
        "authority": "RESEARCH_ONLY",
        "live_decision_influence": False,
        "automatic_live_promotion": False,
        "orders_submitted": 0,
    }


def persist(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
