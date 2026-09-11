from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from crypto_ai_swing.research.evidence_maturation import (
    ProspectiveEvidenceMaturation,
)
from crypto_ai_swing.research.forward import (
    CANONICAL_OUTCOME_SCHEMA,
    ForwardEvidenceLedger,
)


def _settings(tmp_path):
    crypto = tmp_path / "crypto"
    crypto.mkdir()
    return SimpleNamespace(
        project_root=tmp_path,
        crypto_repo_root=crypto,
        autonomy={
            "forward_evidence": {
                "path": "output/forward.sqlite",
                "horizons_hours": [1, 4],
                "decision_bucket_minutes": 15,
                "execution_timeframe": "15m",
                "canary_readiness": {
                    "primary_horizon_hours": 4,
                    "minimum_unblocked_buy_outcomes": 2,
                    "minimum_distinct_markets": 1,
                    "minimum_observation_span_hours": 0,
                    "minimum_mean_return_bps": -10000,
                    "minimum_positive_return_rate": 0,
                },
            },
            "strategy_lab": {
                "minimum_discovery_observations": 2,
            },
            "entry_selector": {
                "horizons_hours": [1, 4],
                "minimum_train_observations": 1,
            },
            "evidence_maturation": {
                "minimum_complete_horizon_observations_for_optimization": 2,
            },
        },
    )


def test_round43_evidence_stage_uses_real_matured_rows(tmp_path):
    settings = _settings(tmp_path)
    path = tmp_path / "output/forward.sqlite"
    ledger = ForwardEvidenceLedger(path, decision_bucket_minutes=15)
    base = datetime.now(UTC) - timedelta(hours=12)

    for index in range(2):
        observed = base + timedelta(minutes=15 * index)
        payload = {
            "generated_at": observed.isoformat(),
            "mode": "shadow",
            "markets": ["BTC-EUR"],
            "signals": [
                {
                    "market": "BTC-EUR",
                    "side": "BUY",
                    "score": 0.8,
                    "edge_bps": 40.0,
                }
            ],
            "blocked": [],
            "market_context": {"BTC-EUR": {}},
        }
        ledger.append_cycle(
            payload,
            {"BTC-EUR": observed.isoformat()},
        )

    rows = ledger.conn.execute(
        """
        SELECT observation_id, observed_at
        FROM signal_observations
        ORDER BY observed_at
        """
    ).fetchall()
    for number, (observation_id, observed_at) in enumerate(rows):
        for horizon in (1, 4):
            reference = 100.0
            outcome = 101.0 + number
            ledger.conn.execute(
                """
                INSERT INTO forward_outcomes_v2
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    observation_id,
                    horizon,
                    datetime.now(UTC).isoformat(),
                    observed_at,
                    observed_at,
                    reference,
                    outcome,
                    (outcome / reference - 1.0) * 10000.0,
                    120.0,
                    -40.0,
                    CANONICAL_OUTCOME_SCHEMA,
                ),
            )
    ledger.conn.commit()
    ledger.close()

    report = ProspectiveEvidenceMaturation(settings).snapshot(
        persist=False
    )
    assert report["status"] == "READY"
    assert report["primary_horizon_outcomes"] == 2
    assert report["complete_required_horizon_observations"] == 2
    assert report["research_stage"] == "OPTIMIZATION_READY"
    assert report["synthetic_sample_multiplication"] is False
    assert report["automatic_live_authority"] is False
