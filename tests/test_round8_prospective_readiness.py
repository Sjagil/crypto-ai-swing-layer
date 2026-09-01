from datetime import datetime, timedelta, timezone

from crypto_ai_swing.research.forward import (
    CANONICAL_OUTCOME_SCHEMA,
    ForwardEvidenceLedger,
)


def _seed_outcome(ledger, *, observed_at, market, return_bps):
    payload = {
        "generated_at": observed_at.isoformat(),
        "mode": "shadow",
        "markets": [market],
        "signals": [
            {
                "market": market,
                "side": "BUY",
                "score": 0.8,
                "edge_bps": 80.0,
            }
        ],
        "blocked": [],
        "market_context": {market: {}},
    }
    ledger.append_cycle(payload, {market: observed_at.isoformat()})
    observation_id = ledger.conn.execute(
        "SELECT observation_id FROM signal_observations "
        "WHERE market=? AND observed_at=?",
        (market, observed_at.isoformat()),
    ).fetchone()[0]
    ledger.conn.execute(
        "INSERT INTO forward_outcomes_v2 VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            observation_id,
            4,
            (observed_at + timedelta(hours=4)).isoformat(),
            observed_at.isoformat(),
            observed_at.isoformat(),
            100.0,
            100.0 * (1 + return_bps / 10000.0),
            float(return_bps),
            max(0.0, float(return_bps) + 10.0),
            min(0.0, float(return_bps) - 20.0),
            CANONICAL_OUTCOME_SCHEMA,
        ),
    )
    ledger.conn.commit()


def test_canary_readiness_requires_real_prospective_sample(tmp_path):
    ledger = ForwardEvidenceLedger(tmp_path / "forward.sqlite")
    try:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        markets = [f"M{i}-EUR" for i in range(5)]
        for i in range(30):
            _seed_outcome(
                ledger,
                observed_at=start + timedelta(hours=4 * i),
                market=markets[i % len(markets)],
                return_bps=35.0 + (i % 3),
            )
        readiness = ledger.canary_readiness(
            primary_horizon_hours=4,
            minimum_unblocked_buy_outcomes=30,
            minimum_distinct_markets=5,
            minimum_observation_span_hours=72,
            minimum_mean_return_bps=0.0,
            minimum_positive_return_rate=0.5,
        )
        assert readiness["eligible"] is True
        assert readiness["status"] == "PRELIMINARY_CANARY_ELIGIBLE"
        assert readiness["unblocked_buy"]["observations"] == 30
        assert readiness["unblocked_buy"]["markets"] == 5
        assert readiness["automatic_live_promotion"] is False
        assert readiness["autoscale_authority"] is False
    finally:
        ledger.close()


def test_negative_sparse_forward_evidence_never_grants_canary(tmp_path):
    ledger = ForwardEvidenceLedger(tmp_path / "forward.sqlite")
    try:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        _seed_outcome(
            ledger,
            observed_at=start,
            market="BTC-EUR",
            return_bps=-50,
        )
        _seed_outcome(
            ledger,
            observed_at=start + timedelta(hours=4),
            market="SOL-EUR",
            return_bps=-100,
        )
        readiness = ledger.canary_readiness()
        assert readiness["eligible"] is False
        assert readiness["status"] == "COLLECTING"
        assert "INSUFFICIENT_UNBLOCKED_BUY_OUTCOMES" in readiness["blockers"]
        assert "PROSPECTIVE_MEAN_RETURN_NOT_POSITIVE" in readiness["blockers"]
    finally:
        ledger.close()
