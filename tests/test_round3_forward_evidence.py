from crypto_ai_swing.research.forward import ForwardEvidenceLedger


def test_forward_evidence_is_restart_safe(tmp_path):
    path = tmp_path / "forward.sqlite"
    payload = {
        "generated_at": "2026-08-30T17:00:00+00:00",
        "mode": "shadow",
        "markets": ["BTC-EUR"],
        "signals": [
            {"market": "BTC-EUR", "side": "BUY", "score": 0.8, "edge_bps": 90.0}
        ],
        "blocked": [{"market": "BTC-EUR", "blockers": ["NET_EDGE"]}],
        "market_context": {"BTC-EUR": {"mtf_score": 0.7}},
    }
    ledger = ForwardEvidenceLedger(path)
    try:
        first = ledger.append_cycle(payload, {"BTC-EUR": "2026-08-30T16:00:00+00:00"})
        second = ledger.append_cycle(payload, {"BTC-EUR": "2026-08-30T16:00:00+00:00"})
        status = ledger.status()
    finally:
        ledger.close()
    assert first["signal_observations_inserted"] == 1
    assert second["signal_observations_inserted"] == 0
    assert status == {"cycles": 1, "signals": 1, "blocked_signals": 1}
