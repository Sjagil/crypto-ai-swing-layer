import pandas as pd
import pytest

from crypto_ai_swing.research.forward import ForwardEvidenceLedger


def test_forward_uses_first_future_hour_after_actual_observation(tmp_path):
    ledger = ForwardEvidenceLedger(tmp_path / "forward.sqlite", decision_bucket_minutes=15)
    try:
        payload = {
            "generated_at": "2026-01-01T01:20:00+00:00",
            "mode": "shadow",
            "markets": ["BTC-EUR"],
            "signals": [
                {"market": "BTC-EUR", "side": "BUY", "score": 0.8, "edge_bps": 50.0}
            ],
            "blocked": [],
            "market_context": {"BTC-EUR": {}},
        }
        first = ledger.append_cycle(payload, {"BTC-EUR": "2026-01-01T00:00:00+00:00"})
        payload2 = dict(payload)
        payload2["generated_at"] = "2026-01-01T01:29:00+00:00"
        second = ledger.append_cycle(payload2, {"BTC-EUR": "2026-01-01T00:00:00+00:00"})
        assert first["signal_observations_inserted"] == 1
        assert second["signal_observations_inserted"] == 0

        idx = pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="1h")
        frame = pd.DataFrame(
            {
                "open": [90, 95, 100, 101, 102, 103],
                "high": [91, 120, 105, 103, 104, 105],
                "low": [89, 80, 101, 100, 101, 102],
                "close": [90, 110, 102, 102, 103, 104],
                "volume": [1] * 6,
            },
            index=idx,
        )
        result = ledger.mature_from_frames({"BTC-EUR": frame}, horizons_hours=(1,))
        assert result["outcomes_inserted"] == 1
        report = ledger.report()
        group = report["groups"][0]
        # Observation at 01:20 cannot use the already-known 01:00 bar. The
        # causal execution proxy is the 02:00 open at 100.
        assert group["mean_return_bps"] == pytest.approx(200.0)
        assert group["mean_mfe_bps"] == pytest.approx(500.0)
        # The entire future bar low stayed above entry, so adverse excursion is 0.
        assert group["mean_mae_bps"] == pytest.approx(0.0)
        assert report["causality"]["legacy_v1_excluded_from_current_metrics"] is True
    finally:
        ledger.close()
