import pandas as pd
import pytest

from crypto_ai_swing.research.forward import ForwardEvidenceLedger


def test_forward_outcomes_mature_only_after_horizon(tmp_path):
    ledger = ForwardEvidenceLedger(tmp_path / "forward.sqlite")
    try:
        payload = {
            "generated_at": "2026-01-01T01:00:00+00:00",
            "mode": "shadow",
            "markets": ["BTC-EUR"],
            "signals": [
                {"market": "BTC-EUR", "side": "BUY", "score": 0.8, "edge_bps": 100.0}
            ],
            "blocked": [],
            "market_context": {"BTC-EUR": {}},
        }
        ledger.append_cycle(payload, {"BTC-EUR": "2026-01-01T00:00:00+00:00"})
        idx = pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="1h")
        frame = pd.DataFrame(
            {
                "open": [100, 100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105, 106],
                "low": [99, 99.5, 100, 101, 102, 103],
                "close": [100, 101, 102, 103, 104, 105],
                "volume": [1] * 6,
            },
            index=idx,
        )
        result = ledger.mature_from_frames({"BTC-EUR": frame}, horizons_hours=(1, 4, 24))
        assert result["outcomes_inserted"] == 2
        assert result["by_horizon_hours"] == {1: 1, 4: 1}
        report = ledger.report()
        one_hour = next(x for x in report["groups"] if x["horizon_hours"] == 1)
        assert one_hour["mean_return_bps"] == pytest.approx(100.0)
        assert report["automatic_live_promotion"] is False
    finally:
        ledger.close()
