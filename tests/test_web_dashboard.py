from pathlib import Path

from crypto_ai_swing.monitoring.web_dashboard import _artifact, _mdpro_rows, _overall_state


def test_artifact_missing(tmp_path: Path):
    payload = _artifact(tmp_path / "missing.json")
    assert payload["exists"] is False
    assert payload["payload"] == {}


def test_mdpro_rows_marks_market_sequence():
    payload = {
        "manager": {"markets": {"BTC-EUR": {"status": "READY", "spread_bps": 1.2}}},
        "realtime": {"markets": [{"market": "BTC-EUR", "sequence_valid": True, "fresh": True}]},
    }
    rows = _mdpro_rows(payload)
    assert rows == [
        {
            "market": "BTC-EUR",
            "status": "READY",
            "sequence_valid": True,
            "fresh": True,
            "spread_bps": 1.2,
            "bid_levels": None,
            "ask_levels": None,
            "last_error": None,
        }
    ]


def test_overall_state_requires_live_and_25_mdpro():
    assert _overall_state({"live": {"preflight_ready": True}, "mdpro": {"ready_count": 25}}) == "CANARY_READY"
    assert _overall_state({"live": {"preflight_ready": False, "authority_active": True}, "mdpro": {"ready_count": 25}}) == "CANARY_BLOCKED"
