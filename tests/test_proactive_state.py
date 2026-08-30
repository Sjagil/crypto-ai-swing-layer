from decimal import Decimal
from crypto_ai_swing.orchestration.proactive import ProactiveState, Position


def test_proactive_state_is_restart_safe(tmp_path):
    path = tmp_path / "state.sqlite"
    state = ProactiveState(path)
    state.mark("BTC-EUR", "2026-01-01T00:00:00+00:00", "BUY", {"x": 1})
    state.upsert_position(Position("BTC-EUR", Decimal("0.01"), Decimal("50000"), Decimal("51000"), 0.03, 0.08, 0.04, "now"))
    state.close()
    state = ProactiveState(path)
    assert state.seen("BTC-EUR", "2026-01-01T00:00:00+00:00", "BUY")
    assert "BTC-EUR" in state.positions()
    state.close()
