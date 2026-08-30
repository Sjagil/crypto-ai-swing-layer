from crypto_ai_swing.execution.bitvavo import live_gate_status


def test_live_gate_fails_closed_without_env(monkeypatch):
    for key in (
        "LIVE_TRADING_ALLOWED", "WM_REAL_ORDER_EXECUTION_ENABLED", "WM_I_UNDERSTAND_LIVE_RISK",
        "WM_LIVE_APPROVAL", "CRYPTO_SWING_LIVE_ACK", "WITHDRAWALS_DISABLED",
        "BITVAVO_TRADE_API_KEY", "BITVAVO_API_KEY", "BITVAVO_TRADE_API_SECRET",
        "BITVAVO_API_SECRET", "WM_OPERATOR_ID", "BITVAVO_TRADE_KEY_SCOPE",
    ):
        monkeypatch.delenv(key, raising=False)
    status = live_gate_status({"execution": {"live_enabled": True}})
    assert not status.ready
    assert status.blockers


def test_live_gate_can_become_ready(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ALLOWED", "true")
    monkeypatch.setenv("WM_REAL_ORDER_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("WM_I_UNDERSTAND_LIVE_RISK", "true")
    monkeypatch.setenv("WM_LIVE_APPROVAL", "APPROVED_LIVE_ACTIVE")
    monkeypatch.setenv("WITHDRAWALS_DISABLED", "true")
    monkeypatch.setenv("BITVAVO_TRADE_API_KEY", "key")
    monkeypatch.setenv("BITVAVO_TRADE_API_SECRET", "secret")
    monkeypatch.setenv("WM_OPERATOR_ID", "123")
    monkeypatch.setenv("BITVAVO_TRADE_KEY_SCOPE", "read,trade")
    status = live_gate_status({"execution": {"live_enabled": True}})
    assert status.ready
