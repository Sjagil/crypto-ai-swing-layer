from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_round10_reuses_native_crypto_components():
    source = (ROOT / "src/crypto_ai_swing/bridge/crypto_operations.py").read_text()
    for value in (
        "utils.common",
        "atomic_write_json",
        "notifications.telegram",
        "TelegramNotifier",
        "risk.correlation_analyzer",
        "CorrelationAnalyzer",
        "risk.risk_manager",
        "OperationalDegradation",
        "research.trading_math",
        "calculate_position_size",
        "core.live_asset_preflight",
        "live_account_health",
    ):
        assert value in source


def test_round10_supervisor_has_progress_heartbeat_and_native_ops():
    source = (ROOT / "src/crypto_ai_swing/orchestration/supervisor.py").read_text()
    assert "NativeOperationsBridge" in source
    assert "_heartbeat_pump" in source
    assert "last_progress_at" in source
    assert "task_heartbeat_seconds" in source
    assert "OPERATIONAL_DEGRADATION" in source
    assert "OPERATIONAL_RECOVERY" in source


def test_round10_runtime_health_supports_busy_state_and_pid():
    source = (ROOT / "scripts/runtime_health.py").read_text()
    assert "HEALTHY_BUSY" in source
    assert "last_progress_at" in source
    assert "SUPERVISOR_PID_NOT_ALIVE" in source
