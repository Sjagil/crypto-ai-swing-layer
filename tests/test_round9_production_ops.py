from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_scripts_exist_and_default_shadow():
    run = (ROOT / "scripts/run_production.sh").read_text()
    install = (ROOT / "scripts/install_launchd.sh").read_text()
    assert 'CRYPTO_SWING_PRODUCTION_MODE:-shadow' in run
    assert 'CRYPTO_SWING_PRODUCTION_MODE:-shadow' in install
    assert 'CRYPTO_SWING_CANARY_EXECUTE' not in install
    assert '${MODE^^}' not in run
    assert "tr '[:lower:]' '[:upper:]'" in run
    assert 'exec "$PYTHON" -m crypto_ai_swing.cli supervisor --mode "$MODE"' in run


def test_live_startup_gate_requires_all_three_layers():
    gate = (ROOT / "scripts/live_startup_gate.py").read_text()
    assert 'readiness.get("eligible") is not True' in gate
    assert 'canary.get("ready") is not True' in gate
    assert 'CRYPTO_SWING_CANARY_EXECUTE' in gate
    assert '"orders_submitted": 0' in gate


def test_launchd_refuses_direct_live_install():
    install = (ROOT / "scripts/install_launchd.sh").read_text()
    assert 'if [[ "$MODE" == "live" ]]' in install
    assert "Refusing to install LaunchAgent directly in live mode." in install
