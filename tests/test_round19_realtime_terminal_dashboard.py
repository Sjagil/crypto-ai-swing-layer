from types import SimpleNamespace

from crypto_ai_swing.monitoring.terminal_dashboard import (
    RealtimeTerminalDashboard,
    _find,
)


def test_nested_runtime_lookup():
    value = {"a": {"runtime_dependencies": {"public_stream_ready": True}}}
    assert _find(value, "runtime_dependencies")["public_stream_ready"] is True


def test_dashboard_surface_has_no_execution_methods(tmp_path):
    crypto_root = tmp_path / "crypto"
    crypto_root.mkdir()
    settings = SimpleNamespace(
        project_root=tmp_path,
        crypto_repo_root=crypto_root,
    )
    dashboard = RealtimeTerminalDashboard(
        settings,
        markets=("BTC-EUR",),
        public_feed=False,
    )
    forbidden = {"submit_buy", "submit_exit", "approve", "place_order"}
    assert forbidden.isdisjoint(set(dir(dashboard)))
