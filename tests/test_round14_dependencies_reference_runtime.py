from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from crypto_ai_swing.bridge import reference_runtime
from crypto_ai_swing.research.native_dependencies import (
    inspect_native_campaign_dependencies,
)
from crypto_ai_swing.research.native_tournament import run_native_alpha_tournament


def _native_settings(tmp_path: Path):
    lab = tmp_path / "lab"
    processed = tmp_path / "processed"
    (lab / "reports").mkdir(parents=True)
    processed.mkdir(parents=True)
    return SimpleNamespace(
        paths=SimpleNamespace(
            lab_dir=lab,
            processed_data_dir=processed,
        )
    )


def test_multi_alpha_dependency_reports_missing_absolute_momentum(tmp_path):
    settings = _native_settings(tmp_path)
    reports = settings.paths.lab_dir / "reports"
    (reports / "residual_reversal_campaign_v1.json").write_text("{}")
    for market in ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR"):
        (settings.paths.processed_data_dir / f"{market}_1d.parquet").write_bytes(b"x")

    payload = inspect_native_campaign_dependencies(
        crypto_repo_root=tmp_path,
        campaign="multi-alpha-v2",
        native_settings=settings,
    )

    assert payload["ready"] is False
    assert payload["classification"] == "BLOCKED_PREREQUISITE_EVIDENCE"
    assert payload["missing_report_count"] == 1
    assert payload["missing_dataset_count"] == 0
    assert payload["repair_commands"] == [
        "python main.py lab campaign run --name absolute-momentum-v1 --yes"
    ]
    assert payload["orders_submitted"] == 0


def test_reference_doctor_detects_unregistered_local_env(tmp_path, monkeypatch):
    config = {
        "references": {
            "nautilus_trader": {
                "enabled": True,
                "role": "cross_engine_replay",
                "repository": "https://example.invalid/nautilus.git",
                "directory": "references/nautilus_trader",
                "isolated_env": ".venvs/nautilus",
            }
        }
    }
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "references.yaml").write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )
    (tmp_path / "references" / "nautilus_trader").mkdir(parents=True)
    python_path = tmp_path / ".venvs" / "nautilus" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    (tmp_path / ".venvs" / "stocks").mkdir(parents=True)

    monkeypatch.setattr(
        reference_runtime,
        "_probe_python",
        lambda path, module: {
            "healthy": True,
            "python": "3.12.0",
            "executable": str(path),
            "module": module,
            "module_found": True,
        },
    )

    payload = reference_runtime.reference_environment_status(tmp_path)
    assert payload["references"][0]["status"] == "READY_ISOLATED"
    assert payload["unregistered_envs"] == ["stocks"]
    assert payload["orders_submitted"] == 0


def test_tournament_blocks_prerequisite_before_native_campaign(tmp_path, monkeypatch):
    from crypto_ai_swing.research import native_tournament

    calls = {"native": 0}

    class FakeBridge:
        def __init__(self, root):
            self.root = root

        def run_alpha_campaign(self, name):
            calls["native"] += 1
            raise AssertionError("native campaign must not run when prerequisite is blocked")

    monkeypatch.setattr(native_tournament, "NativeFoundationBridge", FakeBridge)
    monkeypatch.setattr(
        native_tournament,
        "inspect_native_campaign_dependencies",
        lambda **kwargs: {
            "ready": False,
            "classification": "BLOCKED_PREREQUISITE_EVIDENCE",
            "repair_commands": ["repair"],
            "orders_generated": 0,
            "orders_submitted": 0,
        },
    )

    payload = run_native_alpha_tournament(
        crypto_repo_root=tmp_path,
        campaigns=("multi-alpha-v2",),
    )
    row = payload["campaigns"][0]
    assert row["classification"] == "BLOCKED_PREREQUISITE_EVIDENCE"
    assert payload["counts"]["blocked_prerequisite_evidence"] == 1
    assert calls["native"] == 0
