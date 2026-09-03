from __future__ import annotations

from pathlib import Path

import yaml

from crypto_ai_swing.bridge import reference_runtime
from crypto_ai_swing.bridge.native_foundation import NativeFoundationBridge
from crypto_ai_swing.research.native_tournament import _classification


def _write_reference_config(
    root: Path,
    isolated_env: str = ".venvs/vectorbt",
) -> None:
    config = {
        "references": {
            "vectorbt": {
                "enabled": True,
                "role": "vectorized_research_crosscheck",
                "repository": "https://example.invalid/vectorbt.git",
                "directory": "references/vectorbt",
                "isolated_env": isolated_env,
            }
        }
    }
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "references.yaml").write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )


def test_reference_doctor_resolves_parent_workspace_venv(
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "crypto-ai-swing-layer"
    project.mkdir()
    _write_reference_config(project)
    python_path = tmp_path / ".venvs" / "vectorbt" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")

    monkeypatch.setattr(
        reference_runtime,
        "_probe_python",
        lambda path, module: {
            "healthy": True,
            "process_healthy": True,
            "python": "3.12.0",
            "executable": str(path),
            "module": module,
            "module_found": True,
            "error": None,
        },
    )

    payload = reference_runtime.reference_environment_status(project)
    row = payload["references"][0]
    assert row["env_binding"] == "DECLARED_WORKSPACE_RESOLVED"
    assert row["isolated_env"] == str(tmp_path / ".venvs" / "vectorbt")
    assert row["runtime_ready"] is True
    assert row["status"] == "READY_PACKAGE_ISOLATED"
    assert "vectorbt" in payload["detected_envs"]


def test_probe_requires_declared_module_to_exist(tmp_path, monkeypatch):
    python_path = tmp_path / "python"
    python_path.write_text("")

    class Proc:
        returncode = 0
        stdout = (
            '{"python":"3.12.0","executable":"x","platform":"test",'
            '"module":"vectorbt","module_found":false}'
        )
        stderr = ""

    monkeypatch.setattr(
        reference_runtime.subprocess,
        "run",
        lambda *args, **kwargs: Proc(),
    )
    payload = reference_runtime._probe_python(
        python_path,
        "vectorbt",
    )
    assert payload["process_healthy"] is True
    assert payload["module_found"] is False
    assert payload["healthy"] is False
    assert payload["error"] == "PROBE_MODULE_NOT_FOUND"


def test_compact_campaign_preserves_native_trial_and_forward_evidence():
    payload = {
        "campaign": "RESIDUAL_REVERSAL_V1",
        "status": "COMPLETED_NOT_PROMOTED",
        "generated_trial_count": 8,
        "registered_unique_trials": 8,
        "total_known_trials": 21345,
        "primary_strategy_id": "RR_B60_H5_Z20",
        "pbo": 0.02857142857142857,
        "economic_pass": False,
        "statistical_pass": False,
        "paper_candidates": 0,
        "live_ready": False,
        "forward_summaries": {
            "RR_A": {
                "status": "COLLECTING_FORWARD_DATA",
                "closed_daily_observations": 38,
                "required_closed_daily_observations": 365,
                "forward_rebalances": 4,
                "required_forward_rebalances": 30,
                "forward_net_return": 0.014,
                "formal_performance_gates_evaluated": False,
                "regime_coverage": {
                    "counts": {
                        "volatility": {
                            "HIGH": 0,
                            "LOW": 38,
                        }
                    }
                },
            },
            "RR_B": {
                "status": "COLLECTING_FORWARD_DATA",
                "closed_daily_observations": 38,
                "required_closed_daily_observations": 365,
                "forward_rebalances": 2,
                "required_forward_rebalances": 30,
                "forward_net_return": -0.002,
                "formal_performance_gates_evaluated": False,
                "regime_coverage": {
                    "counts": {
                        "volatility": {
                            "HIGH": 0,
                            "LOW": 38,
                        }
                    }
                },
            },
        },
    }

    summary = NativeFoundationBridge._compact_campaign(payload)
    assert summary["selected_candidate"] == "RR_B60_H5_Z20"
    assert summary["candidate_count"] == 8
    assert summary["registered_unique_trials"] == 8
    assert summary["total_known_trials"] == 21345
    assert summary["pbo"] == 0.02857142857142857

    forward = summary["forward_evidence"]
    assert forward["maximum_closed_daily_observations"] == 38
    assert forward["required_closed_daily_observations"] == 365
    assert forward["maximum_forward_rebalances"] == 4
    assert forward["required_forward_rebalances"] == 30
    assert forward["maximum_high_volatility_observations"] == 0
    assert forward["formal_performance_gates_evaluated_for_any"] is False
    assert forward["diagnostic_forward_returns_authorize_promotion"] is False
    assert forward["best_diagnostic_forward"] == {
        "candidate": "RR_A",
        "net_return": 0.014,
    }


def test_tournament_separates_failed_history_from_collecting_forward_data():
    payload = {
        "summary": {
            "economic_pass": False,
            "statistical_pass": False,
            "paper_candidate_permitted": False,
            "live_ready": False,
            "forward_evidence": {
                "statuses": ["COLLECTING_FORWARD_DATA"],
            },
        }
    }
    assert (
        _classification(payload)
        == "HISTORICAL_GATES_FAILED_FORWARD_COLLECTING"
    )


def test_tournament_marks_qualified_history_waiting_for_forward_data():
    payload = {
        "summary": {
            "economic_pass": True,
            "statistical_pass": True,
            "paper_candidate_permitted": False,
            "live_ready": False,
            "forward_evidence": {
                "statuses": ["COLLECTING_FORWARD_DATA"],
            },
        }
    }
    assert _classification(payload) == "FORWARD_EVIDENCE_COLLECTING"
