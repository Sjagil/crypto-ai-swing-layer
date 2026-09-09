from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from crypto_ai_swing.research.optuna_runtime import (
    OptunaRuntimeError,
    OptunaStudyRuntime,
    resolve_optuna_python,
)
from crypto_ai_swing.research.selector_optuna import (
    SelectorOptunaTuner,
    _trial_budget,
)


def test_contract_hash_is_stable_and_sensitive():
    one = OptunaStudyRuntime.contract_hash(
        search_space={"x": {"kind": "int", "low": 1, "high": 3}},
        objective_version="v1",
        metadata={"a": 1},
    )
    two = OptunaStudyRuntime.contract_hash(
        search_space={"x": {"kind": "int", "low": 1, "high": 3}},
        objective_version="v1",
        metadata={"a": 1},
    )
    changed = OptunaStudyRuntime.contract_hash(
        search_space={"x": {"kind": "int", "low": 1, "high": 4}},
        objective_version="v1",
        metadata={"a": 1},
    )
    assert one == two
    assert one != changed


def _synthetic_rows(count: int = 120):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for index in range(count):
        observed = start + timedelta(hours=24 * index)
        oid = f"obs-{index:04d}"
        for horizon in (24, 72, 168):
            rows.append(
                {
                    "observation_id": oid,
                    "observed_dt": observed,
                    "matured_dt": observed + timedelta(hours=horizon),
                    "horizon_hours": horizon,
                }
            )
    return rows


class _FakeSelector:
    horizons = (24, 72, 168)
    minimum_oos_selected = 20
    minimum_train = 12


def test_outer_split_embargoes_all_calibration_labels(tmp_path):
    fake = SimpleNamespace(
        selector=_FakeSelector(),
    )
    rows = _synthetic_rows()
    split = SelectorOptunaTuner._outer_split(fake, rows)
    calibration = set(split.calibration_ids)
    holdout = set(split.holdout_ids)
    calibration_maturity = max(
        row["matured_dt"] for row in rows if row["observation_id"] in calibration
    )
    holdout_start = min(
        row["observed_dt"] for row in rows if row["observation_id"] in holdout
    )
    assert holdout_start > calibration_maturity
    assert calibration.isdisjoint(holdout)


def test_isolated_optuna_runtime_ask_tell_and_resume(tmp_path):
    project_root = Path.cwd()
    try:
        python_path, _ = resolve_optuna_python(project_root)
    except OptunaRuntimeError as exc:
        pytest.skip(str(exc))

    storage = tmp_path / "study.sqlite3"
    runtime = OptunaStudyRuntime(
        project_root,
        storage_path=storage,
        python_path=python_path,
    )
    search_space = {
        "x": {"kind": "float", "low": -1.0, "high": 1.0, "step": 0.1},
        "n": {"kind": "int", "low": 2, "high": 5},
    }
    ensured = runtime.ensure_study(
        study_name="round28-test",
        search_space=search_space,
        objective_version="test_v1",
        metadata={"outer_holdout_used_for_search": False},
        seed=28,
    )
    contract = str(ensured["contract_hash"])
    asked = runtime.ask(
        study_name="round28-test",
        contract_hash=contract,
        search_space=search_space,
        seed=28,
    )
    runtime.tell(
        study_name="round28-test",
        contract_hash=contract,
        trial_number=int(asked["trial_number"]),
        value=1.25,
    )

    resumed = OptunaStudyRuntime(
        project_root,
        storage_path=storage,
        python_path=python_path,
    )
    summary = resumed.summary(study_name="round28-test", contract_hash=contract)
    assert summary["trial_count"] == 1
    assert summary["best"]["value"] == pytest.approx(1.25)


def test_study_contract_mismatch_fails_closed(tmp_path):
    project_root = Path.cwd()
    try:
        python_path, _ = resolve_optuna_python(project_root)
    except OptunaRuntimeError as exc:
        pytest.skip(str(exc))
    runtime = OptunaStudyRuntime(
        project_root,
        storage_path=tmp_path / "contract.sqlite3",
        python_path=python_path,
    )
    space = {"x": {"kind": "int", "low": 1, "high": 2}}
    ensured = runtime.ensure_study(
        study_name="contract-test",
        search_space=space,
        objective_version="v1",
    )
    wrong = "0" * 64
    assert wrong != ensured["contract_hash"]
    with pytest.raises(OptunaRuntimeError, match="CONTRACT_MISMATCH"):
        runtime.summary(study_name="contract-test", contract_hash=wrong)
def test_explicit_python_path_preserves_venv_entrypoint(monkeypatch, tmp_path):
    project_root = tmp_path / "repo"
    worker = project_root / "scripts" / "workers" / "optuna_study_worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# test worker\n", encoding="utf-8")

    base_python = tmp_path / "base-python"
    base_python.write_text("#!/bin/sh\n", encoding="utf-8")

    venv_python = project_root / ".venvs" / "optuna" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)

    def fake_probe(path):
        return {
            "ready": True,
            "python": str(path),
            "version": "4.9.0",
            "reason": None,
        }

    monkeypatch.setattr(
        "crypto_ai_swing.research.optuna_runtime._probe_python",
        fake_probe,
    )

    runtime = OptunaStudyRuntime(
        project_root,
        storage_path=tmp_path / "study.sqlite3",
        python_path=venv_python,
    )

    assert runtime.python_path == venv_python.absolute()
    assert runtime.python_path != venv_python.resolve()
def test_outer_split_requires_complete_horizon_labels():
    fake = SimpleNamespace(selector=_FakeSelector())
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for index in range(150):
        observed = start + timedelta(hours=12 * index)
        rows.append(
            {
                "observation_id": f"partial-{index:04d}",
                "observed_dt": observed,
                "matured_dt": observed + timedelta(hours=24),
                "horizon_hours": 24,
            }
        )
    with pytest.raises(
        ValueError,
        match="INSUFFICIENT_COMPLETE_HORIZON_OBSERVATIONS_FOR_NESTED_OPTUNA",
    ):
        SelectorOptunaTuner._outer_split(fake, rows)


def test_outer_split_allows_calibration_before_outer_holdout_exists():
    fake = SimpleNamespace(selector=SimpleNamespace(
        horizons=(24, 72, 168),
        minimum_oos_selected=30,
        minimum_train=24,
    ))
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for index in range(72):
        observed = start + timedelta(hours=12 * index)
        for horizon in (24, 72, 168):
            rows.append(
                {
                    "observation_id": f"complete-{index:04d}",
                    "observed_dt": observed,
                    "matured_dt": observed + timedelta(hours=horizon),
                    "horizon_hours": horizon,
                }
            )
    split = SelectorOptunaTuner._outer_split(fake, rows)
    assert split.calibration_observations == 72
    assert split.holdout_observations == 0
    assert split.holdout_start is None


def test_trial_budget_uses_target_total_semantics():
    study = {
        "trials": [
            {"state": "COMPLETE"},
            {"state": "FAIL"},
            {"state": "RUNNING"},
        ]
    }
    budget = _trial_budget(study, 5)
    assert budget == {
        "target_total": 5,
        "finished_before": 2,
        "running_before": 1,
        "remaining": 3,
    }
    assert _trial_budget({"trials": [{"state": "COMPLETE"}] * 40}, 40)[
        "remaining"
    ] == 0
