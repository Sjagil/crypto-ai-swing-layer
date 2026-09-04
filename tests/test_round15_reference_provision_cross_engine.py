from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from crypto_ai_swing.bridge import reference_provision
from crypto_ai_swing.validation.cross_engine import _native_fixed_size_replay, _signal_schedule


def _frame(rows: int = 300) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    x = np.linspace(0, 12, rows)
    close = 100.0 + x + np.sin(x) * 4.0
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.ones(rows) * 10.0,
        },
        index=index,
    )


def test_signal_schedule_is_causal_and_long_only():
    frame = _frame()
    schedule = _signal_schedule(frame, fast=10, slow=30)
    assert list(schedule.columns) == ["target_long", "entry", "exit"]
    assert not bool(schedule["target_long"].iloc[0])
    assert int(schedule["entry"].sum()) >= 1


def test_native_fixed_size_replay_is_bounded():
    frame = _frame()
    schedule = _signal_schedule(frame, fast=10, slow=30)
    result = _native_fixed_size_replay(
        frame,
        schedule,
        initial_cash=10000.0,
        trade_notional=1000.0,
        fee_fraction=0.0025,
        execution_drag_fraction=0.0005,
    )
    assert result["status"] == "COMPLETED"
    assert result["orders"] >= 1
    assert result["final_equity"] > 0
    assert 0 <= result["max_drawdown"] < 1


def test_reference_provision_plan_does_not_execute(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/references.yaml").write_text(
        yaml.safe_dump(
            {
                "references": {
                    "vectorbt": {
                        "enabled": True,
                        "repository": "https://example.invalid/vectorbt.git",
                        "directory": "references/vectorbt",
                        "isolated_env": ".venvs/vectorbt",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "config/reference_provision.yaml").write_text(
        yaml.safe_dump({"packages": {"vectorbt": {"pip": "vectorbt>=0.28,<1"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        reference_provision,
        "reference_environment_status",
        lambda *args, **kwargs: {"summary": {"enabled": 1}},
    )
    payload = reference_provision.provision_references(
        tmp_path,
        venv_root=tmp_path / ".venvs",
        repo_root=tmp_path / "references",
        apply=False,
    )
    kinds = [x["kind"] for x in payload["references"][0]["actions"]]
    assert "create_env" in kinds
    assert "install_package" in kinds
    assert "clone_source" in kinds
