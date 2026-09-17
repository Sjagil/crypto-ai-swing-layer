from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from crypto_ai_swing.orchestration.round43_controller import (
    Round43AutomationController,
)
from scripts.round43_launchd import build_launchd_plist


def test_round43_controller_forbids_live_before_runtime_construction(tmp_path):
    settings = SimpleNamespace(
        project_root=tmp_path,
        autonomy={},
    )
    with pytest.raises(ValueError, match="LIVE_MODE_FORBIDDEN"):
        Round43AutomationController(settings, mode="live")


def test_round43_launchd_has_no_live_mode(tmp_path):
    root = tmp_path
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/python").write_text("", encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts/round43_runtime.py").write_text("", encoding="utf-8")
    crypto = root / "crypto"
    crypto.mkdir()
    settings = SimpleNamespace(
        project_root=root,
        crypto_repo_root=crypto,
    )
    payload = build_launchd_plist(settings, mode="shadow")
    assert "--forever" in payload["ProgramArguments"]
    assert payload["ProgramArguments"][0] == "/usr/bin/caffeinate"
    assert "-s" in payload["ProgramArguments"]
    assert "live" not in payload["ProgramArguments"]
    environment = payload["EnvironmentVariables"]
    assert Path(environment["HF_HOME"]).is_absolute()
    assert Path(environment["HUGGINGFACE_HUB_CACHE"]).is_absolute()
    assert ":\\" not in environment["HF_HOME"]
    assert ":\\" not in environment["HUGGINGFACE_HUB_CACHE"]
    with pytest.raises(ValueError, match="LIVE_MODE_FORBIDDEN"):
        build_launchd_plist(settings, mode="live")


def test_round43_config_never_relaxes_live_or_risk_gates():
    config = yaml.safe_load(Path("config/autonomy.yaml").read_text())
    automation = config["round43_automation"]
    optimizer = config["optimization_controller"]
    assert automation["allow_live_mode"] is False
    assert optimizer["threshold_relaxation_allowed"] is False
    assert optimizer["automatic_live_authority"] is False
    assert optimizer["automatic_live_promotion"] is False

def test_round43_runtime_resanitizes_windows_cache_paths(monkeypatch):
    from scripts.round43_runtime import _configure_ml_cache_environment

    monkeypatch.setenv("HF_HOME", r"D:\huggingface_cache")
    monkeypatch.setenv(
        "HUGGINGFACE_HUB_CACHE",
        r"D:\huggingface_cache\hub",
    )
    monkeypatch.setenv(
        "TRANSFORMERS_CACHE",
        r"D:\huggingface_cache\transformers",
    )

    payload = _configure_ml_cache_environment()

    assert Path(payload["HF_HOME"]).is_absolute()
    assert Path(payload["HUGGINGFACE_HUB_CACHE"]).is_absolute()
    assert not payload["HF_HOME"].startswith("D:")
    assert not payload["HUGGINGFACE_HUB_CACHE"].startswith("D:")
    assert "TRANSFORMERS_CACHE" not in os.environ

def test_round43_runtime_sanitizes_all_hf_cache_aliases(monkeypatch):
    from scripts.round43_runtime import _configure_ml_cache_environment

    bad = {
        "HF_HOME": r"D:\huggingface_cache",
        "HF_HUB_CACHE": r"D:\huggingface_cache\hub",
        "HUGGINGFACE_HUB_CACHE": r"D:\huggingface_cache\hub",
        "HF_ASSETS_CACHE": r"D:\huggingface_cache\assets",
        "HF_XET_CACHE": r"D:\huggingface_cache\xet",
        "SENTENCE_TRANSFORMERS_HOME": r"D:\huggingface_cache\sentence_transformers",
        "TRANSFORMERS_CACHE": r"D:\huggingface_cache\transformers",
    }
    for key, value in bad.items():
        monkeypatch.setenv(key, value)

    payload = _configure_ml_cache_environment()

    for key, value in payload.items():
        assert Path(value).is_absolute(), key
        assert not value.startswith("D:"), key
    assert "TRANSFORMERS_CACHE" not in os.environ
