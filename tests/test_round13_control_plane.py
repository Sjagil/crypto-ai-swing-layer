from __future__ import annotations

from pathlib import Path

from crypto_ai_swing.orchestration.control_plane import (
    mode_runtime_path,
)


def test_mode_state_paths_are_isolated(tmp_path: Path):
    shadow = mode_runtime_path(tmp_path, "shadow")
    paper = mode_runtime_path(tmp_path, "paper")
    canary = mode_runtime_path(tmp_path, "canary")

    assert shadow != paper
    assert paper != canary
    assert shadow != canary
    assert shadow.name == "shadow"
    assert paper.name == "paper"
    assert canary.name == "canary"
