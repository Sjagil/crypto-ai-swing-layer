from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crypto_ai_swing.settings import Settings


@dataclass(frozen=True)
class CanonicalEngineContract:
    root: Path
    expected_head: str
    actual_head: str
    clean: bool
    branch: str
    matched: bool
    lock_path: Path


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed for canonical crypto checkout"
        )
    return result.stdout.strip()


def load_engine_lock(swing_root: Path) -> dict[str, Any]:
    path = swing_root / "config" / "canonical_crypto_engine.json"
    if not path.is_file():
        raise RuntimeError(
            "canonical crypto engine lock missing; run the Round36.1 installer"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("canonical crypto engine lock is invalid")
    return payload


def validate_engine_contract(
    swing_root: Path,
    *,
    settings: Settings | None = None,
    require_clean: bool = True,
) -> CanonicalEngineContract:
    selected = settings or Settings.load(swing_root)
    crypto_root = Path(selected.crypto_repo_root).expanduser().resolve()
    lock_path = swing_root / "config" / "canonical_crypto_engine.json"
    lock = load_engine_lock(swing_root)
    expected = str(lock.get("commit") or "").strip()
    if len(expected) < 7:
        raise RuntimeError("canonical crypto engine lock has no valid commit")

    actual = _git(crypto_root, "rev-parse", "HEAD")
    branch = _git(crypto_root, "branch", "--show-current")
    porcelain = _git(crypto_root, "status", "--porcelain")
    clean = not bool(porcelain)
    matched = actual == expected

    contract = CanonicalEngineContract(
        root=crypto_root,
        expected_head=expected,
        actual_head=actual,
        clean=clean,
        branch=branch,
        matched=matched,
        lock_path=lock_path,
    )
    if not matched:
        raise RuntimeError(
            "canonical crypto engine HEAD differs from the pinned contract: "
            f"expected={expected} actual={actual}"
        )
    if require_clean and not clean:
        raise RuntimeError(
            "canonical crypto engine checkout is dirty; live orchestration "
            "is fail-closed until source changes are committed or reverted"
        )
    return contract
