#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import plistlib
import subprocess
from pathlib import Path
from typing import Any

from crypto_ai_swing.settings import Settings

LABEL = "com.sjagil.crypto-ai-swing.round43"


def build_launchd_plist(
    settings,
    *,
    mode: str = "shadow",
) -> dict[str, Any]:
    selected = str(mode).lower()
    if selected not in {"shadow", "paper"}:
        raise ValueError("ROUND43_LAUNCHD_LIVE_MODE_FORBIDDEN")

    root = Path(settings.project_root).resolve()
    python = root / ".venv/bin/python"
    runtime = root / "scripts/round43_runtime.py"
    log_root = root / "output/crypto_ai_swing/autonomy/round43/logs"
    log_root.mkdir(parents=True, exist_ok=True)

    if not python.is_file():
        raise FileNotFoundError(f"venv python missing: {python}")
    if not runtime.is_file():
        raise FileNotFoundError(f"runtime script missing: {runtime}")

    return {
        "Label": LABEL,
        "ProgramArguments": [
            "/usr/bin/caffeinate",
            "-s",
            str(python),
            str(runtime),
            "--mode",
            selected,
            "--forever",
        ],
        "WorkingDirectory": str(root),
        "EnvironmentVariables": {
            "CRYPTO_REPO_PATH": str(settings.crypto_repo_root),
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": str(Path.home() / ".cache" / "huggingface"),
            "HF_HUB_CACHE": str(
                Path.home() / ".cache" / "huggingface" / "hub"
            ),
            "HUGGINGFACE_HUB_CACHE": str(
                Path.home() / ".cache" / "huggingface" / "hub"
            ),
            "HF_ASSETS_CACHE": str(
                Path.home() / ".cache" / "huggingface" / "assets"
            ),
            "HF_XET_CACHE": str(
                Path.home() / ".cache" / "huggingface" / "xet"
            ),
            "SENTENCE_TRANSFORMERS_HOME": str(
                Path.home()
                / ".cache"
                / "huggingface"
                / "sentence_transformers"
            ),
        },
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "ThrottleInterval": 30,
        "StandardOutPath": str(log_root / "stdout.log"),
        "StandardErrorPath": str(log_root / "stderr.log"),
    }


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _plist_path() -> Path:
    return (
        Path.home()
        / "Library/LaunchAgents"
        / f"{LABEL}.plist"
    )


def _launchctl(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["launchctl", *args],
        check=check,
        text=True,
        capture_output=True,
    )


def install(settings, *, mode: str) -> dict[str, Any]:
    if platform.system() != "Darwin":
        raise RuntimeError("ROUND43_LAUNCHD_REQUIRES_MACOS")
    payload = build_launchd_plist(settings, mode=mode)
    path = _plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        plistlib.dump(payload, handle, sort_keys=True)

    _launchctl("bootout", _domain(), str(path), check=False)
    result = _launchctl(
        "bootstrap",
        _domain(),
        str(path),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"launchctl bootstrap failed: {result.stderr.strip()}"
        )
    _launchctl(
        "enable",
        f"{_domain()}/{LABEL}",
        check=False,
    )
    return {
        "status": "INSTALLED",
        "label": LABEL,
        "mode": mode,
        "plist": str(path),
        "automatic_restart": True,
        "run_at_login": True,
        "automatic_live_authority": False,
        "automatic_live_promotion": False,
    }


def uninstall() -> dict[str, Any]:
    path = _plist_path()
    _launchctl("bootout", _domain(), str(path), check=False)
    path.unlink(missing_ok=True)
    return {
        "status": "UNINSTALLED",
        "label": LABEL,
        "automatic_live_authority": False,
    }


def status() -> dict[str, Any]:
    result = _launchctl(
        "print",
        f"{_domain()}/{LABEL}",
        check=False,
    )
    return {
        "status": "RUNNING" if result.returncode == 0 else "NOT_LOADED",
        "label": LABEL,
        "plist": str(_plist_path()),
        "plist_exists": _plist_path().is_file(),
        "launchctl_excerpt": (
            result.stdout[-4000:]
            if result.returncode == 0
            else result.stderr[-1000:]
        ),
        "automatic_live_authority": False,
        "automatic_live_promotion": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("install", "status", "uninstall"),
        default="status",
        nargs="?",
    )
    parser.add_argument(
        "--mode",
        choices=("shadow", "paper"),
        default="shadow",
    )
    args = parser.parse_args()
    settings = Settings.load(Path.cwd())

    if args.action == "install":
        payload = install(settings, mode=args.mode)
    elif args.action == "uninstall":
        payload = uninstall()
    else:
        payload = status()

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
