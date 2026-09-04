from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from crypto_ai_swing.bridge.reference_runtime import reference_environment_status

PACKAGE_DEFAULTS: dict[str, str] = {
    "nautilus_trader": "nautilus_trader",
    "vectorbt": "vectorbt>=0.28,<1",
    "optuna": "optuna>=4,<5",
    "skfolio": "skfolio>=0.12,<1",
    "stable_baselines3_contrib": "sb3-contrib>=2.3,<3",
}


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 1800) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "ok": proc.returncode == 0,
    }


def _python_in(env: Path) -> Path:
    candidates = (
        env / "bin" / "python",
        env / "bin" / "python3",
        env / "Scripts" / "python.exe",
    )
    return next((p for p in candidates if p.is_file()), candidates[0])


def provision_references(
    project_root: Path,
    *,
    venv_root: Path,
    repo_root: Path | None = None,
    names: list[str] | None = None,
    apply: bool = False,
    include_disabled: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    venv_root = Path(venv_root).expanduser().resolve()
    repo_root = Path(repo_root).expanduser().resolve() if repo_root else root / "references"

    refs_cfg = yaml.safe_load((root / "config/references.yaml").read_text(encoding="utf-8")) or {}
    provision_cfg = yaml.safe_load(
        (root / "config/reference_provision.yaml").read_text(encoding="utf-8")
    ) or {}
    refs = dict(refs_cfg.get("references") or {})
    raw_packages = dict(provision_cfg.get("packages") or {})
    package_specs = {
        **PACKAGE_DEFAULTS,
        **{
            key: (value.get("pip") if isinstance(value, dict) else value)
            for key, value in raw_packages.items()
        },
    }
    compatibility_cfg = dict(provision_cfg.get("compatibility") or {})
    defaults = dict(provision_cfg.get("defaults") or {})
    install_timeout = int(defaults.get("install_timeout_seconds", 1800))
    clone_timeout = int(defaults.get("clone_timeout_seconds", 900))
    clone_depth = int(defaults.get("clone_depth", 1))
    base_python = str(defaults.get("python") or sys.executable or "python3")

    requested = set(names or [])
    rows: list[dict[str, Any]] = []
    venv_root.mkdir(parents=True, exist_ok=True)
    repo_root.mkdir(parents=True, exist_ok=True)

    for name, raw in refs.items():
        spec = dict(raw or {})
        enabled = bool(spec.get("enabled", False))
        if requested and name not in requested:
            continue
        if not include_disabled and not enabled:
            continue

        env_name = Path(str(spec.get("isolated_env") or f".venvs/{name}")).name
        env_path = venv_root / env_name
        python_path = _python_in(env_path)
        repository = str(spec.get("repository") or "").strip()
        directory = Path(str(spec.get("directory") or name)).name
        source_path = repo_root / directory
        package_requirement = package_specs.get(name)
        compatibility_requirements = [
            str(value)
            for value in (compatibility_cfg.get(name) or [])
            if str(value).strip()
        ]
        actions: list[dict[str, Any]] = []

        if not env_path.exists():
            command = [base_python, "-m", "venv", str(env_path)]
            action = {"kind": "create_env", "target": str(env_path), "command": command}
            if apply:
                action["result"] = _run(command, timeout=install_timeout)
            actions.append(action)
        if apply:
            python_path = _python_in(env_path)

        if package_requirement:
            command = [str(python_path), "-m", "pip", "install", package_requirement]
            action = {
                "kind": "install_package",
                "requirement": package_requirement,
                "command": command,
            }
            if apply and python_path.is_file():
                action["result"] = _run(command, timeout=install_timeout)
            actions.append(action)

        for compatibility_requirement in compatibility_requirements:
            command = [
                str(python_path), "-m", "pip", "install",
                compatibility_requirement,
            ]
            action = {
                "kind": "install_compatibility",
                "requirement": compatibility_requirement,
                "command": command,
            }
            if apply and python_path.is_file():
                action["result"] = _run(command, timeout=install_timeout)
            actions.append(action)

        if repository and not source_path.exists():
            command = [
                "git", "clone", "--depth", str(clone_depth), repository, str(source_path)
            ]
            action = {
                "kind": "clone_source",
                "repository": repository,
                "target": str(source_path),
                "command": command,
            }
            if apply:
                action["result"] = _run(command, cwd=repo_root, timeout=clone_timeout)
            actions.append(action)

        rows.append(
            {
                "name": name,
                "enabled": enabled,
                "env_path": str(env_path),
                "source_path": str(source_path),
                "package_requirement": package_requirement,
                "compatibility_requirements": compatibility_requirements,
                "repository": repository or None,
                "actions": actions,
            }
        )

    status = reference_environment_status(
        root,
        include_disabled=include_disabled,
        venv_roots=[venv_root],
        repo_roots=[repo_root],
    )
    return {
        "schema_version": "crypto_ai_swing_reference_provision_v1",
        "apply": apply,
        "venv_root": str(venv_root),
        "repo_root": str(repo_root),
        "references": rows,
        "post_status": status,
        "policy": {
            "isolated_envs_only": True,
            "reference_processes_are_research_only": True,
            "reference_processes_may_not_submit_orders": True,
        },
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def write_provision_report(payload: dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output
