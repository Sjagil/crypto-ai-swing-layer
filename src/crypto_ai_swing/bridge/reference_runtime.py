from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROBE_MODULE_BY_REFERENCE: dict[str, str] = {
    "nautilus_trader": "nautilus_trader",
    "vectorbt": "vectorbt",
    "optuna": "optuna",
    "skfolio": "skfolio",
    "stable_baselines3_contrib": "sb3_contrib",
    "qlib": "qlib",
    "pybroker": "pybroker",
    "vnpy": "vnpy",
}


def _resolve_env_path(project_root: Path, name: str, spec: dict[str, Any]) -> tuple[Path, str]:
    declared = str(spec.get("isolated_env") or "").strip()
    if declared:
        path = Path(declared)
        return (
            path if path.is_absolute() else project_root / path,
            "DECLARED",
        )

    inferred = project_root / ".venvs" / name
    if inferred.is_dir():
        return inferred, "INFERRED_FROM_EXISTING_ENV"
    return inferred, "UNDECLARED"


def _resolve_python(env_path: Path) -> Path | None:
    candidates = (
        env_path / "bin" / "python",
        env_path / "bin" / "python3",
        env_path / "Scripts" / "python.exe",
    )
    return next((path for path in candidates if path.is_file()), None)


def _probe_python(python_path: Path, module: str | None) -> dict[str, Any]:
    code = (
        "import importlib.util,json,platform,sys;"
        f"m={module!r};"
        "print(json.dumps({'python':sys.version.split()[0],"
        "'executable':sys.executable,'platform':platform.platform(),"
        "'module':m,'module_found':None if m is None else importlib.util.find_spec(m) is not None}))"
    )
    try:
        proc = subprocess.run(
            [str(python_path), "-c", code],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "healthy": False,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }

    if proc.returncode != 0:
        return {
            "healthy": False,
            "returncode": proc.returncode,
            "stderr": proc.stderr[-500:],
        }
    try:
        payload = json.loads(proc.stdout.strip())
    except Exception as exc:  # noqa: BLE001
        return {
            "healthy": False,
            "error": f"INVALID_PROBE_JSON:{type(exc).__name__}",
            "stdout": proc.stdout[-500:],
        }
    return {"healthy": True, **payload}


def reference_environment_status(
    project_root: Path,
    *,
    include_disabled: bool = True,
) -> dict[str, Any]:
    """Audit reference repositories and their isolated Python environments."""

    root = Path(project_root).resolve()
    config_path = root / "config" / "references.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    refs = dict(cfg.get("references") or {})

    rows: list[dict[str, Any]] = []
    known_env_names: set[str] = set()
    for name, raw_spec in refs.items():
        spec = dict(raw_spec or {})
        enabled = bool(spec.get("enabled", False))
        if not include_disabled and not enabled:
            continue

        env_path, env_binding = _resolve_env_path(root, name, spec)
        known_env_names.add(name)
        known_env_names.add(env_path.name)
        python_path = _resolve_python(env_path)

        repository = str(spec.get("repository") or "").strip()
        directory = str(spec.get("directory") or "").strip()
        repo_path = root / directory if directory else None
        repo_exists = bool(repo_path and repo_path.is_dir())
        env_exists = env_path.is_dir()
        probe_module = PROBE_MODULE_BY_REFERENCE.get(name)
        probe = (
            _probe_python(python_path, probe_module)
            if python_path is not None
            else {
                "healthy": False,
                "error": "PYTHON_INTERPRETER_MISSING",
                "module": probe_module,
            }
        )

        if repo_exists and probe.get("healthy"):
            status = "READY_ISOLATED"
        elif repo_exists and not env_exists:
            status = "REPO_PRESENT_ENV_MISSING"
        elif repo_exists:
            status = "REPO_PRESENT_ENV_UNHEALTHY"
        elif probe.get("healthy"):
            status = "ENV_READY_REPO_MISSING"
        else:
            status = "MISSING_OR_UNHEALTHY"

        rows.append(
            {
                "name": name,
                "enabled": enabled,
                "role": spec.get("role"),
                "repository": repository or None,
                "repo_path": str(repo_path) if repo_path else None,
                "repo_exists": repo_exists,
                "isolated_env": str(env_path),
                "env_binding": env_binding,
                "env_exists": env_exists,
                "python": str(python_path) if python_path else None,
                "probe_module": probe_module,
                "probe": probe,
                "status": status,
            }
        )

    env_root = root / ".venvs"
    detected_envs = (
        sorted(path.name for path in env_root.iterdir() if path.is_dir())
        if env_root.is_dir()
        else []
    )
    unregistered_envs = [
        name for name in detected_envs if name not in known_env_names
    ]

    enabled_rows = [row for row in rows if row["enabled"]]
    return {
        "schema_version": "crypto_ai_swing_reference_runtime_status_v1",
        "config": str(config_path),
        "references": rows,
        "summary": {
            "configured": len(rows),
            "enabled": len(enabled_rows),
            "enabled_ready_isolated": sum(
                row["enabled"] and row["status"] == "READY_ISOLATED"
                for row in rows
            ),
            "enabled_repo_missing": sum(
                row["enabled"] and not row["repo_exists"]
                for row in rows
            ),
            "enabled_env_missing_or_unhealthy": sum(
                row["enabled"] and not bool(row["probe"].get("healthy"))
                for row in rows
            ),
        },
        "detected_envs": detected_envs,
        "unregistered_envs": unregistered_envs,
        "policy": {
            "site_packages_must_remain_isolated": True,
            "reference_processes_may_not_submit_exchange_orders": True,
            "reference_outputs_are_challenger_evidence_only": True,
        },
        "orders_generated": 0,
        "orders_submitted": 0,
    }
