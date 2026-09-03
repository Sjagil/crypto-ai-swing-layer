from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterable
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

REFERENCE_VENV_ROOT_ENV = "CRYPTO_SWING_REFERENCE_VENV_ROOT"
REFERENCE_REPO_ROOT_ENV = "CRYPTO_SWING_REFERENCE_REPO_ROOT"


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    rows: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        path = Path(raw).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        rows.append(path)
    return rows


def _candidate_venv_roots(
    project_root: Path,
    explicit: Iterable[Path] | None = None,
) -> list[Path]:
    roots: list[Path] = []
    if explicit:
        roots.extend(Path(path).expanduser() for path in explicit)
    configured = os.getenv(REFERENCE_VENV_ROOT_ENV, "").strip()
    if configured:
        roots.append(Path(configured).expanduser())
    roots.extend(
        (
            project_root / ".venvs",
            project_root.parent / ".venvs",
        )
    )
    return _dedupe_paths(roots)


def _candidate_repo_roots(
    project_root: Path,
    explicit: Iterable[Path] | None = None,
) -> list[Path]:
    roots: list[Path] = []
    if explicit:
        roots.extend(Path(path).expanduser() for path in explicit)
    configured = os.getenv(REFERENCE_REPO_ROOT_ENV, "").strip()
    if configured:
        roots.append(Path(configured).expanduser())
    roots.extend(
        (
            project_root,
            project_root.parent,
        )
    )
    return _dedupe_paths(roots)


def _resolve_env_path(
    project_root: Path,
    name: str,
    spec: dict[str, Any],
    *,
    venv_roots: Iterable[Path],
) -> tuple[Path, str, list[str]]:
    roots = list(venv_roots)
    declared = str(spec.get("isolated_env") or "").strip()
    candidates: list[Path] = []
    project_declared: Path | None = None

    if declared:
        declared_path = Path(declared).expanduser()
        if declared_path.is_absolute():
            candidates.append(declared_path)
            binding = "DECLARED_ABSOLUTE"
        else:
            project_declared = project_root / declared_path
            candidates.append(project_declared)
            parts = declared_path.parts
            if parts and parts[0] == ".venvs":
                suffix = Path(*parts[1:]) if len(parts) > 1 else Path(name)
                candidates.extend(root / suffix for root in roots)
            else:
                candidates.extend(root / declared_path.name for root in roots)
            binding = "DECLARED_RELATIVE"
    else:
        candidates.extend(root / name for root in roots)
        binding = "INFERRED"

    candidates = _dedupe_paths(candidates)
    selected = next((path for path in candidates if path.is_dir()), None)
    if selected is not None:
        if project_declared is not None and selected == project_declared:
            resolved_binding = "DECLARED_PROJECT_LOCAL"
        elif declared:
            resolved_binding = "DECLARED_WORKSPACE_RESOLVED"
        else:
            resolved_binding = "INFERRED_EXISTING"
        return selected, resolved_binding, [str(path) for path in candidates]

    fallback = candidates[0] if candidates else project_root / ".venvs" / name
    return fallback, f"{binding}_MISSING", [str(path) for path in candidates]


def _resolve_repo_path(
    project_root: Path,
    spec: dict[str, Any],
    *,
    repo_roots: Iterable[Path],
) -> tuple[Path | None, str, list[str]]:
    directory = str(spec.get("directory") or "").strip()
    if not directory:
        return None, "UNDECLARED", []

    declared = Path(directory).expanduser()
    project_declared: Path | None = None
    if declared.is_absolute():
        candidates = [declared]
        binding = "DECLARED_ABSOLUTE"
    else:
        project_declared = project_root / declared
        basename = declared.name
        candidates = [project_declared]
        for root in repo_roots:
            candidates.append(root / declared)
            candidates.append(root / basename)
            candidates.append(root / "references" / basename)
        candidates = _dedupe_paths(candidates)
        binding = "DECLARED_RELATIVE"

    selected = next((path for path in candidates if path.is_dir()), None)
    if selected is not None:
        resolved_binding = (
            "DECLARED_PROJECT_LOCAL"
            if project_declared is not None and selected == project_declared
            else "DECLARED_WORKSPACE_RESOLVED"
        )
        return selected, resolved_binding, [str(path) for path in candidates]

    return candidates[0], f"{binding}_MISSING", [str(path) for path in candidates]


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
        "found=None if m is None else importlib.util.find_spec(m) is not None;"
        "print(json.dumps({'python':sys.version.split()[0],"
        "'executable':sys.executable,'platform':platform.platform(),"
        "'module':m,'module_found':found}))"
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
            "process_healthy": False,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "module": module,
        }

    if proc.returncode != 0:
        return {
            "healthy": False,
            "process_healthy": False,
            "returncode": proc.returncode,
            "stderr": proc.stderr[-500:],
            "module": module,
        }

    try:
        payload = json.loads(proc.stdout.strip())
    except Exception as exc:  # noqa: BLE001
        return {
            "healthy": False,
            "process_healthy": True,
            "error": f"INVALID_PROBE_JSON:{type(exc).__name__}",
            "stdout": proc.stdout[-500:],
            "module": module,
        }

    module_found = payload.get("module_found")
    healthy = bool(module is None or module_found is True)
    return {
        "healthy": healthy,
        "process_healthy": True,
        **payload,
        "error": None if healthy else "PROBE_MODULE_NOT_FOUND",
    }


def reference_environment_status(
    project_root: Path,
    *,
    include_disabled: bool = True,
    venv_roots: Iterable[Path] | None = None,
    repo_roots: Iterable[Path] | None = None,
) -> dict[str, Any]:
    """Audit isolated reference runtimes across project and workspace roots."""

    root = Path(project_root).resolve()
    config_path = root / "config" / "references.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    refs = dict(cfg.get("references") or {})

    resolved_venv_roots = _candidate_venv_roots(root, venv_roots)
    resolved_repo_roots = _candidate_repo_roots(root, repo_roots)

    rows: list[dict[str, Any]] = []
    known_env_names: set[str] = set()
    for name, raw_spec in refs.items():
        spec = dict(raw_spec or {})
        enabled = bool(spec.get("enabled", False))
        if not include_disabled and not enabled:
            continue

        env_path, env_binding, env_candidates = _resolve_env_path(
            root,
            name,
            spec,
            venv_roots=resolved_venv_roots,
        )
        known_env_names.add(name)
        known_env_names.add(env_path.name)
        python_path = _resolve_python(env_path)

        repo_path, repo_binding, repo_candidates = _resolve_repo_path(
            root,
            spec,
            repo_roots=resolved_repo_roots,
        )
        repository = str(spec.get("repository") or "").strip()
        repo_exists = bool(repo_path and repo_path.is_dir())
        env_exists = env_path.is_dir()
        probe_module = PROBE_MODULE_BY_REFERENCE.get(name)
        probe = (
            _probe_python(python_path, probe_module)
            if python_path is not None
            else {
                "healthy": False,
                "process_healthy": False,
                "error": "PYTHON_INTERPRETER_MISSING",
                "module": probe_module,
                "module_found": None,
            }
        )

        package_runtime_ready = bool(
            probe.get("healthy") and probe_module is not None
        )
        source_runtime_ready = bool(
            probe.get("process_healthy") and repo_exists
        )
        runtime_ready = bool(package_runtime_ready or source_runtime_ready)

        if source_runtime_ready and package_runtime_ready:
            status = "READY_SOURCE_AND_PACKAGE_ISOLATED"
        elif source_runtime_ready:
            status = "READY_SOURCE_ISOLATED"
        elif package_runtime_ready:
            status = "READY_PACKAGE_ISOLATED"
        elif repo_exists and not env_exists:
            status = "REPO_PRESENT_ENV_MISSING"
        elif repo_exists:
            status = "REPO_PRESENT_ENV_UNHEALTHY"
        elif probe.get("process_healthy") and probe_module is None:
            status = "ENV_READY_SOURCE_REQUIRED"
        elif env_exists:
            status = "ENV_PRESENT_MODULE_UNHEALTHY"
        else:
            status = "MISSING_OR_UNHEALTHY"

        rows.append(
            {
                "name": name,
                "enabled": enabled,
                "role": spec.get("role"),
                "repository": repository or None,
                "repo_path": str(repo_path) if repo_path else None,
                "repo_binding": repo_binding,
                "repo_candidates": repo_candidates,
                "repo_exists": repo_exists,
                "isolated_env": str(env_path),
                "env_binding": env_binding,
                "env_candidates": env_candidates,
                "env_exists": env_exists,
                "python": str(python_path) if python_path else None,
                "probe_module": probe_module,
                "probe": probe,
                "package_runtime_ready": package_runtime_ready,
                "source_runtime_ready": source_runtime_ready,
                "runtime_ready": runtime_ready,
                "status": status,
            }
        )

    detected_env_paths: list[str] = []
    detected_env_names: list[str] = []
    for env_root in resolved_venv_roots:
        if not env_root.is_dir():
            continue
        for path in sorted(
            (item for item in env_root.iterdir() if item.is_dir()),
            key=lambda item: item.name,
        ):
            detected_env_paths.append(str(path))
            detected_env_names.append(path.name)

    detected_env_paths = list(dict.fromkeys(detected_env_paths))
    detected_env_names = list(dict.fromkeys(detected_env_names))
    unregistered_envs = [
        name for name in detected_env_names if name not in known_env_names
    ]

    enabled_rows = [row for row in rows if row["enabled"]]
    return {
        "schema_version": "crypto_ai_swing_reference_runtime_status_v2",
        "config": str(config_path),
        "venv_roots": [str(path) for path in resolved_venv_roots],
        "repo_roots": [str(path) for path in resolved_repo_roots],
        "references": rows,
        "summary": {
            "configured": len(rows),
            "enabled": len(enabled_rows),
            "enabled_runtime_ready": sum(
                bool(row["runtime_ready"]) for row in enabled_rows
            ),
            "enabled_source_runtime_ready": sum(
                bool(row["source_runtime_ready"]) for row in enabled_rows
            ),
            "enabled_package_runtime_ready": sum(
                bool(row["package_runtime_ready"]) for row in enabled_rows
            ),
            "enabled_repo_missing": sum(
                not bool(row["repo_exists"]) for row in enabled_rows
            ),
            "enabled_env_missing_or_unhealthy": sum(
                not bool(row["probe"].get("process_healthy"))
                for row in enabled_rows
            ),
            "enabled_module_probe_failed": sum(
                row["probe_module"] is not None
                and row["probe"].get("module_found") is not True
                for row in enabled_rows
            ),
        },
        "detected_envs": detected_env_names,
        "detected_env_paths": detected_env_paths,
        "unregistered_envs": unregistered_envs,
        "policy": {
            "site_packages_must_remain_isolated": True,
            "reference_processes_may_not_submit_exchange_orders": True,
            "reference_outputs_are_challenger_evidence_only": True,
            "package_ready_does_not_grant_execution_authority": True,
        },
        "orders_generated": 0,
        "orders_submitted": 0,
    }
