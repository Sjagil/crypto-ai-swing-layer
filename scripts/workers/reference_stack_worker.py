#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def _source_paths(source: Path | None) -> list[Path]:
    if source is None or not source.is_dir():
        return []
    candidates = [source, source / "src"]
    return [path for path in candidates if path.is_dir()]


def _version(module_name: str, module) -> str | None:
    value = getattr(module, "__version__", None)
    if value:
        return str(value)
    dist_aliases = {
        "sb3_contrib": "sb3-contrib",
        "nautilus_trader": "nautilus_trader",
        "model": "kronos",
    }
    try:
        return importlib.metadata.version(
            dist_aliases.get(module_name, module_name)
        )
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--modules", default="")
    parser.add_argument("--source")
    parser.add_argument("--source-required", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).expanduser() if args.source else None
    source_paths = _source_paths(source)
    use_source_paths = bool(args.source_required)
    if use_source_paths:
        for path in reversed(source_paths):
            sys.path.insert(0, str(path))

    modules = tuple(
        value.strip()
        for value in args.modules.split(",")
        if value.strip()
    )
    attempts = []
    selected = None
    imported = None
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            attempts.append(
                {
                    "module": module_name,
                    "ready": False,
                    "error": f"{type(exc).__name__}:{str(exc)[:400]}",
                }
            )
            continue
        selected = module_name
        imported = module
        attempts.append(
            {
                "module": module_name,
                "ready": True,
                "version": _version(module_name, module),
                "file": str(getattr(module, "__file__", "") or ""),
            }
        )
        break

    if args.name == "optuna" and imported is not None:
        version = _version(selected or "optuna", imported)
        try:
            major = int(str(version).split(".", 1)[0])
        except (TypeError, ValueError):
            major = -1
        if major != 4:
            attempts.append(
                {
                    "module": selected,
                    "ready": False,
                    "error": f"OPTUNA_MAJOR_VERSION_UNSUPPORTED:{version}",
                }
            )
            imported = None
            selected = None

    module_ready = bool(imported is not None)
    source_ready = bool(source and source.is_dir())
    if modules and module_ready:
        status = "READY"
    elif modules:
        status = "MODULE_IMPORT_FAILED"
    elif source_ready:
        status = "SOURCE_READY"
    else:
        status = "MISSING"

    payload = {
        "schema_version": "crypto_ai_swing_reference_worker_v2",
        "name": args.name,
        "status": status,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "modules": list(modules),
        "selected_module": selected,
        "module_ready": module_ready,
        "module_attempts": attempts,
        "source": str(source) if source else None,
        "source_exists": source_ready,
        "source_paths_added": [str(path) for path in source_paths],
        "execution_authority": False,
        "live_decision_influence": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status in {"READY", "SOURCE_READY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
