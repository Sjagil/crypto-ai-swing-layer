#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def version_for(module_name: str, module) -> str | None:
    value = getattr(module, "__version__", None)
    if value:
        return str(value)
    candidates = {
        "sb3_contrib": "sb3-contrib",
        "nautilus_trader": "nautilus_trader",
    }
    try:
        return importlib.metadata.version(
            candidates.get(module_name, module_name)
        )
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--module")
    parser.add_argument("--source")
    args = parser.parse_args()

    source = (
        Path(args.source).expanduser()
        if args.source
        else None
    )
    payload = {
        "schema_version": "crypto_ai_swing_reference_worker_v1",
        "name": args.name,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "module": args.module or None,
        "source": str(source) if source else None,
        "source_exists": bool(source and source.is_dir()),
        "execution_authority": False,
        "live_decision_influence": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }

    if args.module:
        try:
            module = importlib.import_module(args.module)
        except Exception as exc:
            payload.update(
                {
                    "status": "MODULE_IMPORT_FAILED",
                    "module_ready": False,
                    "error": (
                        f"{type(exc).__name__}:"
                        f"{str(exc)[:400]}"
                    ),
                }
            )
        else:
            payload.update(
                {
                    "status": "READY",
                    "module_ready": True,
                    "version": version_for(
                        args.module,
                        module,
                    ),
                    "module_file": str(
                        getattr(module, "__file__", "")
                        or ""
                    ),
                }
            )
    elif source and source.is_dir():
        payload.update(
            {
                "status": "SOURCE_READY",
                "module_ready": False,
            }
        )
    else:
        payload.update(
            {
                "status": "MISSING",
                "module_ready": False,
            }
        )

    print(json.dumps(payload, indent=2, sort_keys=True))
    return (
        0
        if payload["status"] in {"READY", "SOURCE_READY"}
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
