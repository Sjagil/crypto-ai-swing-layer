from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


class ReferenceStack:
    def __init__(self, project_root: Path) -> None:
        self.root = Path(project_root).resolve()
        path = self.root / "config" / "reference_stack.yaml"
        self.config = (
            yaml.safe_load(path.read_text(encoding="utf-8"))
            or {}
        )
        self.references = dict(
            self.config.get("references") or {}
        )
        self.worker = (
            self.root
            / "scripts"
            / "workers"
            / "reference_stack_worker.py"
        )

    @staticmethod
    def _python(env: Path) -> Path | None:
        for relative in (
            Path("bin/python"),
            Path("bin/python3"),
            Path("Scripts/python.exe"),
        ):
            candidate = env / relative
            if candidate.is_file():
                return candidate
        return None

    def _resolve_env(
        self,
        name: str,
        spec: dict[str, Any],
    ) -> Path | None:
        candidates = [self.root / ".venvs" / name]
        for alias in spec.get("env_aliases") or []:
            candidates.append(
                self.root / ".venvs" / str(alias)
            )
        for path in candidates:
            if path.is_dir():
                return path
        return None

    def _resolve_source(
        self,
        name: str,
        spec: dict[str, Any],
    ) -> Path | None:
        candidates = [self.root / "references" / name]
        for alias in spec.get("env_aliases") or []:
            candidates.append(
                self.root / "references" / str(alias)
            )
        for path in candidates:
            if path.is_dir():
                return path
        return None

    def probe(self, name: str) -> dict[str, Any]:
        spec = dict(self.references[name] or {})
        if spec.get("applicable", True) is False:
            return {
                "name": name,
                "role": spec.get("role"),
                "status": "NOT_APPLICABLE",
                "ready": True,
                "execution_authority": False,
                "live_decision_influence": False,
                "orders_generated": 0,
                "orders_submitted": 0,
            }

        env = self._resolve_env(name, spec)
        source = self._resolve_source(name, spec)
        python = self._python(env) if env else None
        module = spec.get("module")
        runtime_required = bool(
            spec.get("runtime_required")
        )
        source_required = bool(
            spec.get("source_required")
        )

        if python is None and runtime_required:
            return {
                "name": name,
                "role": spec.get("role"),
                "status": "RUNTIME_MISSING",
                "ready": False,
                "runtime_required": True,
                "env": str(env) if env else None,
                "source": (
                    str(source)
                    if source
                    else None
                ),
                "execution_authority": False,
                "live_decision_influence": False,
                "orders_generated": 0,
                "orders_submitted": 0,
            }

        selected_python = (
            python if python is not None else Path(sys.executable)
        )
        cmd = [
            str(selected_python),
            str(self.worker),
            "--name",
            name,
        ]
        if module:
            cmd.extend(["--module", str(module)])
        if source:
            cmd.extend(["--source", str(source)])

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        try:
            worker = json.loads(proc.stdout)
        except json.JSONDecodeError:
            worker = {
                "status": "INVALID_WORKER_RESPONSE",
                "error": proc.stderr[-500:],
            }

        module_ready = (
            worker.get("module_ready") is True
        )
        source_ready = (
            worker.get("source_exists") is True
        )
        if runtime_required:
            ready = module_ready
        elif source_required:
            ready = source_ready
        else:
            ready = bool(
                module_ready
                or source_ready
                or worker.get("status") == "SOURCE_READY"
            )

        return {
            "name": name,
            "role": spec.get("role"),
            "status": (
                "READY"
                if ready
                else str(
                    worker.get("status")
                    or "BLOCKED"
                )
            ),
            "ready": ready,
            "runtime_required": runtime_required,
            "source_required": source_required,
            "env": str(env) if env else None,
            "source": str(source) if source else None,
            "worker": worker,
            "execution_authority": False,
            "live_decision_influence": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    def status(self) -> dict[str, Any]:
        rows = [
            self.probe(name)
            for name in self.references
        ]
        applicable = [
            row
            for row in rows
            if row.get("status") != "NOT_APPLICABLE"
        ]
        required = [
            row
            for row in applicable
            if row.get("runtime_required")
            or row.get("source_required")
        ]
        ready = all(row["ready"] for row in required)
        return {
            "schema_version": (
                "crypto_ai_swing_reference_stack_status_v1"
            ),
            "status": "READY" if ready else "PARTIAL",
            "ready": ready,
            "references": rows,
            "summary": {
                "configured": len(rows),
                "applicable": len(applicable),
                "required": len(required),
                "required_ready": sum(
                    bool(row["ready"])
                    for row in required
                ),
            },
            "policy": dict(
                self.config.get("policy") or {}
            ),
            "orders_generated": 0,
            "orders_submitted": 0,
        }


__all__ = ["ReferenceStack"]
