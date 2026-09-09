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
        config_path = self.root / "config" / "reference_stack.yaml"
        self.config = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        ) or {}
        self.references = dict(self.config.get("references") or {})
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

    def _resolve(
        self,
        bucket: str,
        name: str,
        aliases: list[str],
    ) -> Path | None:
        base = self.root / bucket
        candidates = [base / name, *(base / alias for alias in aliases)]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
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

        aliases = [str(v) for v in spec.get("env_aliases") or []]
        env = self._resolve(".venvs", name, aliases)
        source = self._resolve("references", name, aliases)
        python = self._python(env) if env else None
        modules = [str(v) for v in spec.get("modules") or []]
        runtime_required = bool(spec.get("runtime_required"))
        source_required = bool(spec.get("source_required"))

        if runtime_required and python is None:
            return {
                "name": name,
                "role": spec.get("role"),
                "status": "RUNTIME_MISSING",
                "ready": False,
                "runtime_required": True,
                "source_required": source_required,
                "env": str(env) if env else None,
                "source": str(source) if source else None,
                "execution_authority": False,
                "live_decision_influence": False,
                "orders_generated": 0,
                "orders_submitted": 0,
            }

        selected_python = python or Path(sys.executable)
        cmd = [
            str(selected_python),
            str(self.worker),
            "--name",
            name,
            "--modules",
            ",".join(modules),
        ]
        if source:
            cmd.extend(["--source", str(source)])
        if source_required:
            cmd.append("--source-required")
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        try:
            worker = json.loads(process.stdout)
        except json.JSONDecodeError:
            worker = {
                "status": "INVALID_WORKER_RESPONSE",
                "module_ready": False,
                "source_exists": bool(source),
                "error": process.stderr[-1000:],
            }

        module_ready = worker.get("module_ready") is True
        source_ready = worker.get("source_exists") is True
        runtime_ok = (not runtime_required) or module_ready
        source_ok = (not source_required) or source_ready
        ready = bool(runtime_ok and source_ok)

        return {
            "name": name,
            "role": spec.get("role"),
            "status": "READY" if ready else str(
                worker.get("status") or "BLOCKED"
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
        rows = [self.probe(name) for name in self.references]
        applicable = [
            row for row in rows
            if row.get("status") != "NOT_APPLICABLE"
        ]
        required = [
            row for row in applicable
            if row.get("runtime_required") or row.get("source_required")
        ]
        ready = all(row["ready"] for row in required)
        return {
            "schema_version": "crypto_ai_swing_reference_stack_status_v2",
            "status": "READY" if ready else "PARTIAL",
            "ready": ready,
            "references": rows,
            "summary": {
                "configured": len(rows),
                "applicable": len(applicable),
                "required": len(required),
                "required_ready": sum(
                    bool(row["ready"]) for row in required
                ),
            },
            "policy": dict(self.config.get("policy") or {}),
            "orders_generated": 0,
            "orders_submitted": 0,
        }


__all__ = ["ReferenceStack"]
