from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class OptunaRuntimeError(RuntimeError):
    pass


def _stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _python_candidates(project_root: Path) -> list[Path]:
    root = Path(project_root).resolve()
    rows: list[Path] = []

    explicit_python = os.getenv("CRYPTO_SWING_OPTUNA_PYTHON", "").strip()
    if explicit_python:
        rows.append(Path(explicit_python).expanduser())

    configured_root = os.getenv("CRYPTO_SWING_REFERENCE_VENV_ROOT", "").strip()
    if configured_root:
        env = Path(configured_root).expanduser() / "optuna"
        rows.extend((env / "bin/python", env / "bin/python3", env / "Scripts/python.exe"))

    declared = ".venvs/optuna"
    config_path = root / "config" / "references.yaml"
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        declared = str(
            (((payload.get("references") or {}).get("optuna") or {}).get("isolated_env"))
            or declared
        )
    except (OSError, AttributeError, TypeError, yaml.YAMLError):
        declared = ".venvs/optuna"

    declared_path = Path(declared).expanduser()
    env_candidates: list[Path] = []
    if declared_path.is_absolute():
        env_candidates.append(declared_path)
    else:
        env_candidates.extend(
            (
                root / declared_path,
                root.parent / declared_path,
                root.parent / ".venvs" / "optuna",
            )
        )

    for env in env_candidates:
        rows.extend((env / "bin/python", env / "bin/python3", env / "Scripts/python.exe"))

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in rows:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _probe_python(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"ready": False, "python": str(path), "reason": "PYTHON_NOT_FOUND"}
    code = (
        "import json,sys,optuna;"
        "print(json.dumps({'python':sys.executable,'version':optuna.__version__}))"
    )
    proc = subprocess.run(
        [str(path), "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    if proc.returncode != 0:
        return {
            "ready": False,
            "python": str(path),
            "reason": "OPTUNA_IMPORT_FAILED",
            "stderr": proc.stderr[-500:],
        }
    try:
        payload = json.loads(proc.stdout.strip())
        version = str(payload.get("version") or "")
        major = int(version.split(".", 1)[0])
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
        return {
            "ready": False,
            "python": str(path),
            "reason": f"INVALID_PROBE_OUTPUT:{type(exc).__name__}",
        }
    if major != 4:
        return {
            "ready": False,
            "python": str(path),
            "version": version,
            "reason": "OPTUNA_VERSION_OUTSIDE_SUPPORTED_RANGE_4_X",
        }
    return {
        "ready": True,
        "python": str(path),
        "version": version,
        "reason": None,
    }


def resolve_optuna_python(project_root: Path) -> tuple[Path, dict[str, Any]]:
    probes = [_probe_python(path) for path in _python_candidates(project_root)]
    for probe in probes:
        if probe.get("ready"):
            return Path(str(probe["python"])), {
                "ready": True,
                "selected": probe,
                "candidates": probes,
            }
    raise OptunaRuntimeError(
        "No healthy isolated Optuna 4.x runtime found. Expected .venvs/optuna or "
        "CRYPTO_SWING_OPTUNA_PYTHON. Probes=" + json.dumps(probes, default=str)
    )


class OptunaStudyRuntime:
    SCHEMA = "crypto_ai_swing_optuna_runtime_v1"

    def __init__(
        self,
        project_root: Path,
        *,
        storage_path: Path | None = None,
        python_path: Path | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        if python_path is None:
            selected, probe = resolve_optuna_python(self.project_root)
            self.python_path = selected
            self.probe = probe
        else:
            # Keep the venv entrypoint path intact. Resolving this path dereferences
            # both a workspace .venvs/optuna symlink and bin/python itself, which can
            # turn a healthy venv interpreter into its base pyenv/python executable.
            provided_python = Path(python_path).expanduser()
            self.python_path = provided_python.absolute()
            selected_probe = _probe_python(self.python_path)
            if not selected_probe.get("ready"):
                raise OptunaRuntimeError(json.dumps(selected_probe, default=str))
            self.probe = {
                "ready": True,
                "selected": selected_probe,
                "candidates": [selected_probe],
            }

        self.worker_path = self.project_root / "scripts/workers/optuna_study_worker.py"
        if not self.worker_path.is_file():
            raise OptunaRuntimeError(f"Optuna worker missing: {self.worker_path}")

        default_storage = (
            self.project_root
            / "output/crypto_ai_swing/research/optuna/optuna_studies.sqlite3"
        )
        self.storage_path = Path(storage_path or default_storage).resolve()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def contract_hash(
        *,
        search_space: Mapping[str, Any],
        objective_version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        return _stable_hash(
            {
                "search_space": dict(search_space),
                "objective_version": str(objective_version),
                "metadata": dict(metadata or {}),
            }
        )

    def _run(self, action: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        request = {
            "schema_version": self.SCHEMA,
            "action": str(action),
            "storage_path": str(self.storage_path),
            **dict(payload or {}),
        }
        with tempfile.TemporaryDirectory(prefix="crypto-swing-optuna-") as tmp:
            request_path = Path(tmp) / "request.json"
            response_path = Path(tmp) / "response.json"
            request_path.write_text(
                json.dumps(request, sort_keys=True, default=str),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    str(self.python_path),
                    str(self.worker_path),
                    "--request",
                    str(request_path),
                    "--response",
                    str(response_path),
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
                env={
                    key: value
                    for key, value in os.environ.items()
                    if key
                    in {
                        "HOME",
                        "PATH",
                        "TMPDIR",
                        "LANG",
                        "LC_ALL",
                        "TZ",
                        "SSL_CERT_FILE",
                        "SSL_CERT_DIR",
                    }
                },
            )
            if not response_path.is_file():
                raise OptunaRuntimeError(
                    "Optuna worker produced no response. "
                    f"returncode={proc.returncode} stderr={proc.stderr[-1000:]}"
                )
            try:
                response = json.loads(response_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise OptunaRuntimeError(
                    f"Invalid Optuna worker response: {type(exc).__name__}"
                ) from exc
            if proc.returncode != 0 or response.get("ok") is not True:
                raise OptunaRuntimeError(
                    str(response.get("error") or proc.stderr[-1000:] or "OPTUNA_WORKER_FAILED")
                )
            result = response.get("payload")
            if not isinstance(result, dict):
                raise OptunaRuntimeError("Optuna worker payload is not an object")
            return result

    def status(self) -> dict[str, Any]:
        payload = self._run("status")
        return {
            "schema_version": self.SCHEMA,
            "ready": True,
            "python": str(self.python_path),
            "storage_path": str(self.storage_path),
            "probe": self.probe,
            "worker": payload,
        }

    def ensure_study(
        self,
        *,
        study_name: str,
        search_space: Mapping[str, Any],
        objective_version: str,
        metadata: Mapping[str, Any] | None = None,
        direction: str = "maximize",
        seed: int = 28,
    ) -> dict[str, Any]:
        contract = self.contract_hash(
            search_space=search_space,
            objective_version=objective_version,
            metadata=metadata,
        )
        return self._run(
            "ensure_study",
            {
                "study_name": study_name,
                "direction": direction,
                "seed": int(seed),
                "contract_hash": contract,
                "search_space": dict(search_space),
                "objective_version": str(objective_version),
                "metadata": dict(metadata or {}),
            },
        )

    def recover_running(self, *, study_name: str, contract_hash: str) -> dict[str, Any]:
        return self._run(
            "recover_running",
            {"study_name": study_name, "contract_hash": contract_hash},
        )

    def ask(
        self,
        *,
        study_name: str,
        contract_hash: str,
        search_space: Mapping[str, Any],
        seed: int = 28,
    ) -> dict[str, Any]:
        return self._run(
            "ask",
            {
                "study_name": study_name,
                "contract_hash": contract_hash,
                "search_space": dict(search_space),
                "seed": int(seed),
            },
        )

    def tell(
        self,
        *,
        study_name: str,
        contract_hash: str,
        trial_number: int,
        value: float,
    ) -> dict[str, Any]:
        return self._run(
            "tell",
            {
                "study_name": study_name,
                "contract_hash": contract_hash,
                "trial_number": int(trial_number),
                "value": float(value),
            },
        )

    def fail(
        self,
        *,
        study_name: str,
        contract_hash: str,
        trial_number: int,
        reason: str,
    ) -> dict[str, Any]:
        return self._run(
            "fail",
            {
                "study_name": study_name,
                "contract_hash": contract_hash,
                "trial_number": int(trial_number),
                "reason": str(reason)[:500],
            },
        )

    def summary(self, *, study_name: str, contract_hash: str) -> dict[str, Any]:
        return self._run(
            "summary",
            {"study_name": study_name, "contract_hash": contract_hash},
        )
