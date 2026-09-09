from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import optuna
from optuna.trial import TrialState


class WorkerError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise WorkerError("request must be a JSON object")
    return payload


def _write(
    path: Path,
    *,
    ok: bool,
    payload: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ok": bool(ok),
                "payload": dict(payload or {}),
                "error": error,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )


def _storage_url(path: str) -> str:
    selected = Path(path).expanduser().resolve()
    selected.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{selected}"


def _lock_path(storage_path: str) -> Path:
    selected = Path(storage_path).expanduser().resolve()
    return selected.with_suffix(selected.suffix + ".lock")


class StudyLock:
    def __init__(self, path: Path, *, stale_seconds: int = 3600) -> None:
        self.path = path
        self.stale_seconds = int(stale_seconds)
        self.acquired = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(100):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    if age > self.stale_seconds:
                        self.path.unlink(missing_ok=True)
                        continue
                except Exception:
                    pass
                time.sleep(0.05)
                continue
            else:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps({"pid": os.getpid(), "created_at": time.time()}))
                self.acquired = True
                return self
        raise WorkerError(f"study lock busy: {self.path}")

    def __exit__(self, exc_type, exc, tb):
        if self.acquired:
            self.path.unlink(missing_ok=True)
        self.acquired = False


def _sampler(seed: int, trial_count: int = 0):
    return optuna.samplers.TPESampler(
        seed=int(seed) + int(trial_count),
        multivariate=True,
        group=True,
        constant_liar=True,
    )


def _create_or_load(request: Mapping[str, Any], *, sampler_seed: int | None = None):
    study_name = str(request.get("study_name") or "").strip()
    if not study_name:
        raise WorkerError("study_name is required")
    direction = str(request.get("direction") or "maximize")
    if direction not in {"maximize", "minimize"}:
        raise WorkerError("direction must be maximize or minimize")
    storage = _storage_url(str(request["storage_path"]))
    seed = int(request.get("seed", 28))
    initial_sampler = _sampler(seed if sampler_seed is None else sampler_seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=initial_sampler,
        load_if_exists=True,
    )
    if sampler_seed is None:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=_sampler(seed, len(study.trials)),
        )
    return study


def _validate_contract(study, expected: str) -> None:
    actual = str(study.user_attrs.get("contract_hash") or "")
    if not actual:
        study.set_user_attr("contract_hash", expected)
        return
    if actual != expected:
        raise WorkerError(
            "OPTUNA_STUDY_CONTRACT_MISMATCH: "
            f"expected={expected} actual={actual}"
        )


def _suggest(trial, name: str, spec: Mapping[str, Any]):
    kind = str(spec.get("kind") or "").lower()
    if kind == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        log = bool(spec.get("log", False))
        step = spec.get("step")
        if not math.isfinite(low) or not math.isfinite(high) or high < low:
            raise WorkerError(f"invalid float search space for {name}")
        if log and step is not None:
            raise WorkerError(f"float space {name} cannot use log and step together")
        return trial.suggest_float(
            name,
            low,
            high,
            step=(float(step) if step is not None else None),
            log=log,
        )
    if kind == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        if high < low or step < 1:
            raise WorkerError(f"invalid int search space for {name}")
        return trial.suggest_int(name, low, high, step=step, log=bool(spec.get("log", False)))
    if kind == "categorical":
        choices = list(spec.get("choices") or [])
        if not choices:
            raise WorkerError(f"categorical space {name} has no choices")
        return trial.suggest_categorical(name, choices)
    raise WorkerError(f"unsupported search space kind for {name}: {kind}")


def _trial_rows(study) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        rows.append(
            {
                "number": int(trial.number),
                "state": trial.state.name,
                "value": trial.value,
                "params": dict(trial.params),
            }
        )
    return rows


def _best(study) -> dict[str, Any] | None:
    completed = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not completed:
        return None
    trial = study.best_trial
    return {
        "number": int(trial.number),
        "value": float(trial.value),
        "params": dict(trial.params),
    }


def handle(request: Mapping[str, Any]) -> dict[str, Any]:
    action = str(request.get("action") or "")
    if action == "status":
        return {
            "status": "READY",
            "optuna_version": optuna.__version__,
            "worker_pid": os.getpid(),
            "authority": "RESEARCH_ONLY",
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    storage_path = str(request.get("storage_path") or "")
    if not storage_path:
        raise WorkerError("storage_path is required")
    contract_hash = str(request.get("contract_hash") or "").strip()
    if not contract_hash:
        raise WorkerError("contract_hash is required")

    with StudyLock(_lock_path(storage_path)):
        if action == "ensure_study":
            study = _create_or_load(request)
            _validate_contract(study, contract_hash)
            if "search_space" not in study.user_attrs:
                study.set_user_attr("search_space", dict(request.get("search_space") or {}))
            if "objective_version" not in study.user_attrs:
                study.set_user_attr(
                    "objective_version",
                    str(request.get("objective_version") or ""),
                )
            if "metadata" not in study.user_attrs:
                study.set_user_attr("metadata", dict(request.get("metadata") or {}))
            return {
                "status": "READY",
                "study_name": study.study_name,
                "contract_hash": contract_hash,
                "trial_count": len(study.trials),
                "best": _best(study),
            }

        study = _create_or_load(request)
        _validate_contract(study, contract_hash)

        if action == "recover_running":
            recovered: list[int] = []
            for trial in study.trials:
                if trial.state == TrialState.RUNNING:
                    study.tell(trial.number, state=TrialState.FAIL, skip_if_finished=True)
                    recovered.append(int(trial.number))
            return {
                "status": "READY",
                "recovered_trial_numbers": recovered,
                "recovered_count": len(recovered),
            }

        if action == "ask":
            search_space = dict(request.get("search_space") or {})
            if not search_space:
                raise WorkerError("search_space is required")
            trial = study.ask()
            params = {
                name: _suggest(trial, name, dict(spec or {}))
                for name, spec in sorted(search_space.items())
            }
            return {
                "status": "RUNNING",
                "trial_number": int(trial.number),
                "params": params,
            }

        if action == "tell":
            number = int(request["trial_number"])
            value = float(request["value"])
            if not math.isfinite(value):
                raise WorkerError("objective value must be finite")
            frozen = study.tell(number, value, skip_if_finished=True)
            return {
                "status": frozen.state.name,
                "trial_number": int(frozen.number),
                "value": frozen.value,
                "best": _best(study),
            }

        if action == "fail":
            number = int(request["trial_number"])
            frozen = study.tell(number, state=TrialState.FAIL, skip_if_finished=True)
            return {
                "status": frozen.state.name,
                "trial_number": int(frozen.number),
                "reason": str(request.get("reason") or "")[:500],
            }

        if action == "summary":
            return {
                "status": "READY",
                "study_name": study.study_name,
                "contract_hash": contract_hash,
                "trial_count": len(study.trials),
                "best": _best(study),
                "trials": _trial_rows(study),
            }

    raise WorkerError(f"unsupported action: {action}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    response_path = Path(args.response)
    try:
        request = _read(Path(args.request))
        payload = handle(request)
    except Exception as exc:
        _write(
            response_path,
            ok=False,
            error=f"{type(exc).__name__}:{str(exc)[:2000]}",
        )
        return 0
    _write(response_path, ok=True, payload=payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
