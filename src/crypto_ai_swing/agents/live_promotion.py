from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, dict) else {}
    except Exception:
        return {}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        selected = float(value)
    except (TypeError, ValueError):
        return default
    return selected if selected == selected else default


class LiveModelGovernor:
    """Evidence-gated model promotion only.

    It may grant model influence after OOS qualification. It never grants
    exchange authority, changes the Pi executor, or increases hard capital caps.
    """

    def __init__(self, settings) -> None:
        self.settings = settings
        self.cfg = dict(settings.agents.get("live_promotion", {}) or {})
        self.root = settings.project_root / "output/crypto_ai_swing/agents"
        self.history = self.root / "promotion_history.jsonl"

    def _supervised_score(self, bundle: dict[str, Any]) -> float:
        m = dict(bundle.get("metrics") or {})
        return float(
            2.0 * _f(m.get("selected_conservative_mean_net"))
            + 1.5 * _f(m.get("selected_market_balanced_mean_net"))
            + 0.10 * (_f(m.get("selected_positive_market_fraction")) - 0.5)
            + 0.05
            * (
                max(
                    _f(m.get("alpha_auc"), 0.5),
                    _f(m.get("test_alpha_auc"), 0.5),
                )
                - 0.5
            )
            + 0.10
            * _f(dict(m.get("return_test") or {}).get("skill_vs_zero"))
        )

    def _supervised_eligible(
        self, bundle: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        failures: list[str] = []
        heads = dict(bundle.get("head_qualifications") or {})
        if not bool(
            bundle.get("shadow_decision_qualified")
            or heads.get("alpha")
            or heads.get("return")
        ):
            failures.append("NO_QUALIFIED_DIRECTIONAL_HEAD")

        metrics = dict(bundle.get("metrics") or {})
        if metrics.get("positive_oos_net_proxy") is not True:
            failures.append("POSITIVE_OOS_NET_NOT_CONFIRMED")
        if metrics.get("point_in_time_universe_qualified") is False:
            failures.append("POINT_IN_TIME_UNIVERSE_NOT_QUALIFIED")
        if int(bundle.get("dataset_rows") or 0) < int(
            self.cfg.get("minimum_supervised_rows", 8000)
        ):
            failures.append("INSUFFICIENT_SUPERVISED_ROWS")
        if not any(
            bool(heads.get(name))
            for name in ("alpha", "return", "regime", "risk")
        ):
            failures.append("NO_QUALIFIED_MODEL_HEAD")
        return not failures, failures

    def _current_supervised_score(self) -> float | None:
        pointer = _read_json(self.root / "live.pointer.json")
        artifact = Path(str(pointer.get("artifact_path") or ""))
        if not artifact.is_file():
            return None
        try:
            return self._supervised_score(dict(joblib.load(artifact)))
        except Exception:
            return None

    def promote_supervised(self) -> dict[str, Any]:
        if not bool(self.cfg.get("supervised_enabled", True)):
            return {"status": "DISABLED", "kind": "supervised"}

        source_pointer = _read_json(self.root / "latest.pointer.json")
        source_artifact = Path(
            str(source_pointer.get("artifact_path") or "")
        )
        if not source_artifact.is_file():
            return {"status": "NO_CANDIDATE", "kind": "supervised"}

        try:
            candidate = dict(joblib.load(source_artifact))
        except Exception as exc:
            return {
                "status": "CANDIDATE_LOAD_ERROR",
                "kind": "supervised",
                "error": f"{type(exc).__name__}:{str(exc)[:240]}",
            }

        eligible, failures = self._supervised_eligible(candidate)
        if not eligible:
            return {
                "status": "NOT_QUALIFIED",
                "kind": "supervised",
                "failures": failures,
            }

        score = self._supervised_score(candidate)
        current_score = self._current_supervised_score()
        source_hash = str(source_pointer.get("artifact_hash") or "")
        current = _read_json(self.root / "live.pointer.json")

        if source_hash and source_hash == current.get("source_artifact_hash"):
            return {
                "status": "CURRENT_CHAMPION",
                "kind": "supervised",
                "score": score,
            }

        minimum_delta = float(
            self.cfg.get("minimum_promotion_score_delta", 0.0)
        )
        if current_score is not None and score < current_score + minimum_delta:
            return {
                "status": "CHALLENGER_NOT_BETTER",
                "kind": "supervised",
                "candidate_score": score,
                "champion_score": current_score,
                "minimum_delta": minimum_delta,
            }

        promoted = copy.deepcopy(candidate)
        promoted.update(
            {
                "status": "CANARY",
                "live_decision_influence": True,
                "automatic_live_promotion": True,
                "promoted_at": _now(),
                "promotion_score": score,
                "promotion_scope": "MODEL_INFLUENCE_ONLY",
                "execution_authority_granted": False,
                "capital_authority_granted": False,
            }
        )

        identity = hashlib.sha256(
            json.dumps(
                {
                    "source": source_hash,
                    "dataset_id": promoted.get("dataset_id"),
                    "score": score,
                    "heads": promoted.get("head_qualifications"),
                },
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()

        directory = self.root / "live" / identity
        directory.mkdir(parents=True, exist_ok=True)
        artifact = directory / "bundle.joblib"
        joblib.dump(promoted, artifact)
        artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()

        manifest = {
            key: value
            for key, value in promoted.items()
            if key != "models"
        }
        manifest.update(
            {
                "artifact_path": str(artifact.resolve()),
                "artifact_hash": artifact_hash,
                "source_artifact_hash": source_hash,
            }
        )
        manifest_path = directory / "manifest.json"
        _atomic_json(manifest_path, manifest)

        pointer_path = self.root / "live.pointer.json"
        previous = self.root / "live.previous.pointer.json"
        if pointer_path.is_file():
            shutil.copy2(pointer_path, previous)

        pointer = {
            "schema_version": "swing_agent_live_pointer_v1",
            "artifact_path": str(artifact.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "artifact_hash": artifact_hash,
            "source_artifact_hash": source_hash,
            "status": "CANARY",
            "live_decision_influence": True,
            "head_qualifications": promoted.get("head_qualifications") or {},
            "promotion_score": score,
            "promoted_at": _now(),
            "automatic_live_promotion": True,
            "execution_authority_granted": False,
        }
        _atomic_json(pointer_path, pointer)
        self._record({"event": "SUPERVISED_PROMOTED", **pointer})
        return {
            **pointer,
            "model_status": pointer.get("status"),
            "status": "PROMOTED",
            "kind": "supervised",
        }

    @staticmethod
    def _rl_score(manifest: dict[str, Any]) -> float:
        metrics = dict(manifest.get("test_metrics") or {})
        return float(
            _f(metrics.get("mean_return"))
            + 0.35 * _f(metrics.get("mean_excess_vs_buy_hold"))
            + 0.20 * _f(metrics.get("median_return"))
            - 0.50 * _f(metrics.get("worst_maximum_drawdown"), 1.0)
        )

    def promote_rl(self) -> dict[str, Any]:
        if not bool(self.cfg.get("rl_enabled", True)):
            return {"status": "DISABLED", "kind": "rl"}

        root = self.root / "rl"
        pointer = _read_json(root / "latest.pointer.json")
        artifact = Path(str(pointer.get("artifact_path") or ""))
        manifest_path = Path(str(pointer.get("manifest_path") or ""))
        manifest = _read_json(manifest_path)

        if not artifact.is_file() or not manifest:
            return {"status": "NO_CANDIDATE", "kind": "rl"}
        if not bool(
            manifest.get("qualified", pointer.get("qualified", False))
        ):
            return {"status": "NOT_QUALIFIED", "kind": "rl"}

        stochastic = dict(manifest.get("stochastic_validation") or {})
        if stochastic and stochastic.get("passed") is False:
            return {
                "status": "NOT_QUALIFIED",
                "kind": "rl",
                "failures": ["STOCHASTIC_VALIDATION_FAILED"],
            }

        score = self._rl_score(manifest)
        current_pointer = _read_json(root / "live.pointer.json")
        current_manifest = _read_json(
            Path(str(current_pointer.get("manifest_path") or ""))
        )
        current_score = (
            self._rl_score(current_manifest) if current_manifest else None
        )
        source_artifact = str(artifact.resolve())

        if source_artifact == current_pointer.get("source_artifact_path"):
            return {
                "status": "CURRENT_CHAMPION",
                "kind": "rl",
                "score": score,
            }

        minimum_delta = float(
            self.cfg.get("minimum_rl_promotion_score_delta", 0.0)
        )
        if current_score is not None and score < current_score + minimum_delta:
            return {
                "status": "CHALLENGER_NOT_BETTER",
                "kind": "rl",
                "candidate_score": score,
                "champion_score": current_score,
                "minimum_delta": minimum_delta,
            }

        live_manifest = {
            **manifest,
            "status": "CANARY",
            "live_decision_influence": True,
            "automatic_live_promotion": True,
            "promoted_at": _now(),
            "promotion_score": score,
            "promotion_scope": "MODEL_INFLUENCE_ONLY",
            "execution_authority_granted": False,
            "capital_authority_granted": False,
        }

        live_root = root / "live"
        live_root.mkdir(parents=True, exist_ok=True)
        identity = hashlib.sha256(
            json.dumps(
                {
                    "artifact": source_artifact,
                    "score": score,
                    "seed": manifest.get("selected_seed"),
                },
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()

        live_manifest_path = live_root / f"{identity}.json"
        _atomic_json(live_manifest_path, live_manifest)

        pointer_path = root / "live.pointer.json"
        previous = root / "live.previous.pointer.json"
        if pointer_path.is_file():
            shutil.copy2(pointer_path, previous)

        live_pointer = {
            "schema_version": "swing_rl_live_pointer_v1",
            "artifact_path": source_artifact,
            "manifest_path": str(live_manifest_path.resolve()),
            "source_artifact_path": source_artifact,
            "qualified": True,
            "status": "CANARY",
            "live_decision_influence": True,
            "promotion_score": score,
            "promoted_at": _now(),
            "automatic_live_promotion": True,
            "execution_authority_granted": False,
        }
        _atomic_json(pointer_path, live_pointer)
        self._record({"event": "RL_PROMOTED", **live_pointer})
        return {
            **live_pointer,
            "model_status": live_pointer.get("status"),
            "status": "PROMOTED",
            "kind": "rl",
        }

    def cycle(self) -> dict[str, Any]:
        payload = {
            "schema_version": "crypto_ai_swing_live_model_governor_v1",
            "generated_at": _now(),
            "supervised": self.promote_supervised(),
            "rl": self.promote_rl(),
            "model_live_influence_enabled": True,
            "execution_authority_changed": False,
            "capital_caps_changed": False,
        }
        _atomic_json(
            self.root / "live_promotion_status.json",
            payload,
        )
        return payload

    def _record(self, payload: dict[str, Any]) -> None:
        self.history.parent.mkdir(parents=True, exist_ok=True)
        with self.history.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"at": _now(), **payload},
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
