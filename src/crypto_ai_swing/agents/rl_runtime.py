from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from crypto_ai_swing.agents.rl_multi_market import (
    RL_STATE_FEATURES,
    RL_STATE_VERSION,
)

from crypto_ai_swing.agents.dataset import DEFAULT_FEATURES
from crypto_ai_swing.agents.canonical_features import canonical_model_frame
from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.data.features import build_features as legacy_build_features


def _runtime_state_vector(
    manifest: dict[str, Any],
) -> np.ndarray:
    if (
        str(
            manifest.get(
                "observation_state_version"
            )
            or ""
        )
        == RL_STATE_VERSION
    ):
        return np.zeros(
            len(RL_STATE_FEATURES),
            dtype=np.float32,
        )

    return np.zeros(
        2,
        dtype=np.float32,
    )


def _runtime_live_influence_allowed(
    manifest: dict[str, Any],
) -> bool:
    requested = bool(
        manifest.get(
            "live_decision_influence",
            False,
        )
    )

    if not requested:
        return False

    if (
        str(
            manifest.get(
                "observation_state_version"
            )
            or ""
        )
        == RL_STATE_VERSION
        and not bool(
            manifest.get(
                "position_manager_live_ready",
                False,
            )
        )
    ):
        return False

    return True


class RLRuntime:
    def __init__(self, settings, *, mode: str = "shadow") -> None:
        self.mode = str(mode).lower()
        self.crypto = CryptoLibraryBridge(settings.crypto_repo_root)
        self.root = settings.project_root / "output/crypto_ai_swing/agents/rl"
        self.candidate_pointer = self.root / "latest.pointer.json"
        self.live_pointer = self.root / "live.pointer.json"
        self.previous_live_pointer = self.root / "live.previous.pointer.json"
        self._pointer_path = None
        self._mtime = None
        self._model = None
        self._manifest: dict[str, Any] = {}
        self._error = None

    def _pointers(self):
        if self.mode == "live":
            return [
                self.live_pointer,
                self.previous_live_pointer,
                self.candidate_pointer,
            ]
        return [self.candidate_pointer]

    def _load(self) -> None:
        selected = next((p for p in self._pointers() if p.is_file()), None)
        if selected is None:
            self._model = None
            self._manifest = {}
            self._pointer_path = None
            return
        if (
            self._model is not None
            and self._mtime == selected.stat().st_mtime
            and self._pointer_path == selected
        ):
            return

        last_error = None
        for pointer_path in self._pointers():
            if not pointer_path.is_file():
                continue
            try:
                from stable_baselines3 import PPO

                pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                manifest_path = Path(pointer["manifest_path"])
                self._manifest = dict(
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                )
                self._model = PPO.load(str(Path(pointer["artifact_path"])))
                self._mtime = pointer_path.stat().st_mtime
                self._pointer_path = pointer_path
                self._error = None
                return
            except Exception as exc:
                last_error = f"{type(exc).__name__}:{str(exc)[:240]}"

        self._model = None
        self._manifest = {}
        self._mtime = selected.stat().st_mtime
        self._pointer_path = selected
        self._error = last_error

    def status(self) -> dict[str, Any]:
        self._load()
        live_influence = bool(
            self.mode == "live"
            and self._model is not None
            and self._manifest.get("qualified", False)
            and self._manifest.get("live_decision_influence", False)
            and str(self._manifest.get("status") or "").upper()
            in {"CANARY", "ACTIVE"}
        )
        return {
            "status": (
                self._manifest.get("status", "SHADOW")
                if self._model is not None
                else ("ERROR" if self._error else "NOT_TRAINED")
            ),
            "qualified": bool(self._manifest.get("qualified", False)),
            "trained_at": self._manifest.get("trained_at"),
            "selected_seed": self._manifest.get("selected_seed"),
            "test_metrics": self._manifest.get("test_metrics") or {},
            "bayesian": self._manifest.get("bayesian") or {},
            "stochastic_validation": (
                self._manifest.get("stochastic_validation") or {}
            ),
            "error": self._error,
            "live_decision_influence": _runtime_live_influence_allowed(self._manifest),
            "automatic_live_promotion": bool(
                self._manifest.get("automatic_live_promotion", False)
            ),
            "pointer": str(self._pointer_path or self.candidate_pointer),
            "live_pointer_selected": (
                self.mode == "live"
                and self._pointer_path
                in {self.live_pointer, self.previous_live_pointer}
            ),
        }

    def predict_frame(self, frame) -> dict[str, Any]:
        self._load()
        if self._model is None:
            return {
                **self.status(),
                "score": None,
                "long_probability": None,
            }

        features = tuple(
            self._manifest.get("feature_columns") or DEFAULT_FEATURES
        )
        if self._manifest.get("feature_source") == "canonical_feature_pipeline_v1":
            table = canonical_model_frame(
                self.crypto,
                frame,
                market=str(frame.attrs.get("market") or "UNKNOWN"),
                timeframe=str(frame.attrs.get("timeframe") or "1h"),
                benchmark=None,
            ).replace([np.inf, -np.inf], np.nan)
        else:
            table = legacy_build_features(frame).replace(
                [np.inf, -np.inf],
                np.nan,
            )
        if table.empty:
            return {
                **self.status(),
                "score": None,
                "long_probability": None,
                "reason": "NO_COMPLETE_FEATURE_ROW",
            }

        row = table.iloc[-1].reindex(list(features)).astype(float)
        means = self._manifest.get("feature_mean") or {}
        stds = self._manifest.get("feature_std") or {}
        if means and stds:
            mean = np.asarray([
                float(means.get(name, 0.0))
                for name in features
            ])
            std = np.asarray([
                max(1e-12, float(stds.get(name, 1.0)))
                for name in features
            ])
            raw = row.to_numpy(float)
            raw = np.where(np.isfinite(raw), raw, mean)
            values = np.clip((raw - mean) / std, -10.0, 10.0)
        else:
            values = row.to_numpy(float)

        obs = np.concatenate([
            values.astype(np.float32),
            _runtime_state_vector(self._manifest),
        ])
        action, _ = self._model.predict(obs, deterministic=True)
        action = int(np.asarray(action).reshape(-1)[0])

        probability = None
        try:
            tensor, _ = self._model.policy.obs_to_tensor(obs)
            distribution = self._model.policy.get_distribution(tensor)
            probs = (
                distribution.distribution.probs
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            if len(probs) >= 2:
                probability = float(probs[1])
        except Exception:
            pass

        score = (
            2.0 * probability - 1.0
            if probability is not None
            else (1.0 if action == 1 else -1.0)
        )
        status = self.status()
        return {
            **status,
            "action": "LONG" if action == 1 else "FLAT",
            "long_probability": probability,
            "score": float(np.clip(score, -1.0, 1.0)),
            "authority": (
                "LIVE_MODEL_INFLUENCE"
                if status["live_decision_influence"]
                else "ADVISORY_ONLY"
            ),
        }
