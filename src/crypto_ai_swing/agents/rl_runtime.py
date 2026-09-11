from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from crypto_ai_swing.agents.dataset import DEFAULT_FEATURES
from crypto_ai_swing.agents.canonical_features import canonical_model_frame
from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.data.features import build_features as legacy_build_features


class RLRuntime:
    def __init__(self, settings) -> None:
        self.crypto = CryptoLibraryBridge(settings.crypto_repo_root)
        self.root = settings.project_root / "output/crypto_ai_swing/agents/rl"
        self.pointer = self.root / "latest.pointer.json"
        self._mtime = None
        self._model = None
        self._manifest: dict[str, Any] = {}
        self._error = None

    def _load(self) -> None:
        if not self.pointer.is_file():
            self._model = None
            self._manifest = {}
            return
        mtime = self.pointer.stat().st_mtime
        if self._model is not None and self._mtime == mtime:
            return
        try:
            from stable_baselines3 import PPO
            pointer = json.loads(self.pointer.read_text(encoding="utf-8"))
            manifest_path = Path(pointer["manifest_path"])
            self._manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            self._model = PPO.load(str(Path(pointer["artifact_path"])))
            self._mtime = mtime
            self._error = None
        except Exception as exc:
            self._model = None
            self._manifest = {}
            self._mtime = mtime
            self._error = f"{type(exc).__name__}:{str(exc)[:240]}"

    def status(self) -> dict[str, Any]:
        self._load()
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
            "live_decision_influence": False,
            "automatic_live_promotion": False,
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
            table = legacy_build_features(frame).replace([np.inf, -np.inf], np.nan)
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
            mean = np.asarray([float(means.get(name, 0.0)) for name in features])
            std = np.asarray([max(1e-12, float(stds.get(name, 1.0))) for name in features])
            raw = row.to_numpy(float)
            raw = np.where(np.isfinite(raw), raw, mean)
            values = (raw - mean) / std
            values = np.clip(values, -10.0, 10.0)
        else:
            # Backward compatibility with v2 artifacts.
            values = row.to_numpy(float)

        obs = np.concatenate(
            [
                values.astype(np.float32),
                np.asarray([0.0, 0.0], dtype=np.float32),
            ]
        )
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
        return {
            **self.status(),
            "action": "LONG" if action == 1 else "FLAT",
            "long_probability": probability,
            "score": float(np.clip(score, -1.0, 1.0)),
            "authority": "ADVISORY_ONLY",
            "live_decision_influence": False,
        }
