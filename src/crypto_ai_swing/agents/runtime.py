from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from crypto_ai_swing.agents.council import AgentCouncilDecision, build_council_decision
from crypto_ai_swing.agents.canonical_features import canonical_model_frame
from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.data.features import build_features as legacy_build_features


class AgentRuntime:
    def __init__(self, settings, *, mode: str = "shadow") -> None:
        self.settings = settings
        self.mode = str(mode).lower()
        self.crypto = CryptoLibraryBridge(settings.crypto_repo_root)
        self.root = settings.project_root / "output/crypto_ai_swing/agents"
        self.candidate_pointer = self.root / "latest.pointer.json"
        self.live_pointer = self.root / "live.pointer.json"
        self.previous_live_pointer = self.root / "live.previous.pointer.json"
        self._pointer_path: Path | None = None
        self._mtime: float | None = None
        self._bundle: dict[str, Any] | None = None
        self._error: str | None = None

    def _pointers(self) -> list[Path]:
        if self.mode == "live":
            return [
                self.live_pointer,
                self.previous_live_pointer,
                self.candidate_pointer,
            ]
        return [self.candidate_pointer]

    def _load(self) -> dict[str, Any] | None:
        selected = next((p for p in self._pointers() if p.is_file()), None)
        if selected is None:
            self._bundle = None
            self._error = None
            self._pointer_path = None
            return None
        if (
            self._bundle is not None
            and self._mtime == selected.stat().st_mtime
            and self._pointer_path == selected
        ):
            return self._bundle

        last_error = None
        for pointer_path in self._pointers():
            if not pointer_path.is_file():
                continue
            try:
                pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                bundle = joblib.load(Path(str(pointer["artifact_path"])))
                if not isinstance(bundle, dict):
                    raise ValueError("agent artifact is not a dict")
                self._bundle = bundle
                self._error = None
                self._mtime = pointer_path.stat().st_mtime
                self._pointer_path = pointer_path
                return bundle
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:300]}"

        self._bundle = None
        self._error = last_error
        self._pointer_path = selected
        self._mtime = selected.stat().st_mtime
        return None

    def status(self) -> dict[str, Any]:
        bundle = self._load()
        if bundle is None:
            return {
                "status": "NOT_TRAINED" if self._error is None else "ERROR",
                "error": self._error,
                "pointer": str(self._pointer_path or self.candidate_pointer),
                "shadow_decision_qualified": False,
                "head_qualifications": {},
                "live_decision_influence": False,
            }
        return {
            key: bundle.get(key)
            for key in (
                "status",
                "trained_at",
                "expires_at",
                "dataset_id",
                "dataset_rows",
                "markets",
                "timeframe",
                "horizon_bars",
                "metrics",
                "shadow_decision_qualified",
                "head_qualifications",
                "alpha_probability_threshold",
                "alpha_calibration_method",
                "global_known_trial_count",
                "live_decision_influence",
                "automatic_live_promotion",
                "promoted_at",
                "promotion_score",
                "promotion_scope",
            )
        } | {
            "pointer": str(self._pointer_path or self.candidate_pointer),
            "live_pointer_selected": (
                self.mode == "live"
                and self._pointer_path
                in {self.live_pointer, self.previous_live_pointer}
            ),
        }

    def predict_frame(
        self,
        market: str,
        frame: pd.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> AgentCouncilDecision:
        context = dict(context or {})
        bundle = self._load()
        if bundle is None:
            return build_council_decision(
                {},
                context,
                artifact_status="NOT_TRAINED",
                live_decision_influence=False,
                mode=self.mode,
                shadow_decision_qualified=False,
                head_qualifications={},
            )
        raw_expiry = bundle.get("expires_at")
        if raw_expiry:
            expiry = datetime.fromisoformat(str(raw_expiry).replace("Z", "+00:00"))
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > expiry.astimezone(timezone.utc):
                return build_council_decision(
                    {},
                    context,
                    artifact_status="EXPIRED",
                    live_decision_influence=False,
                    mode=self.mode,
                    shadow_decision_qualified=False,
                    head_qualifications={},
                )

        features = tuple(bundle.get("feature_columns") or ())
        if bundle.get("feature_source") == "canonical_feature_pipeline_v1":
            feat = canonical_model_frame(
                self.crypto,
                frame,
                market=str(market).upper(),
                timeframe=str(
                    bundle.get("timeframe")
                    or frame.attrs.get("timeframe")
                    or "1h"
                ),
                benchmark=None,
            ).replace([np.inf, -np.inf], np.nan)
        else:
            feat = legacy_build_features(frame).replace(
                [np.inf, -np.inf],
                np.nan,
            )
        if feat.empty or not features:
            return build_council_decision(
                {},
                context,
                artifact_status=str(bundle.get("status") or "SHADOW"),
                live_decision_influence=False,
                mode=self.mode,
                shadow_decision_qualified=False,
                head_qualifications={},
            )

        x = feat.iloc[[-1]].reindex(columns=list(features))
        models = dict(bundle.get("models") or {})
        predictions: dict[str, float | None] = {
            "alpha_probability": None,
            "regime_probability": None,
            "predicted_return": None,
            "predicted_mae": None,
        }
        try:
            predictions["alpha_probability"] = float(
                models["alpha"].predict_proba(x)[0, 1]
            )
        except Exception:
            pass
        try:
            predictions["regime_probability"] = float(
                models["regime"].predict_proba(x)[0, 1]
            )
        except Exception:
            pass
        try:
            predictions["predicted_return"] = float(
                models["return"].predict(x)[0]
            )
        except Exception:
            pass
        try:
            predictions["predicted_mae"] = max(
                0.0,
                float(models["risk"].predict(x)[0]),
            )
        except Exception:
            pass

        legacy_qualified = bool(bundle.get("shadow_decision_qualified", False))
        head_qualifications = bundle.get("head_qualifications")
        if not isinstance(head_qualifications, dict):
            head_qualifications = {
                "alpha": legacy_qualified,
                "regime": legacy_qualified,
                "return": legacy_qualified,
                "risk": legacy_qualified,
                "execution": True,
            }

        return build_council_decision(
            predictions,
            context,
            artifact_status=str(bundle.get("status") or "SHADOW"),
            live_decision_influence=bool(
                bundle.get("live_decision_influence", False)
            ),
            mode=self.mode,
            shadow_decision_qualified=legacy_qualified,
            head_qualifications=head_qualifications,
        )
