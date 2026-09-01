from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from crypto_ai_swing.agents.council import AgentCouncilDecision, build_council_decision
from crypto_ai_swing.data.features import build_features


class AgentRuntime:
    def __init__(self, settings, *, mode: str = "shadow") -> None:
        self.settings = settings
        self.mode = str(mode).lower()
        self.root = settings.project_root / "output/crypto_ai_swing/agents"
        self.pointer = self.root / "latest.pointer.json"
        self._mtime: float | None = None
        self._bundle: dict[str, Any] | None = None
        self._error: str | None = None

    def _load(self) -> dict[str, Any] | None:
        if not self.pointer.is_file():
            self._bundle = None
            self._error = None
            return None
        mtime = self.pointer.stat().st_mtime
        if self._bundle is not None and self._mtime == mtime:
            return self._bundle
        try:
            pointer = json.loads(self.pointer.read_text(encoding="utf-8"))
            artifact = Path(str(pointer["artifact_path"]))
            bundle = joblib.load(artifact)
            if not isinstance(bundle, dict):
                raise ValueError("agent artifact is not a dict")
            self._bundle = bundle
            self._error = None
            self._mtime = mtime
        except Exception as exc:
            self._bundle = None
            self._error = f"{type(exc).__name__}: {str(exc)[:300]}"
            self._mtime = mtime
        return self._bundle

    def status(self) -> dict[str, Any]:
        bundle = self._load()
        if bundle is None:
            return {
                "status": "NOT_TRAINED" if self._error is None else "ERROR",
                "error": self._error,
                "pointer": str(self.pointer),
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
                "live_decision_influence",
            )
        } | {"automatic_live_promotion": False}

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
        feat = build_features(frame).replace([np.inf, -np.inf], np.nan)
        feat = feat.dropna(subset=list(features))
        if feat.empty:
            return build_council_decision(
                {},
                context,
                artifact_status=str(bundle.get("status") or "SHADOW"),
                live_decision_influence=False,
                mode=self.mode,
                shadow_decision_qualified=False,
                head_qualifications={},
            )
        x = feat.iloc[[-1]].loc[:, features]
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
            predictions["predicted_return"] = float(models["return"].predict(x)[0])
        except Exception:
            pass
        try:
            predictions["predicted_mae"] = max(
                0.0, float(models["risk"].predict(x)[0])
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
            live_decision_influence=bool(bundle.get("live_decision_influence", False)),
            mode=self.mode,
            shadow_decision_qualified=legacy_qualified,
            head_qualifications=head_qualifications,
        )
