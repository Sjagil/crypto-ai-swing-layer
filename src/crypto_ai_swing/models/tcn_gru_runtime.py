from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class TCNGRURuntime:
    """Hot-reloading runtime for the evidence-qualified TCN+GRU champion."""

    def __init__(self, settings, *, mode: str = "shadow") -> None:
        self.settings = settings
        self.mode = str(mode).lower()
        self.root = Path(settings.project_root) / "output/crypto_ai_swing/agents/tcn_gru"
        self.live_pointer = self.root / "live.pointer.json"
        self.candidate_pointer = self.root / "latest.pointer.json"
        self._pointer: Path | None = None
        self._mtime: float | None = None
        self._model = None
        self._meta: dict[str, Any] = {}
        self._error: str | None = None

    def _selected_pointer(self) -> Path | None:
        if self.mode == "live":
            return self.live_pointer if self.live_pointer.is_file() else None
        if self.live_pointer.is_file():
            return self.live_pointer
        return self.candidate_pointer if self.candidate_pointer.is_file() else None

    def _load(self):
        pointer = self._selected_pointer()
        if pointer is None:
            self._model = None
            self._meta = {}
            self._error = None
            return None
        mtime = pointer.stat().st_mtime
        if self._model is not None and self._pointer == pointer and self._mtime == mtime:
            return self._model
        try:
            meta = json.loads(pointer.read_text(encoding="utf-8"))
            if self.mode == "live" and meta.get("qualified") is not True:
                raise ValueError("TCN+GRU live pointer is not qualified")
            artifact = Path(str(meta.get("artifact_path") or ""))

            # Torch belongs to the optional AI runtime. Import the temporal
            # model only when a TCN+GRU artifact actually needs to be loaded.
            from crypto_ai_swing.models.tcn_gru import (
                load_tcn_gru_challenger,
            )

            model = load_tcn_gru_challenger(artifact)
            self._model = model
            self._meta = dict(meta)
            self._error = None
            self._pointer = pointer
            self._mtime = mtime
            return model
        except Exception as exc:
            self._model = None
            self._meta = {}
            self._error = f"{type(exc).__name__}:{str(exc)[:300]}"
            self._pointer = pointer
            self._mtime = mtime
            return None

    def status(self) -> dict[str, Any]:
        model = self._load()
        return {
            "status": "READY" if model is not None else "NOT_READY",
            "mode": self.mode,
            "pointer": str(self._pointer or self.live_pointer),
            "qualified": bool(self._meta.get("qualified", False)),
            "promotion_score": self._meta.get("promotion_score"),
            "error": self._error,
            "live_decision_influence": bool(
                model is not None and self.mode == "live" and self._meta.get("qualified") is True
            ),
        }

    def predict_frame(self, market: str, frame: pd.DataFrame) -> dict[str, Any]:
        model = self._load()
        if model is None:
            return {
                "market": str(market).upper(),
                "probability": None,
                "qualified": False,
                "live_decision_influence": False,
                "error": self._error,
            }
        result = model.predict_latest_ohlcv(
            str(market).upper(),
            frame,
            device="cpu" if self.mode == "live" else "auto",
        )
        return {
            **result,
            "qualified": bool(self._meta.get("qualified", self.mode != "live")),
            "live_decision_influence": bool(
                self.mode != "live" or self._meta.get("qualified") is True
            ),
            "promotion_score": self._meta.get("promotion_score"),
            "metrics": model.metadata.get("metrics") or {},
        }
