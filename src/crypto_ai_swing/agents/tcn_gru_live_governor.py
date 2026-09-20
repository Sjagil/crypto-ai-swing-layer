from __future__ import annotations

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
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default


class TCNGRULiveGovernor:
    """Automatically promote only an OOS-qualified TCN+GRU challenger.

    This governor changes model influence, not exchange authority or portfolio
    risk limits. It retains a previous pointer for immediate rollback.
    """

    def __init__(self, settings) -> None:
        self.settings = settings
        self.root = Path(settings.project_root) / "output/crypto_ai_swing/agents/tcn_gru"
        self.history = self.root / "promotion_history.jsonl"

    @staticmethod
    def _eligible(payload: dict[str, Any]) -> tuple[bool, list[str], float]:
        metrics = dict(payload.get("metrics") or {})
        test = dict(metrics.get("test") or {})
        prob = dict(test.get("probability") or {})
        econ = dict(test.get("economics") or {})
        failures: list[str] = []

        rows = int(prob.get("rows") or 0)
        auc = _f(prob.get("auc"), 0.5)
        brier = _f(prob.get("brier"), 1.0)
        count = int(econ.get("count") or 0)
        market_count = int(econ.get("market_count") or 0)
        conservative = _f(econ.get("conservative_mean_net"), -1.0)
        market_balanced = _f(econ.get("market_balanced_mean_net"), -1.0)
        positive_fraction = _f(econ.get("positive_market_fraction"), 0.0)

        if rows < 250:
            failures.append("INSUFFICIENT_TEST_ROWS")
        if auc < 0.51:
            failures.append("TEST_AUC_TOO_LOW")
        if brier > 0.26:
            failures.append("TEST_BRIER_TOO_HIGH")
        if count < 60:
            failures.append("INSUFFICIENT_SELECTED_TEST_ROWS")
        if market_count < 5:
            failures.append("INSUFFICIENT_TEST_MARKET_BREADTH")
        if conservative <= 0.0:
            failures.append("NON_POSITIVE_CONSERVATIVE_NET")
        if market_balanced <= 0.0:
            failures.append("NON_POSITIVE_MARKET_BALANCED_NET")
        if positive_fraction < 0.50:
            failures.append("INSUFFICIENT_POSITIVE_MARKET_FRACTION")

        score = (
            2.0 * conservative
            + 1.5 * market_balanced
            + 0.08 * (auc - 0.5)
            + 0.05 * (0.25 - brier)
            + 0.05 * (positive_fraction - 0.5)
        )
        return not failures, failures, float(score)

    def _current_score(self) -> float | None:
        pointer = _read_json(self.root / "live.pointer.json")
        artifact = Path(str(pointer.get("artifact_path") or ""))
        if not artifact.is_file():
            return None
        try:
            payload = dict(joblib.load(artifact))
        except Exception:
            return None
        return self._eligible(payload)[2]

    def cycle(self) -> dict[str, Any]:
        candidate_pointer = _read_json(self.root / "latest.pointer.json")
        artifact = Path(str(candidate_pointer.get("artifact_path") or ""))
        if not artifact.is_file():
            return {"status": "NO_CANDIDATE", "qualified": False}

        expected = str(candidate_pointer.get("artifact_hash") or "")
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if expected and expected != actual:
            return {"status": "HASH_MISMATCH", "qualified": False}

        payload = dict(joblib.load(artifact))
        eligible, failures, score = self._eligible(payload)
        if not eligible:
            result = {
                "status": "NOT_QUALIFIED",
                "qualified": False,
                "failures": failures,
                "candidate_score": score,
                "checked_at": _now(),
            }
            _atomic_json(self.root / "qualification.json", result)
            return result

        current_score = self._current_score()
        if current_score is not None and score <= current_score + 1e-6:
            return {
                "status": "CHALLENGER_NOT_BETTER",
                "qualified": True,
                "candidate_score": score,
                "champion_score": current_score,
            }

        live = self.root / "live.pointer.json"
        previous = self.root / "live.previous.pointer.json"
        if live.is_file():
            shutil.copy2(live, previous)

        pointer = {
            "schema_version": "tcn_gru_live_pointer_v1",
            "status": "LIVE_QUALIFIED",
            "qualified": True,
            "artifact_path": str(artifact.resolve()),
            "manifest_path": candidate_pointer.get("manifest_path"),
            "artifact_hash": actual,
            "source_artifact_hash": actual,
            "promotion_score": score,
            "promoted_at": _now(),
            "live_decision_influence": True,
            "automatic_live_promotion": True,
            "execution_authority_granted": False,
            "capital_authority_granted": False,
        }
        _atomic_json(live, pointer)
        with self.history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(pointer, sort_keys=True) + "\n")
        return {"status": "PROMOTED", **pointer}
