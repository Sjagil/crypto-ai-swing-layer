from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

MODEL_SUFFIXES = {".joblib", ".pkl", ".pickle", ".pt", ".pth", ".zip", ".safetensors", ".onnx"}
TOKENS = (
    "agent", "model", "training", "train", "alpha", "regime", "return",
    "risk", "execution", "bayesian", "stochastic", "monte", "calibration",
    "qualification", "selection", "prospective", "forward", "evidence", "rl", "optimization", "research",
)


def _safe_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _find_key(value: Any, keys: tuple[str, ...], depth: int = 0) -> Any:
    if depth > 6:
        return None
    if isinstance(value, dict):
        for key in keys:
            if key in value:
                return value[key]
        for item in value.values():
            found = _find_key(item, keys, depth + 1)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value[:25]:
            found = _find_key(item, keys, depth + 1)
            if found is not None:
                return found
    return None


def _metric(payload: dict[str, Any]) -> str:
    preferred = (
        ("auc", "AUC"), ("roc_auc", "AUC"), ("validation_auc", "valAUC"),
        ("brier", "Brier"), ("brier_score", "Brier"), ("mae", "MAE"),
        ("rmse", "RMSE"), ("mean_return_bps", "mean"),
        ("positive_return_rate", "hit"), ("observations", "N"),
        ("simulation_count", "MC"), ("simulations", "MC"),
    )
    for key, label in preferred:
        value = _find_key(payload, (key,))
        if value is not None:
            if isinstance(value, float):
                return f"{label}={value:.4f}"
            return f"{label}={value}"
    return "-"


def _status(payload: dict[str, Any]) -> str:
    value = _find_key(payload, ("status", "state", "classification", "lifecycle_state", "qualification"))
    if value is not None:
        return str(value)
    passed = _find_key(payload, ("passed", "eligible", "qualified"))
    if passed is True:
        return "PASS"
    if passed is False:
        return "NOT_QUALIFIED"
    return "ARTIFACT"


def _authority(payload: dict[str, Any]) -> str:
    value = _find_key(payload, ("authority", "scope", "mode"))
    if value is not None:
        return str(value)
    influence = _find_key(payload, ("live_decision_influence",))
    if influence is False:
        return "SHADOW/ADVISORY"
    if influence is True:
        return "LIVE_INFLUENCE"
    return "-"


def _updated(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).strftime("%m-%d %H:%M:%S")


def _candidate_files(roots: tuple[Path, ...], limit: int = 300) -> list[Path]:
    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if len(candidates) >= limit:
                break
            if not path.is_file():
                continue
            lower = str(path).lower()
            if not any(token in lower for token in TOKENS):
                continue
            if path.suffix.lower() == ".json" or path.suffix.lower() in MODEL_SUFFIXES:
                candidates.append(path)
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates


def _configured_agents(project_root: Path) -> list[dict[str, Any]]:
    path = project_root / "config" / "agents.yaml"
    if not path.is_file():
        return []
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    panel = dict(config.get("research_agent_panel") or {})
    names = panel.get("agents") or []
    authority = str(panel.get("authority") or "ADVISORY_ONLY")
    influence = bool(panel.get("live_decision_influence", False))
    rows = []
    for name in names:
        rows.append({
            "component": str(name),
            "state": "CONFIGURED",
            "authority": authority,
            "detail": "live influence" if influence else "no live influence",
            "updated": "config",
            "source": str(path),
        })
    return rows


def collect_ai_training_state(project_root: Path, crypto_repo_root: Path, *, max_agent_rows: int = 10, max_training_rows: int = 10) -> dict[str, Any]:
    roots = (
        project_root / "output", project_root / "artifacts", project_root / "models", project_root / "results",
        crypto_repo_root / "output", crypto_repo_root / "artifacts", crypto_repo_root / "models", crypto_repo_root / "results",
    )
    agent_rows = _configured_agents(project_root)
    training_rows = []
    for path in _candidate_files(roots):
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = _safe_json(path)
            if not payload:
                continue
            row = {
                "component": path.stem,
                "state": _status(payload),
                "authority": _authority(payload),
                "detail": _metric(payload),
                "updated": _updated(path),
                "source": str(path),
            }
            lower = str(path).lower()
            if any(token in lower for token in ("training", "train", "qualification", "calibration", "selection", "prospective", "forward", "evidence")):
                training_rows.append(row)
            else:
                agent_rows.append(row)
        elif suffix in MODEL_SUFFIXES:
            training_rows.append({
                "component": path.name,
                "state": "MODEL_ARTIFACT",
                "authority": "-",
                "detail": f"{path.stat().st_size / 1024:.1f} KiB",
                "updated": _updated(path),
                "source": str(path),
            })

    def dedupe(rows):
        seen = set()
        result = []
        for row in rows:
            key = (str(row["component"]), str(row["source"]))
            if key in seen:
                continue
            seen.add(key)
            result.append(row)
        return result

    agents = dedupe(agent_rows)[:max_agent_rows]
    training = dedupe(training_rows)[:max_training_rows]
    return {
        "schema_version": "crypto_ai_swing_ai_training_observability_v1",
        "agents": agents,
        "training": training,
        "agent_count": len(agents),
        "training_row_count": len(training),
        "authority": "READ_ONLY_OBSERVABILITY",
        "live_decision_influence": False,
    }
