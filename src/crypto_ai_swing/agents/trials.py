from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "swing_agent_global_trial_ledger_v2"


def _canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return sha256(raw.encode("utf-8")).hexdigest()


def _hypotheses(experiment: Mapping[str, Any]) -> list[dict[str, Any]]:
    models = sorted(str(x) for x in experiment.get("models", []))
    calibrations = sorted(
        str(x) for x in experiment.get("calibration_methods", [])
    )
    thresholds = sorted(
        round(float(x), 6)
        for x in experiment.get("thresholds", [])
    )
    if not models or not calibrations or not thresholds:
        return []

    shared = {
        "schema_version": str(
            experiment.get("schema_version", "agent_alpha_tournament_v4")
        ),
        "timeframe": str(experiment.get("timeframe", "")),
        "horizon_bars": int(experiment.get("horizon_bars") or 0),
        "minimum_net_move_bps": float(
            experiment.get("minimum_net_move_bps") or 0.0
        ),
        "feature_columns": list(experiment.get("feature_columns", [])),
    }

    rows: list[dict[str, Any]] = []
    for model, calibration, threshold in product(
        models,
        calibrations,
        thresholds,
    ):
        rows.append(
            {
                **shared,
                "model": model,
                "calibration_method": calibration,
                "threshold": threshold,
            }
        )
    return rows


def _load(path: Path, historical_floor: int) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": SCHEMA,
            "historical_trial_floor": max(0, int(historical_floor)),
            "experiments": [],
            "hypotheses": {},
        }

    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        old = {}
    if not isinstance(old, dict):
        old = {}

    experiments = [
        row
        for row in old.get("experiments", [])
        if isinstance(row, dict)
    ]

    hypotheses: dict[str, dict[str, Any]] = {}
    existing_hypotheses = old.get("hypotheses", {})
    if isinstance(existing_hypotheses, dict):
        for key, value in existing_hypotheses.items():
            if isinstance(value, dict):
                hypotheses[str(key)] = value

    for row in experiments:
        experiment = row.get("experiment", {})
        if not isinstance(experiment, dict):
            continue
        for hypothesis in _hypotheses(experiment):
            key = _canonical_hash(hypothesis)
            hypotheses.setdefault(
                key,
                {
                    "hypothesis_id": key,
                    "hypothesis": hypothesis,
                    "first_seen_at": row.get("registered_at"),
                },
            )

    return {
        "schema_version": SCHEMA,
        "historical_trial_floor": max(
            int(old.get("historical_trial_floor") or 0),
            max(0, int(historical_floor)),
        ),
        "experiments": experiments,
        "hypotheses": hypotheses,
    }


def register_trial_family(
    root: Path,
    *,
    experiment: Mapping[str, Any],
    trial_count: int,
    historical_floor: int = 0,
) -> dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "global_trial_ledger.json"
    ledger = _load(path, historical_floor)

    experiment_payload = dict(experiment)
    generated = _hypotheses(experiment_payload)
    if not generated:
        raise ValueError("experiment cannot generate hypothesis identities")

    expected = len(generated)
    if int(trial_count) != expected:
        raise ValueError(
            f"trial_count={trial_count} does not match "
            f"generated hypothesis count={expected}"
        )

    experiment_id = _canonical_hash(experiment_payload)
    existing_ids = {
        str(row.get("experiment_id"))
        for row in ledger["experiments"]
        if row.get("experiment_id")
    }
    newly_registered_evaluation = experiment_id not in existing_ids

    if newly_registered_evaluation:
        ledger["experiments"].append(
            {
                "experiment_id": experiment_id,
                "registered_at": datetime.now(timezone.utc).isoformat(),
                "trial_count": expected,
                "dataset_id": experiment_payload.get("dataset_id"),
                "experiment": experiment_payload,
            }
        )

    newly_registered_hypotheses = 0
    for hypothesis in generated:
        hypothesis_id = _canonical_hash(hypothesis)
        if hypothesis_id not in ledger["hypotheses"]:
            ledger["hypotheses"][hypothesis_id] = {
                "hypothesis_id": hypothesis_id,
                "hypothesis": hypothesis,
                "first_seen_at": datetime.now(timezone.utc).isoformat(),
            }
            newly_registered_hypotheses += 1

    unique_count = len(ledger["hypotheses"])
    floor = int(ledger["historical_trial_floor"])

    # The historical floor is a lower bound, not an additive count.
    global_known = max(floor, unique_count)

    ledger.update(
        {
            "schema_version": SCHEMA,
            "unique_hypothesis_count": unique_count,
            "evaluation_count": len(ledger["experiments"]),
            "global_known_trial_count": global_known,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "accounting_semantics": (
                "UNIQUE_HYPOTHESES_WITH_REPEATED_EVALUATIONS_DEDUPED"
            ),
        }
    )
    path.write_text(
        json.dumps(ledger, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "schema_version": SCHEMA,
        "path": str(path.resolve()),
        "experiment_id": experiment_id,
        "newly_registered": newly_registered_evaluation,
        "newly_registered_hypotheses": newly_registered_hypotheses,
        "current_experiment_trials": expected,
        "historical_trial_floor": floor,
        "evaluation_count": len(ledger["experiments"]),
        "unique_hypothesis_count": unique_count,
        "global_known_trial_count": global_known,
        "accounting_semantics": (
            "UNIQUE_HYPOTHESES_WITH_REPEATED_EVALUATIONS_DEDUPED"
        ),
    }
