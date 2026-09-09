from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

import numpy as np

from crypto_ai_swing.research.entry_selector import (
    SELECTOR_FEATURES,
    ProspectiveSwingEntrySelector,
)
from crypto_ai_swing.research.optuna_runtime import OptunaStudyRuntime


OBJECTIVE_VERSION = "prospective_selector_nested_temporal_v2_complete_horizons"


@dataclass(frozen=True)
class OuterSplit:
    calibration_ids: tuple[str, ...]
    holdout_ids: tuple[str, ...]
    calibration_observations: int
    holdout_observations: int
    label_maturity_cutoff: str
    holdout_start: str | None


def _group_rows(
    rows: Iterable[dict[str, Any]],
    *,
    required_horizons: Iterable[int] | None = None,
) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["observation_id"]), []).append(row)
    groups = sorted(
        grouped.values(),
        key=lambda group: (group[0]["observed_dt"], group[0]["observation_id"]),
    )
    if required_horizons is None:
        return groups
    required = {int(value) for value in required_horizons}
    if not required:
        return groups
    return [
        group
        for group in groups
        if required.issubset({int(row["horizon_hours"]) for row in group})
    ]


def _trial_budget(study: Mapping[str, Any], target_total: int) -> dict[str, int]:
    trials = list(study.get("trials") or [])
    finished = sum(
        1
        for trial in trials
        if str(trial.get("state") or "").upper() in {"COMPLETE", "FAIL", "PRUNED"}
    )
    running = sum(
        1
        for trial in trials
        if str(trial.get("state") or "").upper() == "RUNNING"
    )
    return {
        "target_total": int(target_total),
        "finished_before": int(finished),
        "running_before": int(running),
        "remaining": max(0, int(target_total) - int(finished)),
    }


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "observations": 0,
            "mean_bps": None,
            "positive_fraction": None,
            "profit_factor": None,
        }
    winners = arr[arr > 0.0]
    losers = arr[arr <= 0.0]
    gross_profit = float(winners.sum()) if len(winners) else 0.0
    gross_loss = abs(float(losers.sum())) if len(losers) else 0.0
    return {
        "observations": int(len(arr)),
        "mean_bps": float(arr.mean()),
        "positive_fraction": float(np.mean(arr > 0.0)),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
    }


def _float(value: Any, default: float) -> float:
    try:
        selected = float(value)
    except (TypeError, ValueError):
        return default
    return selected if math.isfinite(selected) else default


@contextmanager
def _selector_parameters(selector: ProspectiveSwingEntrySelector, params: Mapping[str, Any]):
    mapping = {
        "ridge": "ridge",
        "maximum_factors": "maximum_factors",
        "minimum_geometry_probability": "minimum_probability",
        "minimum_expected_normal_net_bps": "minimum_expected_normal",
        "minimum_expected_stressed_net_bps": "minimum_expected_stressed",
        "minimum_expected_mfe_cost_multiple": "minimum_mfe_cost_multiple",
        "minimum_expected_mfe_to_mae": "minimum_excursion_ratio",
        "minimum_return_direction_stability": "minimum_direction_stability",
    }
    before = {attribute: getattr(selector, attribute) for attribute in mapping.values()}
    bootstrap_before = selector.bootstrap_draws
    try:
        for parameter, attribute in mapping.items():
            if parameter in params:
                value = params[parameter]
                if attribute == "maximum_factors":
                    value = int(value)
                else:
                    value = float(value)
                setattr(selector, attribute, value)
        selector.bootstrap_draws = min(int(bootstrap_before), 500)
        yield
    finally:
        for attribute, value in before.items():
            setattr(selector, attribute, value)
        selector.bootstrap_draws = bootstrap_before


class SelectorOptunaTuner:
    SCHEMA = "crypto_ai_swing_selector_optuna_v1"

    def __init__(
        self,
        selector: ProspectiveSwingEntrySelector,
        *,
        runtime: OptunaStudyRuntime | None = None,
    ) -> None:
        self.selector = selector
        self.settings = selector.settings
        self.project_root = Path(self.settings.project_root).resolve()
        self.runtime = runtime or OptunaStudyRuntime(self.project_root)
        self.root = self.project_root / "output/crypto_ai_swing/research/optuna"
        self.root.mkdir(parents=True, exist_ok=True)

    def search_space(self) -> dict[str, dict[str, Any]]:
        baseline_probability = float(self.selector.minimum_probability)
        baseline_normal = float(self.selector.minimum_expected_normal)
        baseline_stressed = float(self.selector.minimum_expected_stressed)
        baseline_cost_multiple = float(self.selector.minimum_mfe_cost_multiple)
        baseline_excursion = float(self.selector.minimum_excursion_ratio)
        baseline_stability = float(self.selector.minimum_direction_stability)
        baseline_ridge = max(0.01, float(self.selector.ridge))
        baseline_factors = max(2, int(self.selector.maximum_factors))
        return {
            "ridge": {
                "kind": "float",
                "low": max(0.05, baseline_ridge / 5.0),
                "high": min(20.0, baseline_ridge * 5.0),
                "log": True,
            },
            "maximum_factors": {
                "kind": "int",
                "low": max(2, baseline_factors - 4),
                "high": min(len(SELECTOR_FEATURES), baseline_factors + 4, 12),
                "step": 1,
            },
            "minimum_geometry_probability": {
                "kind": "float",
                "low": baseline_probability,
                "high": min(0.90, baseline_probability + 0.18),
                "step": 0.01,
            },
            "minimum_expected_normal_net_bps": {
                "kind": "float",
                "low": baseline_normal,
                "high": baseline_normal + 80.0,
                "step": 5.0,
            },
            "minimum_expected_stressed_net_bps": {
                "kind": "float",
                "low": baseline_stressed,
                "high": baseline_stressed + 50.0,
                "step": 5.0,
            },
            "minimum_expected_mfe_cost_multiple": {
                "kind": "float",
                "low": baseline_cost_multiple,
                "high": baseline_cost_multiple + 2.0,
                "step": 0.1,
            },
            "minimum_expected_mfe_to_mae": {
                "kind": "float",
                "low": baseline_excursion,
                "high": baseline_excursion + 1.0,
                "step": 0.05,
            },
            "minimum_return_direction_stability": {
                "kind": "float",
                "low": baseline_stability,
                "high": min(0.95, baseline_stability + 0.30),
                "step": 0.01,
            },
        }

    def _outer_split(
        self,
        rows: list[dict[str, Any]],
        *,
        calibration_fraction: float = 0.70,
        minimum_holdout_observations: int | None = None,
    ) -> OuterSplit:
        all_groups = _group_rows(rows)
        groups = _group_rows(rows, required_horizons=self.selector.horizons)
        minimum_holdout = int(
            minimum_holdout_observations
            if minimum_holdout_observations is not None
            else max(20, self.selector.minimum_oos_selected)
        )
        minimum_calibration = max(self.selector.minimum_train * 3, 60)
        if len(groups) < minimum_calibration:
            coverage: dict[int, int] = {int(h): 0 for h in self.selector.horizons}
            for group in all_groups:
                present = {int(row["horizon_hours"]) for row in group}
                for horizon in coverage:
                    coverage[horizon] += int(horizon in present)
            raise ValueError(
                "INSUFFICIENT_COMPLETE_HORIZON_OBSERVATIONS_FOR_NESTED_OPTUNA: "
                f"have_complete={len(groups)} need_calibration>={minimum_calibration} "
                f"have_any={len(all_groups)} horizon_coverage={coverage}"
            )

        # Freeze the calibration prefix at the minimum size. Because completion of
        # 24/72/168h labels is chronological, this prefix remains stable as new
        # prospective evidence arrives. The outer holdout is never allowed to
        # change the Optuna search dataset.
        cut = minimum_calibration
        calibration = groups[:cut]
        calibration_ids = tuple(str(group[0]["observation_id"]) for group in calibration)
        calibration_id_set = set(calibration_ids)
        calibration_rows = [
            row for row in rows if str(row["observation_id"]) in calibration_id_set
        ]
        maturity_cutoff = max(row["matured_dt"] for row in calibration_rows)
        holdout_groups = [
            group for group in groups[cut:] if group[0]["observed_dt"] > maturity_cutoff
        ]
        # The outer holdout is allowed to accumulate after optimization. Requiring
        # it before Optuna would unnecessarily delay a leakage-safe calibration
        # search. It remains untouched until enough embargoed observations exist.
        _ = minimum_holdout
        holdout_ids = tuple(str(group[0]["observation_id"]) for group in holdout_groups)
        return OuterSplit(
            calibration_ids=calibration_ids,
            holdout_ids=holdout_ids,
            calibration_observations=len(calibration_ids),
            holdout_observations=len(holdout_ids),
            label_maturity_cutoff=maturity_cutoff.isoformat(),
            holdout_start=(
                holdout_groups[0][0]["observed_dt"].isoformat()
                if holdout_groups
                else None
            ),
        )

    @staticmethod
    def _folds(
        ids: tuple[str, ...],
        *,
        fold_count: int,
        minimum_train: int,
    ) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
        if fold_count < 2:
            raise ValueError("fold_count must be at least 2")
        start = max(minimum_train, len(ids) // 3)
        evaluation = list(ids[start:])
        if len(evaluation) < fold_count:
            raise ValueError("insufficient calibration observations for inner folds")
        chunks = [
            tuple(str(x) for x in chunk.tolist())
            for chunk in np.array_split(np.asarray(evaluation), fold_count)
            if len(chunk)
        ]
        positions = {value: index for index, value in enumerate(ids)}
        folds: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        for chunk in chunks:
            first = positions[chunk[0]]
            train = tuple(ids[:first])
            folds.append((train, chunk))
        return folds

    def _fit_models_before(
        self,
        rows: list[dict[str, Any]],
        *,
        train_ids: tuple[str, ...],
        decision_time,
    ) -> dict[int, dict[str, Any]]:
        train_set = set(train_ids)
        models: dict[int, dict[str, Any]] = {}
        for horizon in self.selector.horizons:
            prior = [
                row
                for row in rows
                if str(row["observation_id"]) in train_set
                and int(row["horizon_hours"]) == int(horizon)
                and row["matured_dt"] <= decision_time
            ]
            models[int(horizon)] = self.selector._fit_model(prior, int(horizon))
        return models

    def _evaluate_fixed_models(
        self,
        rows: list[dict[str, Any]],
        *,
        eval_ids: tuple[str, ...],
        models: Mapping[int, Mapping[str, Any]],
    ) -> dict[str, Any]:
        eval_set = set(eval_ids)
        groups = [
            group for group in _group_rows(rows) if str(group[0]["observation_id"]) in eval_set
        ]
        normal: list[float] = []
        stressed: list[float] = []
        selected_horizons: dict[str, int] = {}

        for group in groups:
            context = group[0]["context"]
            candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
            current_by_horizon = {int(row["horizon_hours"]): row for row in group}
            for horizon in self.selector.horizons:
                current = current_by_horizon.get(int(horizon))
                if current is None:
                    continue
                decision = self.selector._decision(
                    dict(models.get(int(horizon)) or {}),
                    context,
                    normal_cost_bps=float(current["normal_cost_bps"]),
                )
                if decision.get("passes"):
                    candidates.append((float(decision.get("utility") or -1e12), decision, current))
            if not candidates:
                continue
            _, decision, current = max(candidates, key=lambda item: item[0])
            normal.append(float(current["normal_net_bps"]))
            stressed.append(float(current["stressed_net_bps"]))
            key = str(decision.get("horizon_hours") or current["horizon_hours"])
            selected_horizons[key] = selected_horizons.get(key, 0) + 1

        normal_summary = _summary(normal)
        stressed_summary = _summary(stressed)
        return {
            "evaluated_observations": len(groups),
            "selected": len(normal),
            "selection_fraction": len(normal) / len(groups) if groups else 0.0,
            "normal_net": normal_summary,
            "stressed_net": stressed_summary,
            "selected_horizon_counts": selected_horizons,
        }

    def _fold_score(self, result: Mapping[str, Any]) -> float:
        selected = int(result.get("selected") or 0)
        evaluated = int(result.get("evaluated_observations") or 0)
        normal = dict(result.get("normal_net") or {})
        stressed = dict(result.get("stressed_net") or {})
        if selected == 0 or evaluated == 0:
            return -10000.0
        normal_mean = _float(normal.get("mean_bps"), -1000.0)
        stressed_mean = _float(stressed.get("mean_bps"), -1000.0)
        positive = _float(normal.get("positive_fraction"), 0.0)
        pf = max(0.01, _float(normal.get("profit_factor"), 0.01))
        selection_fraction = selected / evaluated
        score = (
            stressed_mean
            + 0.35 * normal_mean
            + 25.0 * (positive - 0.50)
            + 8.0 * math.log(min(pf, 10.0))
        )
        minimum_selected = max(5, self.selector.minimum_oos_selected // 3)
        if selected < minimum_selected:
            score -= 5.0 * (minimum_selected - selected)
        if selection_fraction > self.selector.maximum_selection_fraction:
            score -= 500.0 * (
                selection_fraction - self.selector.maximum_selection_fraction
            )
        if selection_fraction < 0.02:
            score -= 50.0
        return float(score)

    def _evaluate_calibration(
        self,
        rows: list[dict[str, Any]],
        split: OuterSplit,
        params: Mapping[str, Any],
        *,
        fold_count: int = 3,
    ) -> dict[str, Any]:
        folds = self._folds(
            split.calibration_ids,
            fold_count=fold_count,
            minimum_train=self.selector.minimum_train,
        )
        results: list[dict[str, Any]] = []
        scores: list[float] = []
        with _selector_parameters(self.selector, params):
            for index, (train_ids, eval_ids) in enumerate(folds):
                first_eval = next(
                    group
                    for group in _group_rows(rows)
                    if str(group[0]["observation_id"]) == eval_ids[0]
                )
                decision_time = first_eval[0]["observed_dt"]
                models = self._fit_models_before(
                    rows,
                    train_ids=train_ids,
                    decision_time=decision_time,
                )
                result = self._evaluate_fixed_models(
                    rows,
                    eval_ids=eval_ids,
                    models=models,
                )
                score = self._fold_score(result)
                result = {"fold": index, "score": score, **result}
                results.append(result)
                scores.append(score)

        if not scores:
            return {"score": -10000.0, "folds": [], "status": "BLOCKED"}
        robust = float(median(scores) - 0.50 * np.std(np.asarray(scores, dtype=float)))
        selected_total = sum(int(row.get("selected") or 0) for row in results)
        if selected_total < self.selector.minimum_oos_selected:
            robust -= 5.0 * (self.selector.minimum_oos_selected - selected_total)
        return {
            "status": "READY",
            "score": robust,
            "folds": results,
            "selected_total": selected_total,
            "fold_score_median": float(median(scores)),
            "fold_score_std": float(np.std(np.asarray(scores, dtype=float))),
        }

    def _evaluate_holdout(
        self,
        rows: list[dict[str, Any]],
        split: OuterSplit,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        holdout_set = set(split.holdout_ids)
        first_group = next(
            group for group in _group_rows(rows) if str(group[0]["observation_id"]) in holdout_set
        )
        decision_time = first_group[0]["observed_dt"]
        with _selector_parameters(self.selector, params):
            bootstrap_before = self.selector.bootstrap_draws
            self.selector.bootstrap_draws = max(bootstrap_before, 3000)
            try:
                models = self._fit_models_before(
                    rows,
                    train_ids=split.calibration_ids,
                    decision_time=decision_time,
                )
                result = self._evaluate_fixed_models(
                    rows,
                    eval_ids=split.holdout_ids,
                    models=models,
                )
            finally:
                self.selector.bootstrap_draws = bootstrap_before
        return {
            "status": "READY",
            "untouched_by_optuna": True,
            "adaptive_retraining_on_holdout": False,
            **result,
        }

    def run(
        self,
        *,
        trials: int = 40,
        study_name: str = "prospective-swing-entry-selector-v028",
        seed: int = 28,
        fold_count: int = 3,
    ) -> dict[str, Any]:
        if trials < 1:
            raise ValueError("trials must be positive")
        rows = self.selector._load_rows()
        try:
            split = self._outer_split(rows)
        except ValueError as exc:
            return {
                "schema_version": self.SCHEMA,
                "status": "COLLECTING",
                "qualified": False,
                "reason_codes": [str(exc)],
                "observations": len(
                    _group_rows(rows, required_horizons=self.selector.horizons)
                ),
                "observations_any_required_label": len(_group_rows(rows)),
                "complete_horizons_required": list(self.selector.horizons),
                "authority": "RESEARCH_ONLY",
                "automatic_policy_activation": False,
                "live_decision_influence": False,
                "orders_generated": 0,
                "orders_submitted": 0,
            }

        search_space = self.search_space()
        calibration_id_hash = hashlib.sha256(
            "|".join(split.calibration_ids).encode("utf-8")
        ).hexdigest()
        metadata = {
            "selector_schema": self.selector.SCHEMA,
            "horizons": list(self.selector.horizons),
            "features": list(SELECTOR_FEATURES),
            "calibration_observations": split.calibration_observations,
            "calibration_ids_hash": calibration_id_hash,
            "label_maturity_cutoff": split.label_maturity_cutoff,
            "complete_horizons_required": True,
            "outer_holdout_used_for_search": False,
        }
        ensured = self.runtime.ensure_study(
            study_name=study_name,
            search_space=search_space,
            objective_version=OBJECTIVE_VERSION,
            metadata=metadata,
            seed=seed,
        )
        contract_hash = str(ensured["contract_hash"])
        recovery = self.runtime.recover_running(
            study_name=study_name,
            contract_hash=contract_hash,
        )
        before = self.runtime.summary(
            study_name=study_name,
            contract_hash=contract_hash,
        )
        trial_budget = _trial_budget(before, int(trials))

        local_trials: list[dict[str, Any]] = []
        for _ in range(int(trial_budget["remaining"])):
            asked = self.runtime.ask(
                study_name=study_name,
                contract_hash=contract_hash,
                search_space=search_space,
                seed=seed,
            )
            number = int(asked["trial_number"])
            params = dict(asked.get("params") or {})
            try:
                evaluation = self._evaluate_calibration(
                    rows,
                    split,
                    params,
                    fold_count=fold_count,
                )
                score = float(evaluation["score"])
                told = self.runtime.tell(
                    study_name=study_name,
                    contract_hash=contract_hash,
                    trial_number=number,
                    value=score,
                )
                local_trials.append(
                    {
                        "trial_number": number,
                        "params": params,
                        "score": score,
                        "evaluation": evaluation,
                        "state": told.get("status"),
                    }
                )
            except Exception as exc:
                self.runtime.fail(
                    study_name=study_name,
                    contract_hash=contract_hash,
                    trial_number=number,
                    reason=f"{type(exc).__name__}:{str(exc)[:400]}",
                )
                local_trials.append(
                    {
                        "trial_number": number,
                        "params": params,
                        "state": "FAIL",
                        "error": f"{type(exc).__name__}:{str(exc)[:400]}",
                    }
                )

        study = self.runtime.summary(
            study_name=study_name,
            contract_hash=contract_hash,
        )
        best = dict(study.get("best") or {})
        minimum_holdout = max(20, self.selector.minimum_oos_selected)
        reason_codes: list[str] = []
        if not best:
            status = "BLOCKED"
            optimization_status = "BLOCKED"
            holdout = {"status": "NOT_EVALUATED", "untouched_by_optuna": True}
        elif split.holdout_observations < minimum_holdout:
            status = "COLLECTING"
            optimization_status = "COMPLETE"
            reason_codes.append(
                "OUTER_OOS_EVIDENCE_STILL_ACCUMULATING: "
                f"have={split.holdout_observations} need>={minimum_holdout}"
            )
            holdout = {
                "status": "COLLECTING",
                "untouched_by_optuna": True,
                "adaptive_retraining_on_holdout": False,
                "observations": split.holdout_observations,
                "required_observations": minimum_holdout,
            }
        else:
            status = "COMPLETE"
            optimization_status = "COMPLETE"
            holdout = self._evaluate_holdout(rows, split, dict(best.get("params") or {}))

        payload = {
            "schema_version": self.SCHEMA,
            "generated_at_unix": time.time(),
            "status": status,
            "optimization_status": optimization_status,
            "reason_codes": reason_codes,
            "study_name": study_name,
            "contract_hash": contract_hash,
            "objective_version": OBJECTIVE_VERSION,
            "search_space": search_space,
            "split": {
                "calibration_observations": split.calibration_observations,
                "holdout_observations": split.holdout_observations,
                "label_maturity_cutoff": split.label_maturity_cutoff,
                "holdout_start": split.holdout_start,
                "outer_holdout_used_for_search": False,
            },
            "recovery": recovery,
            "trial_budget": {
                **trial_budget,
                "executed_this_run": len(local_trials),
                "semantics": "TARGET_TOTAL_FINISHED_TRIALS",
            },
            "new_trials": local_trials,
            "study": study,
            "best": best,
            "outer_holdout": holdout,
            "authority": "RESEARCH_ONLY",
            "automatic_policy_activation": False,
            "paper_shadow_influence": False,
            "live_decision_influence": False,
            "automatic_live_promotion": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        output = self.root / "selector_optuna_latest.json"
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return payload
