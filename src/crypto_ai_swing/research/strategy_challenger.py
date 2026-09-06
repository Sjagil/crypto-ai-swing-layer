from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.research.performance_attribution import (
    PerformanceAttributionEngine,
    bayesian_mean_posterior,
    bootstrap_difference_probability,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _seed(label: str) -> int:
    return int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)


def _utc_ts(raw: Any) -> pd.Timestamp:
    value = pd.Timestamp(raw)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _catalog() -> list[dict[str, Any]]:
    return [
        {"name": "BASELINE", "kind": "all", "description": "All causal unblocked BUY observations."},
        {"name": "MTF_GE_065", "kind": "component_ge", "component": "mtf", "threshold": 0.65},
        {"name": "MTF_GE_070", "kind": "component_ge", "component": "mtf", "threshold": 0.70},
        {"name": "TECHNICAL_GE_060", "kind": "component_ge", "component": "technical", "threshold": 0.60},
        {"name": "EXECUTION_GE_060", "kind": "component_ge", "component": "execution", "threshold": 0.60},
        {"name": "CMC_GE_070", "kind": "component_ge", "component": "cmc", "threshold": 0.70},
        {"name": "RISK_GE_060", "kind": "component_ge", "component": "risk", "threshold": 0.60},
        {"name": "MTF_EXECUTION", "kind": "components_all_ge", "thresholds": {"mtf": 0.68, "execution": 0.55}},
        {"name": "MTF_CMC", "kind": "components_all_ge", "thresholds": {"mtf": 0.68, "cmc": 0.68}},
        {"name": "ALIGNMENT_GE_075", "kind": "descriptor_ge", "field": "mtf_alignment", "threshold": 0.75},
        {"name": "SETUP_TRIGGER_GE_070", "kind": "descriptors_all_ge", "thresholds": {"mtf_setup": 0.70, "mtf_trigger": 0.70}},
        {"name": "NO_OVEREXTENSION", "kind": "not_overextended"},
        {"name": "H1_RSI_LE_75", "kind": "descriptor_le", "field": "h1_rsi", "threshold": 75.0},
        {"name": "M15_RSI_LE_78", "kind": "descriptor_le", "field": "m15_rsi", "threshold": 78.0},
        {"name": "FAMILY_TREND_CONTINUATION", "kind": "family", "value": "TREND_CONTINUATION"},
        {"name": "FAMILY_BREAKOUT_CONFIRMATION", "kind": "family", "value": "BREAKOUT_CONFIRMATION"},
        {"name": "FAMILY_TREND_PULLBACK", "kind": "family", "value": "TREND_PULLBACK"},
        {"name": "FAMILY_RALLY_ACCELERATION", "kind": "family", "value": "RALLY_ACCELERATION"},
        {"name": "STATE_TREND_CONTINUATION", "kind": "breakout_state", "value": "TREND_CONTINUATION"},
        {"name": "STATE_TREND_PULLBACK", "kind": "breakout_state", "value": "TREND_PULLBACK"},
        {"name": "STATE_DONCHIAN_55_CONFIRMED", "kind": "breakout_state", "value": "DONCHIAN_55_CONFIRMED"},
        {"name": "MTF_GE_075", "kind": "component_ge", "component": "mtf", "threshold": 0.75},
        {"name": "SPREAD_LE_6", "kind": "descriptor_le", "field": "spread_bps", "threshold": 6.0},
        {"name": "SPREAD_LE_10", "kind": "descriptor_le", "field": "spread_bps", "threshold": 10.0},
        {"name": "DONCHIAN55_MTF_GE_070", "kind": "all_of", "rules": [
            {"kind": "breakout_state", "value": "DONCHIAN_55_CONFIRMED"},
            {"kind": "component_ge", "component": "mtf", "threshold": 0.70},
        ]},
        {"name": "DONCHIAN55_MTF_GE_075", "kind": "all_of", "rules": [
            {"kind": "breakout_state", "value": "DONCHIAN_55_CONFIRMED"},
            {"kind": "component_ge", "component": "mtf", "threshold": 0.75},
        ]},
        {"name": "DONCHIAN55_NO_OVEREXTENSION", "kind": "all_of", "rules": [
            {"kind": "breakout_state", "value": "DONCHIAN_55_CONFIRMED"},
            {"kind": "not_overextended"},
        ]},
        {"name": "MTF_GE_075_NO_OVEREXTENSION", "kind": "all_of", "rules": [
            {"kind": "component_ge", "component": "mtf", "threshold": 0.75},
            {"kind": "not_overextended"},
        ]},
        {"name": "MTF_GE_075_H1_RSI_LE_75", "kind": "all_of", "rules": [
            {"kind": "component_ge", "component": "mtf", "threshold": 0.75},
            {"kind": "descriptor_le", "field": "h1_rsi", "threshold": 75.0},
        ]},
        {"name": "MTF_GE_075_SPREAD_LE_10", "kind": "all_of", "rules": [
            {"kind": "component_ge", "component": "mtf", "threshold": 0.75},
            {"kind": "descriptor_le", "field": "spread_bps", "threshold": 10.0},
        ]},
        {"name": "DONCHIAN55_MTF_GE_070_SPREAD_LE_10", "kind": "all_of", "rules": [
            {"kind": "breakout_state", "value": "DONCHIAN_55_CONFIRMED"},
            {"kind": "component_ge", "component": "mtf", "threshold": 0.70},
            {"kind": "descriptor_le", "field": "spread_bps", "threshold": 10.0},
        ]},
        {"name": "FAMILY_BREAKOUT_CONFIRMATION_MTF_GE_070", "kind": "all_of", "rules": [
            {"kind": "family", "value": "BREAKOUT_CONFIRMATION"},
            {"kind": "component_ge", "component": "mtf", "threshold": 0.70},
        ]},
        {"name": "FAMILY_RALLY_ACCELERATION_NO_OVEREXTENSION", "kind": "all_of", "rules": [
            {"kind": "family", "value": "RALLY_ACCELERATION"},
            {"kind": "not_overextended"},
        ]},
    ]


def _catalog_hash(catalog: list[dict[str, Any]]) -> str:
    raw = json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def candidate_passes(candidate: Mapping[str, Any], row_or_context: Mapping[str, Any]) -> bool:
    if "components" in row_or_context and "descriptors" in row_or_context:
        components = dict(row_or_context.get("components") or {})
        desc = dict(row_or_context.get("descriptors") or {})
    else:
        from crypto_ai_swing.agents.component_features import extract_components, extract_descriptors
        components = extract_components(row_or_context)
        desc = extract_descriptors(row_or_context)

    kind = str(candidate.get("kind") or "")
    if kind == "all":
        return True
    if kind == "all_of":
        rules = list(candidate.get("rules") or [])
        return bool(rules) and all(
            candidate_passes(rule, row_or_context)
            for rule in rules
            if isinstance(rule, Mapping)
        ) and all(isinstance(rule, Mapping) for rule in rules)
    if kind == "component_ge":
        value = components.get(str(candidate.get("component")))
        return value is not None and _f(value, -1.0) >= _f(candidate.get("threshold"))
    if kind == "components_all_ge":
        for name, threshold in dict(candidate.get("thresholds") or {}).items():
            value = components.get(str(name))
            if value is None or _f(value, -1.0) < _f(threshold):
                return False
        return True
    if kind == "descriptor_ge":
        value = desc.get(str(candidate.get("field")))
        return value is not None and _f(value, -1e9) >= _f(candidate.get("threshold"))
    if kind == "descriptor_le":
        value = desc.get(str(candidate.get("field")))
        return value is not None and _f(value, 1e9) <= _f(candidate.get("threshold"))
    if kind == "descriptors_all_ge":
        for name, threshold in dict(candidate.get("thresholds") or {}).items():
            value = desc.get(str(name))
            if value is None or _f(value, -1e9) < _f(threshold):
                return False
        return True
    if kind == "not_overextended":
        return not bool(desc.get("overextended", False))
    if kind == "family":
        return str(desc.get("strategy_family")) == str(candidate.get("value"))
    if kind == "breakout_state":
        return str(desc.get("breakout_state")) == str(candidate.get("value"))
    return False


class StrategyChallengerLab:
    """Leakage-aware stateful champion/challenger research lab.

    Candidate discovery and validation use only observations available before a
    freeze timestamp. A frozen candidate is then judged only on observations
    that arrive later. A failed prospective test is retired; its test outcomes
    may be used for future discovery only after a new candidate is frozen at a
    later timestamp. Promotion is research-only.
    """

    SCHEMA = "crypto_ai_swing_strategy_challenger_lab_v2"

    def __init__(self, settings, *, mode: str = "paper") -> None:
        self.settings = settings
        self.mode = str(mode)
        self.bridge = CryptoLibraryBridge(settings.crypto_repo_root)
        self.attribution = PerformanceAttributionEngine(settings, mode=mode)
        cfg = dict((getattr(settings, "autonomy", {}) or {}).get("strategy_lab", {}) or {})
        self.minimum_discovery = int(cfg.get("minimum_discovery_observations", 80))
        self.minimum_validation_selected = int(cfg.get("minimum_validation_selected", 20))
        self.minimum_prospective_selected = int(cfg.get("minimum_prospective_selected", 30))
        self.minimum_prospective_span_hours = float(cfg.get("minimum_prospective_span_hours", 72.0))
        self.minimum_positive_fraction = float(cfg.get("minimum_positive_fraction", 0.55))
        self.minimum_uplift_bps = float(cfg.get("minimum_uplift_bps", 5.0))
        self.multiple_testing_alpha = float(cfg.get("multiple_testing_alpha", 0.05))
        self.root = Path(settings.project_root) / "output/crypto_ai_swing/research/strategy_lab"
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.root / "latest.json"
        self.state_path = self.root / "state.json"
        self.trials_path = self.root / "trials.json"
        self.champion_path = self.root / "champion.json"
        self.catalog = _catalog()
        self.catalog_by_name = {row["name"]: row for row in self.catalog}
        self._sync_trials()

    def _load_json(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def _write(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

    def _sync_trials(self) -> None:
        current = self._load_json(self.trials_path)
        known = dict(current.get("trials") or {})
        now = _now()
        for candidate in self.catalog:
            name = str(candidate["name"])
            if name not in known:
                known[name] = {
                    "first_registered_at": now,
                    "definition": candidate,
                    "definition_hash": hashlib.sha256(
                        json.dumps(candidate, sort_keys=True).encode()
                    ).hexdigest(),
                }
        payload = {
            "schema_version": "crypto_ai_swing_strategy_trials_v1",
            "catalog_hash": _catalog_hash(self.catalog),
            "known_trial_count": len(known),
            "trials": known,
        }
        self._write(self.trials_path, payload)

    def _known_trials(self) -> int:
        return int(self._load_json(self.trials_path).get("known_trial_count") or len(self.catalog))

    @staticmethod
    def _span_hours(rows: list[dict[str, Any]]) -> float:
        if len(rows) < 2:
            return 0.0
        try:
            values = pd.to_datetime([r["observed_at"] for r in rows], utc=True)
            return float((values.max() - values.min()).total_seconds() / 3600.0)
        except Exception:
            return 0.0

    def _stochastic(self, selected: list[dict[str, Any]]) -> dict[str, Any]:
        if len(selected) < self.minimum_prospective_selected:
            return {
                "status": "COLLECTING",
                "passed": False,
                "observations": len(selected),
                "required": self.minimum_prospective_selected,
            }
        try:
            module = self.bridge.import_module("research.stochastic_validation")
            policy = module.StochasticValidationPolicy(
                simulations=10_000,
                expected_block_length=max(3, min(10, len(selected) // 6)),
                maximum_drawdown=0.15,
                maximum_drawdown_breach_probability=0.01,
                maximum_terminal_loss_probability=0.05,
                minimum_p05_total_return=0.0,
                dirichlet_blocks=8,
                dirichlet_concentrations=(0.5, 1.0, 5.0),
                minimum_observations=self.minimum_prospective_selected,
                confidence_level=0.95,
                seed=2404,
                batch_size=256,
            )
            normal = np.asarray([r["normal_net_bps"] for r in selected], dtype=float) / 10_000.0
            stressed = np.asarray([r["stressed_net_bps"] for r in selected], dtype=float) / 10_000.0
            return module.validate_strategy_return_paths(
                normal,
                stressed,
                policy=policy,
                seed_offset=0,
            )
        except Exception as exc:
            return {
                "status": "ERROR",
                "passed": False,
                "error": f"{type(exc).__name__}:{str(exc)[:300]}",
            }

    def _metrics(self, rows: list[dict[str, Any]], candidate: Mapping[str, Any]) -> dict[str, Any]:
        selected = [r for r in rows if candidate_passes(candidate, r)]
        baseline = np.asarray([r["normal_net_bps"] for r in rows], dtype=float)
        normal = np.asarray([r["normal_net_bps"] for r in selected], dtype=float)
        stressed = np.asarray([r["stressed_net_bps"] for r in selected], dtype=float)
        baseline_mean = float(baseline.mean()) if len(baseline) else None
        selected_mean = float(normal.mean()) if len(normal) else None
        improvement = (
            selected_mean - baseline_mean
            if selected_mean is not None and baseline_mean is not None
            else None
        )
        se = float(normal.std(ddof=1) / np.sqrt(len(normal))) if len(normal) > 1 else None
        return {
            "observations_total": len(rows),
            "selected": len(selected),
            "selection_fraction": len(selected) / max(1, len(rows)),
            "span_hours": self._span_hours(selected),
            "mean_normal_net_bps": selected_mean,
            "median_normal_net_bps": float(np.median(normal)) if len(normal) else None,
            "mean_stressed_net_bps": float(stressed.mean()) if len(stressed) else None,
            "positive_fraction": float(np.mean(normal > 0.0)) if len(normal) else None,
            "baseline_mean_normal_net_bps": baseline_mean,
            "mean_uplift_vs_baseline_bps": improvement,
            "one_sided_90_lcb_bps": (
                selected_mean - 1.645 * se
                if selected_mean is not None and se is not None
                else None
            ),
            "bayesian": bayesian_mean_posterior(
                normal,
                draws=5000,
                seed=2405 + (_seed(f"candidate:{candidate.get('name')}:bayes") % 10000),
            ),
            "uplift_bootstrap": bootstrap_difference_probability(
                normal,
                baseline,
                draws=3000,
                seed=2406 + (_seed(f"candidate:{candidate.get('name')}:uplift") % 10000),
            ),
            "selected_rows": selected,
        }

    @staticmethod
    def _strip_rows(metrics: dict[str, Any]) -> dict[str, Any]:
        value = dict(metrics)
        value.pop("selected_rows", None)
        return value

    def _discover(self, rows: list[dict[str, Any]], state: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        split = max(1, int(len(rows) * 0.70))
        discovery = rows[:split]
        validation = rows[split:]
        retired = {str(x.get("name")) for x in state.get("retired_candidates", [])}
        champion_name = str((state.get("champion") or {}).get("name") or "")
        ranked = []
        for candidate in self.catalog:
            name = str(candidate["name"])
            if name == "BASELINE" or name in retired or name == champion_name:
                continue
            dev = self._metrics(discovery, candidate)
            val = self._metrics(validation, candidate)
            dev_mean = dev.get("mean_normal_net_bps")
            val_mean = val.get("mean_normal_net_bps")
            objective = -1e12
            if (
                dev.get("selected", 0) >= 20
                and val.get("selected", 0) >= self.minimum_validation_selected
                and dev_mean is not None
                and val_mean is not None
            ):
                objective = (
                    float(val.get("one_sided_90_lcb_bps") or -1e6)
                    + 0.20 * float(val.get("mean_uplift_vs_baseline_bps") or 0.0)
                    + 0.10 * float(dev_mean)
                )
            provisional = {
                "validation_selected": val.get("selected", 0) >= self.minimum_validation_selected,
                "validation_positive_mean": val_mean is not None and float(val_mean) > 0.0,
                "validation_positive_fraction": float(val.get("positive_fraction") or 0.0) >= self.minimum_positive_fraction,
                "validation_bayesian": float((val.get("bayesian") or {}).get("probability_mean_positive", 0.0)) >= 0.90,
                "validation_uplift": float(val.get("mean_uplift_vs_baseline_bps") or -1e9) > 0.0,
            }
            ranked.append({
                "name": name,
                "definition": candidate,
                "objective": objective,
                "provisional_checks": provisional,
                "provisional_pass": all(provisional.values()),
                "discovery": self._strip_rows(dev),
                "validation": self._strip_rows(val),
            })
        ranked.sort(key=lambda x: float(x["objective"]), reverse=True)
        winner = next((x for x in ranked if x["provisional_pass"]), None)
        return winner, ranked

    def _prospective_checks(
        self,
        metrics: dict[str, Any],
        stochastic: dict[str, Any],
    ) -> dict[str, bool]:
        known = max(1, self._known_trials())
        probability_threshold = max(0.95, 1.0 - self.multiple_testing_alpha / known)
        posterior = dict(metrics.get("bayesian") or {})
        return {
            "minimum_selected": int(metrics.get("selected") or 0) >= self.minimum_prospective_selected,
            "minimum_time_span": float(metrics.get("span_hours") or 0.0) >= self.minimum_prospective_span_hours,
            "positive_net_mean": float(metrics.get("mean_normal_net_bps") or -1e9) > 0.0,
            "positive_stressed_mean": float(metrics.get("mean_stressed_net_bps") or -1e9) > 0.0,
            "positive_fraction": float(metrics.get("positive_fraction") or 0.0) >= self.minimum_positive_fraction,
            "bayesian_probability_adjusted": float(posterior.get("probability_mean_positive") or 0.0) >= probability_threshold,
            "bayesian_p05_positive": float(posterior.get("mean_p05_bps") or -1e9) > 0.0,
            "minimum_uplift": float(metrics.get("mean_uplift_vs_baseline_bps") or -1e9) >= self.minimum_uplift_bps,
            "stochastic_validation": bool(stochastic.get("passed", False)),
        }

    def refresh(self) -> dict[str, Any]:
        rows = self.attribution.load_forward_rows()
        state = self._load_json(self.state_path)
        state.setdefault("retired_candidates", [])
        frozen = state.get("frozen_candidate")
        champion = state.get("champion")
        known_trials = self._known_trials()

        if frozen:
            boundary = pd.Timestamp(str(frozen["frozen_after_observed_at"]))
            boundary = boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
            prospective = [
                r for r in rows
                if _utc_ts(r["observed_at"]) > boundary
            ]
            candidate = dict(frozen["definition"])
            metrics = self._metrics(prospective, candidate)
            selected_rows = list(metrics.get("selected_rows") or [])
            enough_to_judge = (
                int(metrics.get("selected") or 0) >= self.minimum_prospective_selected
                and float(metrics.get("span_hours") or 0.0) >= self.minimum_prospective_span_hours
            )
            stochastic = self._stochastic(selected_rows) if enough_to_judge else {
                "status": "COLLECTING", "passed": False,
                "observations": int(metrics.get("selected") or 0),
            }
            checks = self._prospective_checks(metrics, stochastic)
            qualified = all(checks.values())
            if enough_to_judge and qualified:
                new_champion = {
                    "name": frozen["name"],
                    "definition": candidate,
                    "promoted_at": _now(),
                    "promoted_after_observed_at": max(r["observed_at"] for r in rows),
                    "prospective": self._strip_rows(metrics),
                    "checks": checks,
                    "stochastic_validation": stochastic,
                    "scope": "RESEARCH_CHAMPION",
                    "live_decision_influence": False,
                }
                state["champion"] = new_champion
                state["frozen_candidate"] = None
                champion = new_champion
                self._write(self.champion_path, new_champion)
                status = "RESEARCH_CHAMPION"
            elif enough_to_judge:
                state["retired_candidates"].append({
                    "name": frozen["name"],
                    "retired_at": _now(),
                    "reason_codes": [k.upper() + "_FAILED" for k, ok in checks.items() if not ok],
                    "prospective": self._strip_rows(metrics),
                })
                state["frozen_candidate"] = None
                status = "CHALLENGER_RETIRED"
            else:
                status = "PROSPECTIVE_TEST_COLLECTING"

            payload = {
                "schema_version": self.SCHEMA,
                "generated_at": _now(),
                "status": status,
                "observations": len(rows),
                "known_trial_count": known_trials,
                "multiple_testing_probability_threshold": max(
                    0.95, 1.0 - self.multiple_testing_alpha / max(1, known_trials)
                ),
                "frozen_candidate": (
                    frozen if status == "PROSPECTIVE_TEST_COLLECTING" else state.get("frozen_candidate")
                ),
                "prospective_test": self._strip_rows(metrics),
                "prospective_checks": checks,
                "stochastic_validation": stochastic,
                "champion": state.get("champion"),
                "retired_candidate_count": len(state["retired_candidates"]),
                "live_decision_influence": False,
                "automatic_live_promotion": False,
                "orders_submitted": 0,
            }
            self._write(self.state_path, state)
            self._write(self.latest_path, payload)
            return payload

        # A champion remains immutable while a new challenger is collected.
        if champion:
            boundary_raw = champion.get("promoted_after_observed_at")
            boundary = pd.Timestamp(str(boundary_raw)) if boundary_raw else None
            if boundary is not None:
                boundary = boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
                discovery_rows = [r for r in rows if _utc_ts(r["observed_at"]) > boundary]
            else:
                discovery_rows = []
            if len(discovery_rows) < self.minimum_discovery:
                payload = {
                    "schema_version": self.SCHEMA,
                    "generated_at": _now(),
                    "status": "RESEARCH_CHAMPION",
                    "observations": len(rows),
                    "new_observations_since_champion": len(discovery_rows),
                    "minimum_discovery_observations": self.minimum_discovery,
                    "known_trial_count": known_trials,
                    "champion": champion,
                    "frozen_candidate": None,
                    "live_decision_influence": False,
                    "automatic_live_promotion": False,
                    "orders_submitted": 0,
                }
                self._write(self.latest_path, payload)
                return payload
            source_rows = discovery_rows
        else:
            source_rows = rows

        if len(source_rows) < self.minimum_discovery:
            payload = {
                "schema_version": self.SCHEMA,
                "generated_at": _now(),
                "status": "COLLECTING",
                "observations": len(rows),
                "discovery_observations": len(source_rows),
                "minimum_discovery_observations": self.minimum_discovery,
                "known_trial_count": known_trials,
                "catalog_hash": _catalog_hash(self.catalog),
                "champion": champion,
                "frozen_candidate": None,
                "live_decision_influence": False,
                "automatic_live_promotion": False,
                "orders_submitted": 0,
            }
            self._write(self.latest_path, payload)
            self._write(self.state_path, state)
            return payload

        winner, ranked = self._discover(source_rows, state)
        if winner is None:
            payload = {
                "schema_version": self.SCHEMA,
                "generated_at": _now(),
                "status": "NO_PROVISIONAL_CHALLENGER",
                "observations": len(rows),
                "known_trial_count": known_trials,
                "top_candidates": ranked[:10],
                "champion": champion,
                "live_decision_influence": False,
                "automatic_live_promotion": False,
                "orders_submitted": 0,
            }
            self._write(self.latest_path, payload)
            self._write(self.state_path, state)
            return payload

        freeze_boundary = max(r["observed_at"] for r in rows)
        frozen = {
            "name": winner["name"],
            "definition": winner["definition"],
            "frozen_at": _now(),
            "frozen_after_observed_at": freeze_boundary,
            "discovery": winner["discovery"],
            "validation": winner["validation"],
            "catalog_hash": _catalog_hash(self.catalog),
        }
        state["frozen_candidate"] = frozen
        self._write(self.state_path, state)
        payload = {
            "schema_version": self.SCHEMA,
            "generated_at": _now(),
            "status": "CHALLENGER_FROZEN",
            "observations": len(rows),
            "known_trial_count": known_trials,
            "frozen_candidate": frozen,
            "top_candidates": ranked[:10],
            "champion": champion,
            "prospective_test_observations": 0,
            "live_decision_influence": False,
            "automatic_live_promotion": False,
            "orders_submitted": 0,
        }
        self._write(self.latest_path, payload)
        return payload

    def status(self) -> dict[str, Any]:
        return self._load_json(self.latest_path) or {
            "schema_version": self.SCHEMA,
            "status": "NOT_BUILT",
        }

    def champion(self) -> dict[str, Any] | None:
        value = self._load_json(self.champion_path)
        return value or None

    def evaluate_context(self, context: Mapping[str, Any]) -> dict[str, Any]:
        champion = self.champion()
        if not champion:
            return {
                "status": "NO_CHAMPION",
                "passes": None,
                "champion": None,
                "live_decision_influence": False,
            }
        definition = dict(champion.get("definition") or {})
        return {
            "status": "RESEARCH_CHAMPION",
            "passes": candidate_passes(definition, context),
            "champion": champion.get("name"),
            "definition": definition,
            "live_decision_influence": False,
        }
