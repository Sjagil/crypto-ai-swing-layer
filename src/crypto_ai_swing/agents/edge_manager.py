from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from crypto_ai_swing.agents.component_features import (
    COMPONENTS,
    DEFAULT_PRIORS,
    extract_component,
    extract_components,
)

# Backward-compatible private alias retained for v0.22 tests/integrations.
_extract_component = extract_component
from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.research.performance_attribution import (
    bayesian_mean_posterior,
    bootstrap_difference_probability,
)
from crypto_ai_swing.research.net_edge_calibration import NetEdgeCalibrator
from crypto_ai_swing.research.strategy_challenger import StrategyChallengerLab
from crypto_ai_swing.research.swing_geometry import SwingGeometryEngine


def _net_costs(settings) -> tuple[float, float]:
    costs = dict(settings.execution.get("costs", {}) or {})
    fee = float(costs.get("fee_bps_per_side", 25.0))
    slippage = float(costs.get("base_slippage_bps", 2.0))
    cfg = dict((settings.agents.get("edge_manager", {}) or {}))
    spread = float(cfg.get("research_round_trip_spread_bps", 8.0))
    extra_stress = float(cfg.get("stressed_extra_cost_bps", 25.0))
    normal = 2.0 * (fee + slippage) + spread
    return normal, normal + extra_stress


def _normalize_weights(raw: Mapping[str, float], *, cap: float = 0.35) -> dict[str, float]:
    values = {name: max(0.0, float(raw.get(name, 0.0))) for name in COMPONENTS}
    total = sum(values.values())
    if total <= 0:
        return {name: 0.0 for name in COMPONENTS}
    values = {name: value / total for name, value in values.items()}
    capped = {name: min(cap, value) for name, value in values.items()}
    total = sum(capped.values())
    return {name: (value / total if total > 0 else 0.0) for name, value in capped.items()}


def _return_shape(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"positive_fraction": None, "profit_factor": None, "payoff_ratio": None}
    winners = arr[arr > 0.0]
    losers = arr[arr <= 0.0]
    gross_profit = float(winners.sum()) if len(winners) else 0.0
    gross_loss = abs(float(losers.sum())) if len(losers) else 0.0
    mean_winner = float(winners.mean()) if len(winners) else None
    mean_loser = float(losers.mean()) if len(losers) else None
    return {
        "positive_fraction": float(np.mean(arr > 0.0)),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else None,
        "payoff_ratio": (
            mean_winner / abs(mean_loser)
            if mean_winner is not None and mean_loser not in (None, 0.0)
            else None
        ),
    }


class ResearchEdgeManager:
    """Adaptive agent-of-agents with fail-closed prospective qualification.

    The manager can adapt weights in research and paper/shadow only after
    out-of-sample qualification. It can never grant live authority.
    """

    SCHEMA = "research_edge_manager_v4"

    def __init__(self, settings, mode: str = "shadow") -> None:
        self.settings = settings
        self.mode = str(mode).lower()
        self.bridge = CryptoLibraryBridge(settings.crypto_repo_root)
        self.root = settings.project_root / "output/crypto_ai_swing/agents/meta"
        self.latest_path = self.root / "latest.json"
        self.history_path = self.root / "history.jsonl"
        cfg = dict(settings.agents.get("edge_manager", {}) or {})
        self.refresh_seconds = float(cfg.get("refresh_seconds", 900))
        self.minimum_observations = int(cfg.get("minimum_observations", 80))
        self.minimum_holdout_selected = int(cfg.get("minimum_holdout_selected", 30))
        self.horizon_hours = int(cfg.get("horizon_hours", 4))
        self.minimum_positive_fraction = float(cfg.get("minimum_positive_fraction", 0.35))
        self.minimum_profit_factor = float(cfg.get("minimum_profit_factor", 1.20))
        self.minimum_payoff_ratio = float(cfg.get("minimum_payoff_ratio", 1.10))
        self.maximum_component_weight = float(cfg.get("maximum_component_weight", 0.35))
        self.minimum_component_coverage = float(cfg.get("minimum_component_coverage", 0.25))
        self._last_refresh = 0.0
        self._policy: dict[str, Any] = {}
        self.strategy_lab = StrategyChallengerLab(settings, mode=mode)
        self.calibrator = NetEdgeCalibrator(settings, mode=mode)
        self.swing_geometry = SwingGeometryEngine(settings, mode=mode)

    def _read_latest(self) -> dict[str, Any]:
        try:
            data = json.loads(self.latest_path.read_text(encoding="utf-8"))
            return dict(data) if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _row_costs(self, context: Mapping[str, Any]) -> tuple[float, float]:
        costs = dict(self.settings.execution.get("costs", {}) or {})
        fee = float(costs.get("fee_bps_per_side", 25.0))
        slip = float(costs.get("base_slippage_bps", 2.0))
        cfg = dict(self.settings.agents.get("edge_manager", {}) or {})
        fallback_spread = float(cfg.get("research_round_trip_spread_bps", 8.0))
        stress = float(cfg.get("stressed_extra_cost_bps", 25.0))
        screen = dict(context.get("universe_screen") or {})
        challenger = dict(context.get("mtf_challenger") or {})
        diagnostics = dict(challenger.get("diagnostics") or {})
        raw = screen.get("universe_spread_bps", diagnostics.get("spread_bps"))
        try:
            spread = float(raw)
            if not np.isfinite(spread) or spread < 0:
                spread = fallback_spread
        except (TypeError, ValueError):
            spread = fallback_spread
        normal = 2.0 * (fee + slip) + spread
        return normal, normal + stress

    def _load_rows(self, database_path: Path) -> list[dict[str, Any]]:
        if not database_path.is_file():
            return []
        conn = sqlite3.connect(database_path)
        try:
            signal_columns = {
                str(row[1])
                for row in conn.execute(
                    "PRAGMA table_info(signal_observations)"
                ).fetchall()
            }
            time_expr = (
                "s.observed_at"
                if "observed_at" in signal_columns
                else "o.matured_at"
            )
            rows = conn.execute(
                f"""
                SELECT {time_expr},s.market,s.context,o.return_bps
                FROM signal_observations s
                JOIN forward_outcomes_v2 o
                  ON o.observation_id=s.observation_id
                WHERE s.side='BUY'
                  AND s.blocked=0
                  AND o.horizon_hours=?
                ORDER BY {time_expr},s.market
                """,
                (int(getattr(self, "horizon_hours", 4)),),
            ).fetchall()
        finally:
            conn.close()
        output = []
        for observed_at, market, raw_context, return_bps in rows:
            try:
                context = json.loads(raw_context or "{}")
                gross = float(return_bps)
            except Exception:
                continue
            if not isinstance(context, dict) or not np.isfinite(gross):
                continue
            normal_cost, stressed_cost = self._row_costs(context)
            components = extract_components(context)
            if not any(value is not None for value in components.values()):
                continue
            output.append({
                "observed_at": str(observed_at),
                "market": str(market),
                "gross_return_bps": gross,
                "normal_net_bps": gross - normal_cost,
                "stressed_net_bps": gross - stressed_cost,
                "components": components,
                "context": context,
            })
        return output

    def _availability_priors(self, rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, Any]]:
        coverage = {}
        raw = {}
        n = max(1, len(rows))
        for name in COMPONENTS:
            count = sum(r["components"].get(name) is not None for r in rows)
            fraction = count / n
            coverage[name] = {"observations": count, "coverage_fraction": fraction}
            raw[name] = DEFAULT_PRIORS[name] if fraction >= getattr(self, "minimum_component_coverage", 0.25) else 0.0
        if sum(raw.values()) <= 0:
            raw["technical"] = DEFAULT_PRIORS["technical"]
            raw["mtf"] = DEFAULT_PRIORS["mtf"]
        return _normalize_weights(raw, cap=getattr(self, "maximum_component_weight", 0.35)), coverage

    def _learn_weights(self, rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, Any]]:
        returns = pd.Series([r["normal_net_bps"] for r in rows], dtype=float)
        raw: dict[str, float] = {}
        diagnostics: dict[str, Any] = {}
        n = max(1, len(rows))
        for name in COMPONENTS:
            values = pd.Series([r["components"].get(name) for r in rows], dtype="float64")
            mask = values.notna() & returns.notna()
            count = int(mask.sum())
            coverage = count / n
            if count < 20 or coverage < getattr(self, "minimum_component_coverage", 0.25):
                raw[name] = 0.0
                diagnostics[name] = {
                    "status": "INSUFFICIENT",
                    "observations": count,
                    "coverage_fraction": coverage,
                    "raw_trust": 0.0,
                }
                continue
            x = values[mask].to_numpy(dtype=float)
            y = returns[mask].to_numpy(dtype=float)
            corr = pd.Series(x).corr(pd.Series(y), method="spearman")
            corr = float(corr) if corr is not None and np.isfinite(corr) else 0.0
            median = float(np.median(x))
            high = y[x >= median]
            low = y[x < median]
            delta = bootstrap_difference_probability(
                high, low, draws=2000, seed=2410 + COMPONENTS.index(name)
            )
            p_lift = float(delta.get("probability_high_better") or 0.0)
            directional = max(0.0, corr) * max(0.0, 2.0 * (p_lift - 0.5))
            raw_trust = directional * coverage
            evidence_strength = min(1.0, count / 300.0)
            raw[name] = (
                evidence_strength * raw_trust
                + (1.0 - evidence_strength) * 0.15 * DEFAULT_PRIORS[name]
            )
            diagnostics[name] = {
                "status": "READY",
                "observations": count,
                "coverage_fraction": coverage,
                "spearman_ic": corr,
                "high_vs_low": delta,
                "raw_trust": raw_trust,
                "evidence_strength": evidence_strength,
            }
        if sum(raw.values()) <= 0:
            return self._availability_priors(rows)
        return _normalize_weights(raw, cap=getattr(self, "maximum_component_weight", 0.35)), diagnostics

    @staticmethod
    def _score(components: Mapping[str, float | None], weights: Mapping[str, float]) -> float:
        used = [
            (float(value), float(weights.get(name, 0.0)))
            for name, value in components.items()
            if value is not None and float(weights.get(name, 0.0)) > 0.0
        ]
        if not used:
            return 0.5
        total = sum(weight for _, weight in used)
        return float(np.clip(sum(value * weight for value, weight in used) / total, 0.0, 1.0))

    @classmethod
    def _select_threshold(cls, rows, weights) -> tuple[float, dict[str, Any]]:
        best = None
        for threshold in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
            selected = [r["normal_net_bps"] for r in rows if cls._score(r["components"], weights) >= threshold]
            if len(selected) < 20:
                continue
            arr = np.asarray(selected, dtype=float)
            se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 999.0
            lcb = float(arr.mean() - 1.645 * se)
            candidate = {
                "threshold": threshold,
                "selected": len(arr),
                "mean_net_bps": float(arr.mean()),
                "median_net_bps": float(np.median(arr)),
                "positive_fraction": float(np.mean(arr > 0.0)),
                "one_sided_90_lcb_bps": lcb,
                "objective": lcb + 0.10 * float(np.median(arr)),
            }
            if best is None or candidate["objective"] > best["objective"]:
                best = candidate
        if best is None:
            return 0.70, {"threshold": 0.70, "selected": 0, "status": "INSUFFICIENT_TRAIN_SELECTED"}
        return float(best["threshold"]), best

    def _stochastic(self, normal_bps: np.ndarray, stressed_bps: np.ndarray) -> dict[str, Any]:
        if len(normal_bps) < self.minimum_holdout_selected:
            return {"status": "COLLECTING", "passed": False, "observation_count": len(normal_bps), "required": self.minimum_holdout_selected}
        try:
            module = self.bridge.import_module("research.stochastic_validation")
            policy = module.StochasticValidationPolicy(
                simulations=10_000,
                expected_block_length=max(3, min(10, len(normal_bps) // 8)),
                maximum_drawdown=0.15,
                maximum_drawdown_breach_probability=0.01,
                maximum_terminal_loss_probability=0.05,
                minimum_p05_total_return=0.0,
                dirichlet_blocks=8,
                dirichlet_concentrations=(0.5, 1.0, 5.0),
                minimum_observations=self.minimum_holdout_selected,
                confidence_level=0.95,
                seed=2411,
                batch_size=256,
            )
            return module.validate_strategy_return_paths(
                normal_bps / 10_000.0,
                stressed_bps / 10_000.0,
                policy=policy,
                seed_offset=0,
            )
        except Exception as exc:
            return {"status": "ERROR", "passed": False, "error": f"{type(exc).__name__}:{str(exc)[:300]}"}

    def refresh_policy(self, database_path: Path, *, force: bool = False) -> dict[str, Any]:
        now = time.time()
        if not force and self._policy and now - self._last_refresh < self.refresh_seconds:
            return dict(self._policy)
        rows = self._load_rows(Path(database_path))
        fallback_normal, fallback_stressed = _net_costs(self.settings)
        strategy_lab = getattr(self, "strategy_lab", None)
        strategy = strategy_lab.status() if strategy_lab is not None else {"status": "NOT_BUILT"}
        calibrator = getattr(self, "calibrator", None)
        if calibrator is not None:
            calibration = calibrator.refresh(force=force)
            calibration_summary = calibrator.policy_summary(calibration)
        else:
            # Backward-compatible fallback for tests/integrations constructing
            # the manager via __new__ without the v0.25+ collaborators.
            calibration = {
                "status": "COLLECTING",
                "qualified": False,
                "observations": len(rows),
            }
            calibration_summary = {
                **calibration,
                "reason_codes": ["CALIBRATOR_NOT_INITIALIZED"],
                "live_decision_influence": False,
            }
        geometry_engine = getattr(self, "swing_geometry", None)
        if geometry_engine is not None:
            geometry = geometry_engine.refresh(force=force)
            geometry_summary = geometry_engine.policy_summary(geometry)
        else:
            geometry = {
                "status": "COLLECTING",
                "qualified": False,
                "observations": len(rows),
            }
            geometry_summary = {
                **geometry,
                "reason_codes": ["SWING_GEOMETRY_NOT_INITIALIZED"],
                "live_decision_influence": False,
            }

        if len(rows) < self.minimum_observations:
            weights, health = self._availability_priors(rows)
            policy = {
                "schema_version": self.SCHEMA,
                "generated_at": datetime.now(UTC).isoformat(),
                "status": "COLLECTING",
                "qualified": False,
                "observations": len(rows),
                "minimum_observations": self.minimum_observations,
                "weights": weights,
                "base_priors": DEFAULT_PRIORS,
                "component_health": health,
                "threshold": 0.70,
                "strategy_lab": {
                    "status": strategy.get("status"),
                    "champion": (strategy.get("champion") or {}).get("name") if isinstance(strategy.get("champion"), dict) else None,
                },
                "net_edge_calibration": calibration_summary,
                "swing_geometry": geometry_summary,
                "horizon_hours": int(getattr(self, "horizon_hours", 4)),
                "shadow_influence": False,
                "live_decision_influence": False,
                "automatic_live_promotion": False,
                "normal_round_trip_cost_bps_fallback": fallback_normal,
                "stressed_round_trip_cost_bps_fallback": fallback_stressed,
                "reason_codes": ["INSUFFICIENT_MATURED_META_OBSERVATIONS"],
            }
            self._persist(policy)
            return policy

        split = max(50, min(len(rows) - self.minimum_holdout_selected, int(len(rows) * 0.70)))
        train = rows[:split]
        holdout = rows[split:]
        weights, health = self._learn_weights(train)
        threshold, training_selection = self._select_threshold(train, weights)
        strategy_lab = getattr(self, "strategy_lab", None)
        champion = strategy_lab.champion() if strategy_lab is not None else None
        holdout_selected = []
        for row in holdout:
            if self._score(row["components"], weights) < threshold:
                continue
            if champion and strategy_lab is not None and not strategy_lab.evaluate_context(row["context"]).get("passes"):
                continue
            holdout_selected.append(row)

        normal_bps = np.asarray([r["normal_net_bps"] for r in holdout_selected], dtype=float)
        stressed_bps = np.asarray([r["stressed_net_bps"] for r in holdout_selected], dtype=float)
        posterior = bayesian_mean_posterior(normal_bps, draws=5000, seed=2412)
        stochastic = self._stochastic(normal_bps, stressed_bps)
        baseline = np.asarray([r["normal_net_bps"] for r in holdout], dtype=float)
        baseline_mean = float(baseline.mean()) if len(baseline) else None
        selected_mean = float(normal_bps.mean()) if len(normal_bps) else None
        improvement = selected_mean - baseline_mean if selected_mean is not None and baseline_mean is not None else None
        stressed_mean = float(stressed_bps.mean()) if len(stressed_bps) else None
        return_shape = _return_shape(normal_bps)
        checks = {
            "holdout_selected": len(holdout_selected) >= self.minimum_holdout_selected,
            "positive_normal_mean": selected_mean is not None and selected_mean > 0.0,
            "positive_stressed_mean": stressed_mean is not None and stressed_mean > 0.0,
            "positive_fraction": (
                return_shape.get("positive_fraction") is not None
                and float(return_shape["positive_fraction"]) >= getattr(self, "minimum_positive_fraction", 0.35)
            ),
            "profit_factor": float(return_shape.get("profit_factor") or 0.0) >= getattr(self, "minimum_profit_factor", 1.20),
            "payoff_ratio": float(return_shape.get("payoff_ratio") or 0.0) >= getattr(self, "minimum_payoff_ratio", 1.10),
            "bayesian_probability": float(posterior.get("probability_mean_positive") or 0.0) >= 0.95,
            "bayesian_p05": float(posterior.get("mean_p05_bps") or -1e9) > 0.0,
            "stochastic": bool(stochastic.get("passed", False)),
            "improves_baseline": improvement is not None and improvement > 0.0,
            "net_edge_calibration": bool(calibration.get("qualified", False)),
            "swing_geometry": bool(geometry.get("qualified", False)),
        }
        qualified = all(checks.values())
        policy = {
            "schema_version": self.SCHEMA,
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "QUALIFIED" if qualified else "COLLECTING",
            "qualified": qualified,
            "observations": len(rows),
            "train_observations": len(train),
            "holdout_observations": len(holdout),
            "holdout_selected": len(holdout_selected),
            "weights": weights,
            "base_priors": DEFAULT_PRIORS,
            "component_health": health,
            "threshold": threshold,
            "training_selection": training_selection,
            "holdout": {
                "mean_normal_net_bps": selected_mean,
                "mean_stressed_net_bps": stressed_mean,
                "positive_fraction": return_shape.get("positive_fraction"),
                "profit_factor": return_shape.get("profit_factor"),
                "payoff_ratio": return_shape.get("payoff_ratio"),
                "baseline_mean_normal_net_bps": baseline_mean,
                "mean_improvement_bps": improvement,
            },
            "bayesian": posterior,
            "stochastic_validation": stochastic,
            "strategy_lab": {
                "status": strategy.get("status"),
                "champion": champion.get("name") if champion else None,
            },
            "net_edge_calibration": calibration_summary,
            "swing_geometry": geometry_summary,
            "horizon_hours": int(getattr(self, "horizon_hours", 4)),
            "checks": checks,
            "shadow_influence": qualified,
            "live_decision_influence": False,
            "automatic_live_promotion": False,
            "normal_round_trip_cost_bps_fallback": fallback_normal,
            "stressed_round_trip_cost_bps_fallback": fallback_stressed,
            "reason_codes": [name.upper() + "_FAILED" for name, passed in checks.items() if not passed],
        }
        self._persist(policy)
        return policy

    def _persist(self, policy: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_path.write_text(json.dumps(policy, indent=2, sort_keys=True, default=str), encoding="utf-8")
        history_path = getattr(self, "history_path", self.root / "history.jsonl")
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(policy, sort_keys=True, default=str) + "\n")
        self._policy = dict(policy)
        self._last_refresh = time.time()

    def evaluate_market(self, context: Mapping[str, Any], *, policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
        active = dict(policy or self._policy or self._read_latest())
        weights = dict(active.get("weights") or DEFAULT_PRIORS)
        components = extract_components(context)
        score = self._score(components, weights)
        threshold = float(active.get("threshold", 0.70))
        qualified = bool(active.get("shadow_influence", False))
        strategy_lab = getattr(self, "strategy_lab", None)
        strategy_gate = (
            strategy_lab.evaluate_context(context)
            if strategy_lab is not None
            else {"status": "NO_CHAMPION", "passes": None, "champion": None}
        )
        strategy_pass = strategy_gate.get("passes")
        passes_strategy = strategy_pass is not False
        calibrator = getattr(self, "calibrator", None)
        economics_gate = (
            calibrator.evaluate_context(context)
            if calibrator is not None
            else {"status": "COLLECTING", "passes": False}
        )
        passes_economics = economics_gate.get("passes") is True
        geometry_engine = getattr(self, "swing_geometry", None)
        geometry_gate = (
            geometry_engine.evaluate_context(context)
            if geometry_engine is not None
            else {"status": "COLLECTING", "passes": False}
        )
        passes_geometry = geometry_gate.get("passes") is True
        signal_score = (
            score
            if (
                self.mode != "live"
                and qualified
                and score >= threshold
                and passes_strategy
                and passes_economics
                and passes_geometry
            )
            else None
        )
        used_total = sum(
            float(weights.get(name, 0.0))
            for name, value in components.items()
            if value is not None and float(weights.get(name, 0.0)) > 0
        )
        contributions = {
            name: (
                float(value) * float(weights.get(name, 0.0)) / used_total
                if value is not None and used_total > 0 and float(weights.get(name, 0.0)) > 0
                else None
            )
            for name, value in components.items()
        }
        return {
            "schema_version": "research_edge_market_decision_v4",
            "research_score": score,
            "threshold": threshold,
            "components": components,
            "weights": weights,
            "contributions": contributions,
            "passes_research_threshold": score >= threshold,
            "strategy_champion_gate": strategy_gate,
            "net_edge_gate": economics_gate,
            "swing_geometry_gate": geometry_gate,
            "strategy_hint": strategy_gate.get("champion") if strategy_pass else None,
            "signal_score": signal_score,
            "shadow_influence": signal_score is not None,
            "live_decision_influence": False,
            "policy_status": active.get("status", "COLLECTING"),
            "authority": "ADVISORY_ONLY",
        }
