from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from crypto_ai_swing.agents.component_features import (
    extract_components,
    extract_descriptors,
)
from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.research.performance_attribution import (
    PerformanceAttributionEngine,
    bayesian_mean_posterior,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _f(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _seed(label: str) -> int:
    return int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)


def _summary(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"observations": 0}
    winners = arr[arr > 0.0]
    losers = arr[arr <= 0.0]
    gross_profit = float(winners.sum()) if len(winners) else 0.0
    gross_loss = abs(float(losers.sum())) if len(losers) else 0.0
    mean_winner = float(winners.mean()) if len(winners) else None
    mean_loser = float(losers.mean()) if len(losers) else None
    return {
        "observations": int(len(arr)),
        "mean_bps": float(arr.mean()),
        "median_bps": float(np.median(arr)),
        "positive_fraction": float(np.mean(arr > 0.0)),
        "q05_bps": float(np.quantile(arr, 0.05)),
        "q95_bps": float(np.quantile(arr, 0.95)),
        "std_bps": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "mean_winner_bps": mean_winner,
        "mean_loser_bps": mean_loser,
        "payoff_ratio": (
            mean_winner / abs(mean_loser)
            if mean_winner is not None and mean_loser not in (None, 0.0)
            else None
        ),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else None,
    }


def _bucket(value: float | None, *, mid: float, high: float) -> str:
    if value is None:
        return "MISSING"
    if value >= high:
        return "HIGH"
    if value >= mid:
        return "MID"
    return "LOW"


def segment_keys_from_context(context: Mapping[str, Any]) -> list[str]:
    components = extract_components(context)
    desc = extract_descriptors(context)
    state = str(desc.get("breakout_state") or "UNKNOWN")
    family = str(desc.get("strategy_family") or "UNKNOWN")
    mtf_bucket = _bucket(components.get("mtf"), mid=0.65, high=0.75)
    tech_bucket = _bucket(components.get("technical"), mid=0.55, high=0.70)
    spread = _f(desc.get("spread_bps"), None)
    if spread is None:
        spread_bucket = "MISSING"
    elif spread <= 5.0:
        spread_bucket = "LE_5"
    elif spread <= 10.0:
        spread_bucket = "LE_10"
    else:
        spread_bucket = "GT_10"
    overextended = "YES" if bool(desc.get("overextended", False)) else "NO"
    return [
        f"STATE_MTF::{state}::{mtf_bucket}",
        f"STATE::{state}",
        f"FAMILY::{family}",
        f"MTF::{mtf_bucket}",
        f"TECH::{tech_bucket}",
        f"SPREAD::{spread_bucket}",
        f"OVEREXTENDED::{overextended}",
        "GLOBAL",
    ]


def segment_keys(row: Mapping[str, Any]) -> list[str]:
    context = row.get("context") if isinstance(row, Mapping) else None
    if isinstance(context, Mapping):
        return segment_keys_from_context(context)
    return segment_keys_from_context(row)


def exit_efficiency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "NO_DATA", "observations": 0}
    cost_cover = []
    cost_plus_25 = []
    negative_despite_cover = []
    giveback = []
    capture_needed = []
    for row in rows:
        mfe = _f(row.get("mfe_bps"), 0.0) or 0.0
        gross = _f(row.get("gross_return_bps"), 0.0) or 0.0
        cost = _f(row.get("normal_cost_bps"), 0.0) or 0.0
        net = _f(row.get("normal_net_bps"), 0.0) or 0.0
        cost_cover.append(mfe > cost)
        cost_plus_25.append(mfe > cost + 25.0)
        negative_despite_cover.append(net <= 0.0 and mfe > cost)
        giveback.append(mfe - gross)
        if mfe > 0:
            capture_needed.append(cost / mfe)
    return {
        "status": "READY",
        "observations": len(rows),
        "mfe_covers_normal_cost_fraction": float(np.mean(cost_cover)),
        "mfe_covers_cost_plus_25bps_fraction": float(np.mean(cost_plus_25)),
        "negative_realized_despite_mfe_cost_cover_fraction": float(
            np.mean(negative_despite_cover)
        ),
        "mean_mfe_to_realized_giveback_bps": float(np.mean(giveback)),
        "median_mfe_to_realized_giveback_bps": float(np.median(giveback)),
        "median_required_mfe_capture_fraction_for_breakeven": (
            float(np.median(capture_needed)) if capture_needed else None
        ),
        "interpretation": (
            "Diagnostic only. A high MFE-cost-cover fraction with negative "
            "realized net returns indicates that entry/exit timing and "
            "giveback deserve separate prospective testing; it is not proof "
            "that a realizable exit rule exists."
        ),
    }


class NetEdgeCalibrator:
    """Prospective, cost-aware empirical-Bayes edge calibration.

    Current-context estimates may use all matured evidence available at the
    evaluation time. Qualification is stricter: each selected historical row is
    gated using only observations that existed before that row, preventing
    look-ahead from turning an attractive segment into fake OOS evidence.

    The calibrator is research/paper/shadow only and cannot grant live authority.
    """

    SCHEMA = "crypto_ai_swing_net_edge_calibration_v2"

    def __init__(self, settings, *, mode: str = "paper") -> None:
        self.settings = settings
        self.mode = str(mode).lower()
        cfg = dict((getattr(settings, "autonomy", {}) or {}).get(
            "net_edge_calibration", {}
        ) or {})
        self.horizon_hours = int(cfg.get("horizon_hours", 4))
        self.minimum_history = int(cfg.get("minimum_history_observations", 40))
        self.minimum_segment = int(cfg.get("minimum_segment_observations", 8))
        self.prior_strength = float(cfg.get("prior_strength_observations", 24.0))
        self.minimum_expected_net_bps = float(cfg.get("minimum_expected_net_bps", 10.0))
        self.minimum_stressed_net_bps = float(cfg.get("minimum_stressed_net_bps", 0.0))
        self.minimum_qualification_selected = int(cfg.get("minimum_qualification_selected", 30))
        self.minimum_qualification_total = int(cfg.get("minimum_qualification_total", 100))
        self.minimum_positive_fraction = float(cfg.get("minimum_positive_fraction", 0.35))
        self.minimum_profit_factor = float(cfg.get("minimum_profit_factor", 1.20))
        self.minimum_payoff_ratio = float(cfg.get("minimum_payoff_ratio", 1.10))
        self.maximum_cost_fraction = float(cfg.get("maximum_cost_fraction_of_mean_winner", 0.35))
        self.minimum_probability_positive = float(cfg.get("minimum_probability_positive", 0.95))
        self.bootstrap_draws = int(cfg.get("bootstrap_draws", 3000))
        self.refresh_seconds = float(cfg.get("refresh_seconds", 900))
        self.bridge = CryptoLibraryBridge(settings.crypto_repo_root)
        self.attribution = PerformanceAttributionEngine(settings, mode=mode)
        # v0.26 decouples short-horizon attribution from swing qualification.
        # The calibrator owns its configured economic horizon and never rewrites
        # the underlying forward evidence.
        self.attribution.horizon_hours = self.horizon_hours
        self.root = Path(settings.project_root) / "output/crypto_ai_swing/research/net_edge"
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.root / "latest.json"
        self.history_path = self.root / "history.jsonl"
        self._last_refresh = 0.0
        self._latest: dict[str, Any] = {}
        self._rows_cache: list[dict[str, Any]] = []
        self._rows_cache_at = 0.0

    @staticmethod
    def _rows_for_key(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        return [row for row in rows if key in segment_keys(row)]

    def _shrink_estimate(
        self,
        selected: list[dict[str, Any]],
        global_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not selected or not global_rows:
            return {"status": "INSUFFICIENT", "observations": len(selected)}
        n = len(selected)
        k = max(0.0, self.prior_strength)
        weight = n / (n + k) if n + k > 0 else 1.0
        seg_normal = np.asarray([r["normal_net_bps"] for r in selected], dtype=float)
        seg_stressed = np.asarray([r["stressed_net_bps"] for r in selected], dtype=float)
        glob_normal = np.asarray([r["normal_net_bps"] for r in global_rows], dtype=float)
        glob_stressed = np.asarray([r["stressed_net_bps"] for r in global_rows], dtype=float)
        normal_mean = weight * float(seg_normal.mean()) + (1.0 - weight) * float(glob_normal.mean())
        stressed_mean = weight * float(seg_stressed.mean()) + (1.0 - weight) * float(glob_stressed.mean())
        return {
            "status": "READY",
            "observations": n,
            "prior_observations": k,
            "segment_weight": weight,
            "raw_normal_mean_bps": float(seg_normal.mean()),
            "raw_stressed_mean_bps": float(seg_stressed.mean()),
            "global_normal_mean_bps": float(glob_normal.mean()),
            "global_stressed_mean_bps": float(glob_stressed.mean()),
            "shrunk_normal_mean_bps": normal_mean,
            "shrunk_stressed_mean_bps": stressed_mean,
            "positive_fraction": float(np.mean(seg_normal > 0.0)),
        }

    def _posterior_shrunk(
        self,
        selected: list[dict[str, Any]],
        global_rows: list[dict[str, Any]],
        *,
        key: str,
    ) -> dict[str, Any]:
        if len(selected) < self.minimum_segment or not global_rows:
            return {"status": "INSUFFICIENT", "observations": len(selected)}
        n = len(selected)
        k = max(0.0, self.prior_strength)
        weight = n / (n + k) if n + k > 0 else 1.0
        seg = np.asarray([r["normal_net_bps"] for r in selected], dtype=float)
        glob = np.asarray([r["normal_net_bps"] for r in global_rows], dtype=float)
        draws = max(1000, self.bootstrap_draws)
        rng = np.random.default_rng(_seed(f"net-edge:{key}:{n}:{len(global_rows)}"))
        seg_idx = rng.integers(0, len(seg), size=(draws, len(seg)))
        glob_size = min(len(glob), max(8, len(seg)))
        glob_idx = rng.integers(0, len(glob), size=(draws, glob_size))
        mixed = weight * seg[seg_idx].mean(axis=1) + (1.0 - weight) * glob[glob_idx].mean(axis=1)
        return {
            "status": "READY",
            "observations": n,
            "probability_mean_positive": float(np.mean(mixed > 0.0)),
            "mean_p05_bps": float(np.quantile(mixed, 0.05)),
            "mean_p50_bps": float(np.quantile(mixed, 0.50)),
            "mean_p95_bps": float(np.quantile(mixed, 0.95)),
        }

    def _select_segment(
        self,
        rows: list[dict[str, Any]],
        context: Mapping[str, Any],
        *,
        posterior: bool,
    ) -> dict[str, Any]:
        if len(rows) < self.minimum_history:
            return {
                "status": "COLLECTING",
                "history_observations": len(rows),
                "required_history": self.minimum_history,
                "passes": False,
            }
        for key in segment_keys_from_context(context):
            selected = self._rows_for_key(rows, key)
            if key != "GLOBAL" and len(selected) < self.minimum_segment:
                continue
            if key == "GLOBAL":
                selected = rows
            estimate = self._shrink_estimate(selected, rows)
            if estimate.get("status") != "READY":
                continue
            post = (
                self._posterior_shrunk(selected, rows, key=key)
                if posterior
                else {"status": "NOT_COMPUTED"}
            )
            normal = float(estimate.get("shrunk_normal_mean_bps") or -1e9)
            stressed = float(estimate.get("shrunk_stressed_mean_bps") or -1e9)
            passes = (
                normal >= self.minimum_expected_net_bps
                and stressed >= self.minimum_stressed_net_bps
            )
            if posterior and post.get("status") == "READY":
                passes = passes and (
                    float(post.get("probability_mean_positive") or 0.0)
                    >= 0.90
                )
            return {
                "status": "READY",
                "segment": key,
                **estimate,
                "posterior": post,
                "minimum_expected_net_bps": self.minimum_expected_net_bps,
                "minimum_stressed_net_bps": self.minimum_stressed_net_bps,
                "passes": bool(passes),
            }
        return {
            "status": "COLLECTING",
            "history_observations": len(rows),
            "passes": False,
        }

    def _sequential_validation(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        selected: list[dict[str, Any]] = []
        audit: list[dict[str, Any]] = []
        start = max(self.minimum_history, self.minimum_segment)
        stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {"n": 0.0, "normal": 0.0, "stressed": 0.0}
        )

        def add(row: dict[str, Any]) -> None:
            normal = float(row["normal_net_bps"])
            stressed = float(row["stressed_net_bps"])
            for key in segment_keys(row):
                bucket = stats[key]
                bucket["n"] += 1.0
                bucket["normal"] += normal
                bucket["stressed"] += stressed

        def gate_for(context: Mapping[str, Any]) -> dict[str, Any]:
            global_stats = stats.get("GLOBAL")
            if not global_stats or global_stats["n"] < self.minimum_history:
                return {"status": "COLLECTING", "passes": False}
            global_n = global_stats["n"]
            global_normal = global_stats["normal"] / global_n
            global_stressed = global_stats["stressed"] / global_n
            for key in segment_keys_from_context(context):
                bucket = stats.get(key)
                if not bucket:
                    continue
                n = int(bucket["n"])
                if key != "GLOBAL" and n < self.minimum_segment:
                    continue
                weight = n / (n + self.prior_strength)
                seg_normal = bucket["normal"] / max(1, n)
                seg_stressed = bucket["stressed"] / max(1, n)
                normal = weight * seg_normal + (1.0 - weight) * global_normal
                stressed = weight * seg_stressed + (1.0 - weight) * global_stressed
                return {
                    "status": "READY",
                    "segment": key,
                    "observations": n,
                    "shrunk_normal_mean_bps": normal,
                    "shrunk_stressed_mean_bps": stressed,
                    "passes": (
                        normal >= self.minimum_expected_net_bps
                        and stressed >= self.minimum_stressed_net_bps
                    ),
                }
            return {"status": "COLLECTING", "passes": False}

        for idx, row in enumerate(rows):
            if idx >= start:
                gate = gate_for(row["context"])
                if gate.get("passes"):
                    selected.append(row)
                    audit.append({
                        "observed_at": row.get("observed_at"),
                        "market": row.get("market"),
                        "segment": gate.get("segment"),
                        "history_observations": idx,
                        "estimated_normal_net_bps": gate.get("shrunk_normal_mean_bps"),
                        "estimated_stressed_net_bps": gate.get("shrunk_stressed_mean_bps"),
                        "realized_normal_net_bps": row.get("normal_net_bps"),
                    })
            add(row)

        normal = np.asarray([r["normal_net_bps"] for r in selected], dtype=float)
        stressed = np.asarray([r["stressed_net_bps"] for r in selected], dtype=float)
        posterior = bayesian_mean_posterior(
            normal,
            draws=max(2000, self.bootstrap_draws),
            seed=2501,
        ) if len(normal) else {"status": "INSUFFICIENT", "observations": 0}
        stochastic = self._stochastic(normal, stressed)
        normal_summary = _summary(normal)
        gross_selected = np.asarray([r["gross_return_bps"] for r in selected], dtype=float)
        costs_selected = np.asarray([r["normal_cost_bps"] for r in selected], dtype=float)
        economic_winners = (
            gross_selected[gross_selected > costs_selected]
            if len(gross_selected) == len(costs_selected)
            else gross_selected[gross_selected > 0.0]
        )
        cost_fraction = (
            float(costs_selected.mean() / economic_winners.mean())
            if len(costs_selected)
            and len(economic_winners)
            and float(economic_winners.mean()) > 0.0
            else None
        )
        checks = {
            "minimum_total_observations": len(rows) >= self.minimum_qualification_total,
            "minimum_oos_selected": len(selected) >= self.minimum_qualification_selected,
            "positive_normal_mean": len(normal) > 0 and float(normal.mean()) > 0.0,
            "positive_stressed_mean": len(stressed) > 0 and float(stressed.mean()) > 0.0,
            "positive_fraction": len(normal) > 0 and float(np.mean(normal > 0.0)) >= getattr(self, "minimum_positive_fraction", 0.35),
            "profit_factor": float(normal_summary.get("profit_factor") or 0.0) >= getattr(self, "minimum_profit_factor", 1.20),
            "payoff_ratio": float(normal_summary.get("payoff_ratio") or 0.0) >= getattr(self, "minimum_payoff_ratio", 1.10),
            "cost_burden": cost_fraction is not None and cost_fraction <= getattr(self, "maximum_cost_fraction", 0.35),
            "bayesian_probability": float(posterior.get("probability_mean_positive") or 0.0) >= getattr(self, "minimum_probability_positive", 0.95),
            "bayesian_p05": float(posterior.get("mean_p05_bps") or -1e9) > 0.0,
            "stochastic": bool(stochastic.get("passed", False)),
        }
        return {
            "status": "QUALIFIED" if all(checks.values()) else "COLLECTING",
            "qualified": all(checks.values()),
            "total_observations": len(rows),
            "oos_selected": len(selected),
            "normal_net": normal_summary,
            "stressed_net": _summary(stressed),
            "cost_fraction_of_mean_winner": cost_fraction,
            "bayesian": posterior,
            "stochastic_validation": stochastic,
            "checks": checks,
            "recent_audit": audit[-25:],
            "lookahead_policy": "EACH_OOS_ROW_GATED_USING_ONLY_PRIOR_MATURED_ROWS",
        }

    def _stochastic(self, normal: np.ndarray, stressed: np.ndarray) -> dict[str, Any]:
        if len(normal) < self.minimum_qualification_selected:
            return {
                "status": "COLLECTING",
                "passed": False,
                "observation_count": int(len(normal)),
                "required": self.minimum_qualification_selected,
            }
        try:
            module = self.bridge.import_module("research.stochastic_validation")
            policy = module.StochasticValidationPolicy(
                simulations=10_000,
                expected_block_length=max(3, min(10, len(normal) // 8)),
                maximum_drawdown=0.15,
                maximum_drawdown_breach_probability=0.01,
                maximum_terminal_loss_probability=0.05,
                minimum_p05_total_return=0.0,
                dirichlet_blocks=8,
                dirichlet_concentrations=(0.5, 1.0, 5.0),
                minimum_observations=self.minimum_qualification_selected,
                confidence_level=0.95,
                seed=2502,
                batch_size=256,
            )
            return module.validate_strategy_return_paths(
                normal / 10_000.0,
                stressed / 10_000.0,
                policy=policy,
                seed_offset=0,
            )
        except Exception as exc:
            return {
                "status": "ERROR",
                "passed": False,
                "error": f"{type(exc).__name__}:{str(exc)[:300]}",
            }

    def _segment_table(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            for key in segment_keys(row):
                grouped[key].append(row)
        output = []
        for key, selected in grouped.items():
            if key != "GLOBAL" and len(selected) < self.minimum_segment:
                continue
            estimate = self._shrink_estimate(selected, rows)
            if estimate.get("status") != "READY":
                continue
            output.append({
                "segment": key,
                **estimate,
                "posterior": self._posterior_shrunk(
                    selected,
                    rows,
                    key=key,
                ),
                "gross": _summary([r["gross_return_bps"] for r in selected]),
                "normal_net": _summary([r["normal_net_bps"] for r in selected]),
                "stressed_net": _summary([r["stressed_net_bps"] for r in selected]),
                "mean_mfe_bps": float(np.mean([r["mfe_bps"] for r in selected])),
                "mean_mae_bps": float(np.mean([r["mae_bps"] for r in selected])),
            })
        output.sort(
            key=lambda row: (
                float(row.get("shrunk_normal_mean_bps") or -1e12),
                int(row.get("observations") or 0),
            ),
            reverse=True,
        )
        return output

    def _rows(self, *, force: bool = False) -> list[dict[str, Any]]:
        now = time.time()
        if (
            not force
            and self._rows_cache
            and now - self._rows_cache_at < self.refresh_seconds
        ):
            return list(self._rows_cache)
        rows = self.attribution.load_forward_rows()
        self._rows_cache = list(rows)
        self._rows_cache_at = now
        return rows

    def refresh(self, *, force: bool = False) -> dict[str, Any]:
        now = time.time()
        if not force and self._latest and now - self._last_refresh < self.refresh_seconds:
            return dict(self._latest)
        rows = self._rows(force=force)
        validation = self._sequential_validation(rows)
        payload = {
            "schema_version": self.SCHEMA,
            "generated_at": _now(),
            "status": validation.get("status", "COLLECTING"),
            "qualified": bool(validation.get("qualified", False)),
            "observations": len(rows),
            "minimum_history_observations": self.minimum_history,
            "minimum_segment_observations": self.minimum_segment,
            "prior_strength_observations": self.prior_strength,
            "horizon_hours": self.horizon_hours,
            "minimum_expected_net_bps": self.minimum_expected_net_bps,
            "swing_asymmetry_policy": {
                "minimum_positive_fraction": getattr(self, "minimum_positive_fraction", 0.35),
                "minimum_profit_factor": getattr(self, "minimum_profit_factor", 1.20),
                "minimum_payoff_ratio": getattr(self, "minimum_payoff_ratio", 1.10),
                "maximum_cost_fraction_of_mean_winner": getattr(self, "maximum_cost_fraction", 0.35),
            },
            "overall": {
                "gross": _summary([r["gross_return_bps"] for r in rows]),
                "normal_net": _summary([r["normal_net_bps"] for r in rows]),
                "stressed_net": _summary([r["stressed_net_bps"] for r in rows]),
                "mean_normal_cost_bps": (
                    float(np.mean([r["normal_cost_bps"] for r in rows])) if rows else None
                ),
                "mean_stressed_cost_bps": (
                    float(np.mean([r["stressed_cost_bps"] for r in rows])) if rows else None
                ),
            },
            "exit_efficiency": exit_efficiency(rows),
            "segments": self._segment_table(rows)[:40],
            "oos_validation": validation,
            "shadow_decision_influence": bool(validation.get("qualified", False)),
            "live_decision_influence": False,
            "automatic_live_promotion": False,
            "orders_submitted": 0,
            "artifact_path": str(self.latest_path),
        }
        self.latest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        self._latest = dict(payload)
        self._last_refresh = now
        return payload

    def _read_latest(self) -> dict[str, Any]:
        if self._latest:
            return dict(self._latest)
        try:
            value = json.loads(self.latest_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                self._latest = dict(value)
                return dict(value)
        except Exception:
            pass
        return {}

    def evaluate_context(
        self,
        context: Mapping[str, Any],
        *,
        policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        active = dict(policy or self._read_latest())
        models = {
            str(row.get("segment")): dict(row)
            for row in list(active.get("segments") or [])
            if isinstance(row, Mapping) and row.get("segment")
        }
        gate: dict[str, Any] = {
            "status": "COLLECTING",
            "passes": False,
        }
        for key in segment_keys_from_context(context):
            model = models.get(key)
            if not model:
                continue
            observations = int(model.get("observations") or 0)
            if key != "GLOBAL" and observations < self.minimum_segment:
                continue
            posterior = dict(model.get("posterior") or {})
            normal = float(model.get("shrunk_normal_mean_bps") or -1e9)
            stressed = float(model.get("shrunk_stressed_mean_bps") or -1e9)
            preliminary = (
                normal >= self.minimum_expected_net_bps
                and stressed >= self.minimum_stressed_net_bps
                and float(posterior.get("probability_mean_positive") or 0.0)
                >= 0.90
            )
            gate = {
                "status": "READY",
                "segment": key,
                "observations": observations,
                "shrunk_normal_mean_bps": normal,
                "shrunk_stressed_mean_bps": stressed,
                "posterior": posterior,
                "minimum_expected_net_bps": self.minimum_expected_net_bps,
                "minimum_stressed_net_bps": self.minimum_stressed_net_bps,
                "preliminary_pass": bool(preliminary),
                "passes": False,
            }
            break
        qualified = bool(active.get("qualified", False))
        passes = (
            bool(gate.get("preliminary_pass", False))
            and qualified
            and self.mode != "live"
        )
        gate["passes"] = passes
        return {
            "schema_version": "crypto_ai_swing_net_edge_market_gate_v1",
            **gate,
            "calibration_qualified": qualified,
            "shadow_decision_influence": passes,
            "live_decision_influence": False,
            "authority": "ADVISORY_ONLY",
        }

    @staticmethod
    def policy_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
        overall = dict(payload.get("overall") or {})
        validation = dict(payload.get("oos_validation") or {})
        normal = dict(validation.get("normal_net") or {})
        return {
            "status": payload.get("status", "COLLECTING"),
            "qualified": bool(payload.get("qualified", False)),
            "horizon_hours": payload.get("horizon_hours"),
            "observations": payload.get("observations"),
            "mean_normal_cost_bps": overall.get("mean_normal_cost_bps"),
            "mean_normal_net_bps": (overall.get("normal_net") or {}).get("mean_bps"),
            "oos_selected": validation.get("oos_selected"),
            "oos_profit_factor": normal.get("profit_factor"),
            "oos_payoff_ratio": normal.get("payoff_ratio"),
            "cost_fraction_of_mean_winner": validation.get("cost_fraction_of_mean_winner"),
            "reason_codes": [
                name.upper() + "_FAILED"
                for name, passed in dict(validation.get("checks") or {}).items()
                if not passed
            ],
            "artifact_path": str(payload.get("artifact_path") or ""),
            "live_decision_influence": False,
        }
