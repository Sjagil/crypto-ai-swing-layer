from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.covariance import LedoitWolf

from crypto_ai_swing.agents.component_features import (
    extract_components,
    extract_descriptors,
)
from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.research.performance_attribution import (
    PerformanceAttributionEngine,
    bayesian_mean_posterior,
)


TIMEFRAMES = ("15m", "1h", "2h", "4h", "1d", "1w")
BASE_FEATURES = (
    "tf_15m",
    "tf_1h",
    "tf_2h",
    "tf_4h",
    "tf_1d",
    "tf_1w",
    "technical",
    "mtf",
    "risk",
    "execution",
    "cmc",
    "nlp",
    "orderflow",
    "mtf_alignment",
    "mtf_macro",
    "mtf_trend",
    "mtf_setup",
    "mtf_trigger",
    "mtf_execution",
    "h1_rsi_centered",
    "m15_rsi_centered",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _f(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _dt(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except Exception:
        return None


def _seed(label: str) -> int:
    return int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)


def _summary(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float).reshape(-1)
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
        "std_bps": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "positive_fraction": float(np.mean(arr > 0.0)),
        "q05_bps": float(np.quantile(arr, 0.05)),
        "q25_bps": float(np.quantile(arr, 0.25)),
        "q75_bps": float(np.quantile(arr, 0.75)),
        "q95_bps": float(np.quantile(arr, 0.95)),
        "mean_winner_bps": mean_winner,
        "mean_loser_bps": mean_loser,
        "payoff_ratio": (
            mean_winner / abs(mean_loser)
            if mean_winner is not None and mean_loser not in (None, 0.0)
            else None
        ),
        "profit_factor": (
            gross_profit / gross_loss if gross_loss > 0.0 else None
        ),
    }


def return_quality(
    normal_bps: Sequence[float] | np.ndarray,
    gross_bps: Sequence[float] | np.ndarray | None = None,
    costs_bps: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    normal = _summary(normal_bps)
    gross = _summary(gross_bps if gross_bps is not None else normal_bps)
    costs = np.asarray(costs_bps if costs_bps is not None else [], dtype=float)
    costs = costs[np.isfinite(costs)]
    mean_cost = float(costs.mean()) if len(costs) else None
    gross_arr = np.asarray(gross_bps if gross_bps is not None else normal_bps, dtype=float).reshape(-1)
    gross_arr = gross_arr[np.isfinite(gross_arr)]
    if mean_cost is not None and len(gross_arr):
        economic_winners = gross_arr[gross_arr > mean_cost]
    else:
        economic_winners = gross_arr[gross_arr > 0.0]
    winner = (
        float(economic_winners.mean())
        if len(economic_winners)
        else gross.get("mean_winner_bps")
    )
    burden = (
        mean_cost / float(winner)
        if mean_cost is not None and winner is not None and float(winner) > 0.0
        else None
    )
    gross_mean = gross.get("mean_bps")
    return {
        "normal_net": normal,
        "gross": gross,
        "mean_cost_bps": mean_cost,
        "cost_fraction_of_mean_winner": burden,
        "edge_cost_multiple": (
            float(gross_mean) / mean_cost
            if mean_cost and gross_mean is not None
            else None
        ),
    }


def asymmetric_swing_quality_pass(
    summary: Mapping[str, Any],
    *,
    minimum_positive_fraction: float = 0.35,
    minimum_profit_factor: float = 1.20,
    minimum_payoff_ratio: float = 1.10,
) -> bool:
    return bool(
        int(summary.get("observations") or 0) > 0
        and float(summary.get("mean_bps") or -1e12) > 0.0
        and float(summary.get("positive_fraction") or 0.0)
        >= minimum_positive_fraction
        and float(summary.get("profit_factor") or 0.0)
        >= minimum_profit_factor
        and float(summary.get("payoff_ratio") or 0.0)
        >= minimum_payoff_ratio
    )


def _state_score(row: Mapping[str, Any]) -> float | None:
    for key in ("score", "technical_score", "trend_score"):
        value = _f(row.get(key), None)
        if value is not None:
            return float(np.clip(value, -1.0, 1.0))
    return None


def _candidate_states(context: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates = (
        context.get("timeframe_pipeline"),
        context.get("timeframe_decision"),
        context.get("mtf_pipeline"),
        context.get("pipeline"),
    )
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            states = candidate.get("states")
            if isinstance(states, Mapping) and any(tf in states for tf in TIMEFRAMES):
                return states

    def walk(value: Any, depth: int = 0) -> Mapping[str, Any] | None:
        if depth > 4 or not isinstance(value, Mapping):
            return None
        states = value.get("states")
        if isinstance(states, Mapping) and any(tf in states for tf in TIMEFRAMES):
            return states
        for nested in value.values():
            found = walk(nested, depth + 1)
            if found is not None:
                return found
        return None

    return walk(context)


def feature_vector(context: Mapping[str, Any]) -> dict[str, float | None]:
    components = extract_components(context)
    desc = extract_descriptors(context)
    states = _candidate_states(context) or {}
    result: dict[str, float | None] = {}
    for tf in TIMEFRAMES:
        state = states.get(tf)
        result[f"tf_{tf}"] = _state_score(state) if isinstance(state, Mapping) else None
    for name in ("technical", "mtf", "risk", "execution", "cmc", "nlp", "orderflow"):
        result[name] = _f(components.get(name), None)
    for source, target in (
        ("mtf_alignment", "mtf_alignment"),
        ("mtf_macro", "mtf_macro"),
        ("mtf_trend", "mtf_trend"),
        ("mtf_setup", "mtf_setup"),
        ("mtf_trigger", "mtf_trigger"),
        ("mtf_execution", "mtf_execution"),
    ):
        result[target] = _f(desc.get(source), None)
    h1 = _f(desc.get("h1_rsi"), None)
    m15 = _f(desc.get("m15_rsi"), None)
    result["h1_rsi_centered"] = (h1 - 50.0) / 50.0 if h1 is not None else None
    result["m15_rsi_centered"] = (m15 - 50.0) / 50.0 if m15 is not None else None
    return result


def _robust_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.nanmedian(matrix, axis=0)
    mad = np.nanmedian(np.abs(matrix - center), axis=0) * 1.4826
    std = np.nanstd(matrix, axis=0, ddof=1)
    scale = np.where(np.isfinite(mad) & (mad > 1e-8), mad, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    filled = np.where(np.isfinite(matrix), matrix, center)
    return (filled - center) / scale, center, scale


def fit_shrinkage_geometry(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    minimum_feature_coverage: float = 0.60,
    ridge: float = 0.05,
    bootstrap_draws: int = 200,
    seed: int = 2601,
) -> dict[str, Any]:
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if x.ndim != 2 or x.shape[0] != len(y):
        raise ValueError("matrix/labels shape mismatch")
    names = list(feature_names or [f"x{i}" for i in range(x.shape[1])])
    coverage = np.mean(np.isfinite(x), axis=0)
    keep = np.where(coverage >= minimum_feature_coverage)[0]
    if len(keep) < 2 or x.shape[0] < max(12, len(keep) + 4):
        return {
            "status": "COLLECTING",
            "observations": int(x.shape[0]),
            "features_available": int(len(keep)),
            "required_features": 2,
        }
    x = x[:, keep]
    used = [names[i] for i in keep]
    z, center, scale = _robust_scale(x)
    model = LedoitWolf(store_precision=False, assume_centered=False).fit(z)
    covariance = np.asarray(model.covariance_, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    floor = max(1e-8, float(eigenvalues[0]) * 1e-6)
    eigenvalues = np.maximum(eigenvalues, floor)
    covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
    precision = np.linalg.solve(covariance, np.eye(covariance.shape[0]))
    p = eigenvalues / eigenvalues.sum()
    effective_rank = float(np.exp(-np.sum(p * np.log(np.clip(p, 1e-15, None)))))
    cumulative = np.cumsum(p)
    rank90 = int(np.searchsorted(cumulative, 0.90) + 1)
    whitening = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T

    positive = z[y > 0]
    negative = z[y <= 0]
    fisher = np.zeros(z.shape[1], dtype=float)
    if len(positive) >= 3 and len(negative) >= 3:
        delta = positive.mean(axis=0) - negative.mean(axis=0)
        fisher = np.linalg.solve(
            covariance + ridge * np.eye(covariance.shape[0]),
            delta,
        )
        norm = float(np.linalg.norm(fisher))
        if norm > 0.0:
            fisher /= norm
    fisher_scores = z @ fisher
    pos_mean = float(np.mean(fisher_scores[y > 0])) if np.any(y > 0) else 0.0
    neg_mean = float(np.mean(fisher_scores[y <= 0])) if np.any(y <= 0) else 0.0
    if pos_mean < neg_mean:
        fisher *= -1.0
        fisher_scores *= -1.0
        pos_mean, neg_mean = -pos_mean, -neg_mean
    score_center = float(np.median(fisher_scores))
    score_scale = float(np.std(fisher_scores, ddof=1)) if len(fisher_scores) > 1 else 1.0
    score_scale = max(score_scale, 1e-8)

    distances = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", z, precision, z)))
    ood_threshold = float(np.quantile(distances, 0.975))

    stability: list[float] = []
    if np.linalg.norm(fisher) > 0 and bootstrap_draws > 0 and len(z) >= 20:
        rng = np.random.default_rng(seed)
        draws = min(int(bootstrap_draws), 500)
        for _ in range(draws):
            idx = rng.integers(0, len(z), size=len(z))
            zb = z[idx]
            yb = y[idx]
            if np.sum(yb > 0) < 3 or np.sum(yb <= 0) < 3:
                continue
            try:
                cov_b = LedoitWolf().fit(zb).covariance_
                delta_b = zb[yb > 0].mean(axis=0) - zb[yb <= 0].mean(axis=0)
                direction = np.linalg.solve(
                    cov_b + ridge * np.eye(cov_b.shape[0]), delta_b
                )
                norm = float(np.linalg.norm(direction))
                if norm <= 0:
                    continue
                direction /= norm
                cosine = float(np.dot(direction, fisher))
                if cosine < 0:
                    cosine = -cosine
                stability.append(cosine)
            except Exception:
                continue

    factors = []
    for idx in range(min(3, len(eigenvalues))):
        load = eigenvectors[:, idx]
        top = np.argsort(np.abs(load))[::-1][: min(5, len(load))]
        factors.append({
            "factor": idx + 1,
            "variance_fraction": float(p[idx]),
            "top_loadings": [
                {"feature": used[j], "loading": float(load[j])}
                for j in top
            ],
        })

    return {
        "status": "READY",
        "observations": int(len(z)),
        "feature_names": used,
        "feature_coverage": {names[i]: float(coverage[i]) for i in keep},
        "center": center.tolist(),
        "scale": scale.tolist(),
        "covariance": covariance.tolist(),
        "precision": precision.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "effective_rank": effective_rank,
        "rank_90pct_variance": rank90,
        "condition_number": float(eigenvalues[0] / eigenvalues[-1]),
        "shrinkage": float(model.shrinkage_),
        "whitening_matrix": whitening.tolist(),
        "fisher_direction": fisher.tolist(),
        "fisher_positive_mean": pos_mean,
        "fisher_negative_mean": neg_mean,
        "fisher_separation": pos_mean - neg_mean,
        "fisher_score_center": score_center,
        "fisher_score_scale": score_scale,
        "fisher_bootstrap_cosine_median": (
            float(np.median(stability)) if stability else None
        ),
        "fisher_bootstrap_cosine_p05": (
            float(np.quantile(stability, 0.05)) if stability else None
        ),
        "ood_threshold_mahalanobis": ood_threshold,
        "principal_factors": factors,
        "linear_algebra_contract": {
            "covariance": "LEDOIT_WOLF_SHRINKAGE",
            "eigendecomposition": "SYMMETRIC_EIGH_WITH_EIGENVALUE_FLOOR",
            "classifier_direction": "FISHER_SOLVE_SIGMA_PLUS_RIDGE",
            "matrix_inverse_policy": "NO_EXPLICIT_INVERSE_FOR_FISHER",
            "distance": "MAHALANOBIS",
            "whitening": "Q_DIAG_LAMBDA_MINUS_HALF_QT",
        },
    }


def score_geometry(model: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    if str(model.get("status")) != "READY":
        return {"status": "COLLECTING", "passes": False}
    names = list(model.get("feature_names") or [])
    center = np.asarray(model.get("center"), dtype=float)
    scale = np.asarray(model.get("scale"), dtype=float)
    precision = np.asarray(model.get("precision"), dtype=float)
    fisher = np.asarray(model.get("fisher_direction"), dtype=float)
    raw = np.asarray([_f(values.get(name), np.nan) for name in names], dtype=float)
    raw = np.where(np.isfinite(raw), raw, center)
    z = (raw - center) / scale
    fisher_score = float(z @ fisher)
    score_center = float(model.get("fisher_score_center") or 0.0)
    score_scale = max(float(model.get("fisher_score_scale") or 1.0), 1e-8)
    standardized = (fisher_score - score_center) / score_scale
    probability_like = float(1.0 / (1.0 + np.exp(-np.clip(standardized, -20, 20))))
    distance = float(np.sqrt(max(0.0, z @ precision @ z)))
    threshold = max(float(model.get("ood_threshold_mahalanobis") or 0.0), 1e-8)
    ood = distance > threshold
    confidence = float(np.exp(-max(0.0, distance / threshold - 1.0)))
    return {
        "status": "READY",
        "fisher_score": fisher_score,
        "fisher_probability_like": probability_like,
        "mahalanobis_distance": distance,
        "ood_threshold": threshold,
        "out_of_distribution": bool(ood),
        "geometry_confidence": confidence,
        "passes": bool(not ood and probability_like >= 0.55),
    }


class SwingGeometryEngine:
    """Multi-horizon swing research with shrinkage linear algebra.

    Full transaction costs are always deducted. Cost tolerance is expressed as
    a fraction of the swing's observed winner magnitude, never by pretending
    fees/slippage did not occur. Historical horizon selection is delayed-label
    safe: only outcomes matured before a decision may select that decision's
    horizon. This module is research/paper/shadow only.
    """

    SCHEMA = "crypto_ai_swing_multi_horizon_geometry_v1"

    def __init__(self, settings, *, mode: str = "paper") -> None:
        self.settings = settings
        self.mode = str(mode).lower()
        cfg = dict((getattr(settings, "autonomy", {}) or {}).get(
            "swing_geometry", {}
        ) or {})
        self.horizons = tuple(int(x) for x in cfg.get("horizons_hours", (4, 24, 72, 168)))
        self.swing_horizons = tuple(int(x) for x in cfg.get("swing_horizons_hours", (24, 72, 168)))
        self.anchor_horizon = int(cfg.get("anchor_horizon_hours", 24))
        self.minimum_geometry_fit = int(cfg.get("minimum_geometry_fit_observations", 30))
        self.minimum_geometry_qualification = int(cfg.get("minimum_geometry_qualification_observations", 60))
        self.minimum_horizon_history = int(cfg.get("minimum_horizon_history_observations", 20))
        self.minimum_total = int(cfg.get("minimum_qualification_total", 120))
        self.minimum_oos = int(cfg.get("minimum_oos_selected", 30))
        self.minimum_positive_fraction = float(cfg.get("minimum_positive_fraction", 0.35))
        self.minimum_profit_factor = float(cfg.get("minimum_profit_factor", 1.20))
        self.minimum_payoff_ratio = float(cfg.get("minimum_payoff_ratio", 1.10))
        self.maximum_cost_fraction = float(cfg.get("maximum_cost_fraction_of_mean_winner", 0.35))
        self.minimum_probability_positive = float(cfg.get("minimum_probability_positive", 0.95))
        self.minimum_fisher_stability = float(cfg.get("minimum_fisher_bootstrap_cosine_median", 0.55))
        self.minimum_feature_coverage = float(cfg.get("minimum_feature_coverage", 0.60))
        self.refresh_seconds = float(cfg.get("refresh_seconds", 900))
        self.bootstrap_draws = int(cfg.get("bootstrap_draws", 3000))
        self.bridge = CryptoLibraryBridge(settings.crypto_repo_root)
        self.attribution = PerformanceAttributionEngine(settings, mode=mode)
        self.root = Path(settings.project_root) / "output/crypto_ai_swing/research/swing_geometry"
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.root / "latest.json"
        self.history_path = self.root / "history.jsonl"
        self._latest: dict[str, Any] = {}
        self._last_refresh = 0.0

    def forward_database_path(self) -> Path:
        cfg = dict((getattr(self.settings, "autonomy", {}) or {}).get(
            "forward_evidence", {}
        ) or {})
        return Path(self.settings.project_root) / cfg.get(
            "path", "output/crypto_ai_swing/forward/forward.sqlite"
        )

    def _load_rows(self) -> list[dict[str, Any]]:
        path = self.forward_database_path()
        if not path.is_file():
            return []
        conn = sqlite3.connect(path)
        try:
            rows = conn.execute(
                """
                SELECT s.observation_id,s.observed_at,s.market,s.context,
                       o.horizon_hours,o.matured_at,o.return_bps,o.mfe_bps,o.mae_bps
                FROM signal_observations s
                JOIN forward_outcomes_v2 o
                  ON o.observation_id=s.observation_id
                WHERE s.side='BUY'
                  AND s.blocked=0
                  AND o.horizon_hours IN ({})
                ORDER BY s.observed_at,s.observation_id,o.horizon_hours
                """.format(",".join("?" for _ in self.horizons)),
                self.horizons,
            ).fetchall()
        finally:
            conn.close()
        output = []
        for oid, observed_at, market, raw_context, horizon, matured_at, ret, mfe, mae in rows:
            try:
                context = json.loads(raw_context or "{}")
                gross = float(ret)
            except Exception:
                continue
            if not isinstance(context, dict) or not np.isfinite(gross):
                continue
            normal_cost, stressed_cost = self.attribution._row_costs(context)
            output.append({
                "observation_id": str(oid),
                "observed_at": str(observed_at),
                "observed_dt": _dt(observed_at),
                "market": str(market),
                "context": context,
                "horizon_hours": int(horizon),
                "matured_at": str(matured_at),
                "matured_dt": _dt(matured_at),
                "gross_return_bps": gross,
                "mfe_bps": _f(mfe, 0.0) or 0.0,
                "mae_bps": _f(mae, 0.0) or 0.0,
                "normal_cost_bps": normal_cost,
                "stressed_cost_bps": stressed_cost,
                "normal_net_bps": gross - normal_cost,
                "stressed_net_bps": gross - stressed_cost,
            })
        return output

    def _geometry(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        anchor = [r for r in rows if r["horizon_hours"] == self.anchor_horizon]
        if len(anchor) < self.minimum_geometry_fit:
            return {
                "status": "COLLECTING",
                "observations": len(anchor),
                "minimum_fit_observations": self.minimum_geometry_fit,
                "minimum_qualification_observations": self.minimum_geometry_qualification,
            }
        vectors = [feature_vector(r["context"]) for r in anchor]
        matrix = np.asarray(
            [[v.get(name, np.nan) for name in BASE_FEATURES] for v in vectors],
            dtype=float,
        )
        labels = np.asarray([1 if r["normal_net_bps"] > 0.0 else 0 for r in anchor], dtype=int)
        model = fit_shrinkage_geometry(
            matrix,
            labels,
            feature_names=BASE_FEATURES,
            minimum_feature_coverage=self.minimum_feature_coverage,
            bootstrap_draws=200,
            seed=2601,
        )
        model["anchor_horizon_hours"] = self.anchor_horizon
        model["minimum_qualification_observations"] = self.minimum_geometry_qualification
        stability = _f(model.get("fisher_bootstrap_cosine_median"), 0.0) or 0.0
        model["structurally_qualified"] = bool(
            model.get("status") == "READY"
            and len(anchor) >= self.minimum_geometry_qualification
            and stability >= self.minimum_fisher_stability
            and float(model.get("fisher_separation") or 0.0) > 0.0
        )
        model["minimum_fisher_bootstrap_cosine_median"] = self.minimum_fisher_stability
        return model

    def _horizon_table(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        table = {}
        for horizon in self.horizons:
            selected = [r for r in rows if r["horizon_hours"] == horizon]
            quality = return_quality(
                [r["normal_net_bps"] for r in selected],
                [r["gross_return_bps"] for r in selected],
                [r["normal_cost_bps"] for r in selected],
            )
            stressed = _summary([r["stressed_net_bps"] for r in selected])
            posterior = bayesian_mean_posterior(
                [r["normal_net_bps"] for r in selected],
                draws=max(2000, self.bootstrap_draws),
                seed=2600 + horizon,
            ) if selected else {"status": "NO_DATA", "observations": 0}
            table[str(horizon)] = {
                "horizon_hours": horizon,
                "status": "READY" if selected else "COLLECTING",
                **quality,
                "stressed_net": stressed,
                "bayesian_normal_net": posterior,
                "mean_mfe_bps": float(np.mean([r["mfe_bps"] for r in selected])) if selected else None,
                "mean_mae_bps": float(np.mean([r["mae_bps"] for r in selected])) if selected else None,
                "swing_horizon": horizon in self.swing_horizons,
            }
        return table

    def _prior_horizon_choice(
        self,
        history: Mapping[int, list[dict[str, Any]]],
    ) -> dict[str, Any] | None:
        best = None
        for horizon in self.swing_horizons:
            rows = list(history.get(horizon) or [])
            if len(rows) < self.minimum_horizon_history:
                continue
            normal = np.asarray([r["normal_net_bps"] for r in rows], dtype=float)
            gross = np.asarray([r["gross_return_bps"] for r in rows], dtype=float)
            costs = np.asarray([r["normal_cost_bps"] for r in rows], dtype=float)
            summary = _summary(normal)
            quality = return_quality(normal, gross, costs)
            burden = quality.get("cost_fraction_of_mean_winner")
            se = float(normal.std(ddof=1) / np.sqrt(len(normal))) if len(normal) > 1 else 1e9
            lcb80 = float(normal.mean() - 1.2815515655446004 * se)
            preliminary = bool(
                float(summary.get("mean_bps") or -1e12) > 0.0
                and float(summary.get("profit_factor") or 0.0) >= 1.05
                and burden is not None
                and float(burden) <= self.maximum_cost_fraction
            )
            candidate = {
                "horizon_hours": horizon,
                "history_observations": len(rows),
                "mean_normal_net_bps": float(normal.mean()),
                "lcb80_normal_net_bps": lcb80,
                "profit_factor": summary.get("profit_factor"),
                "payoff_ratio": summary.get("payoff_ratio"),
                "positive_fraction": summary.get("positive_fraction"),
                "cost_fraction_of_mean_winner": burden,
                "preliminary_pass": preliminary,
                "objective": lcb80 + 0.05 * float(np.median(normal)),
            }
            if preliminary and (best is None or candidate["objective"] > best["objective"]):
                best = candidate
        return best

    def _stochastic(self, normal: np.ndarray, stressed: np.ndarray) -> dict[str, Any]:
        if len(normal) < self.minimum_oos:
            return {
                "status": "COLLECTING",
                "passed": False,
                "observation_count": int(len(normal)),
                "required": self.minimum_oos,
            }
        try:
            module = self.bridge.import_module("research.stochastic_validation")
            policy = module.StochasticValidationPolicy(
                simulations=10_000,
                expected_block_length=max(3, min(12, len(normal) // 8)),
                maximum_drawdown=0.15,
                maximum_drawdown_breach_probability=0.01,
                maximum_terminal_loss_probability=0.05,
                minimum_p05_total_return=0.0,
                dirichlet_blocks=8,
                dirichlet_concentrations=(0.5, 1.0, 5.0),
                minimum_observations=self.minimum_oos,
                confidence_level=0.95,
                seed=2604,
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

    def _sequential_oos(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_observation: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("observed_dt") is not None and row.get("matured_dt") is not None:
                by_observation[row["observation_id"]].append(row)
        groups = sorted(
            by_observation.values(),
            key=lambda group: (group[0]["observed_dt"], group[0]["observation_id"]),
        )
        matured_sorted = sorted(
            [r for r in rows if r.get("matured_dt") is not None],
            key=lambda row: row["matured_dt"],
        )
        history: dict[int, list[dict[str, Any]]] = defaultdict(list)
        pointer = 0
        selected = []
        audit = []
        for group in groups:
            decision_at = group[0]["observed_dt"]
            while pointer < len(matured_sorted) and matured_sorted[pointer]["matured_dt"] <= decision_at:
                matured = matured_sorted[pointer]
                history[int(matured["horizon_hours"])].append(matured)
                pointer += 1
            choice = self._prior_horizon_choice(history)
            if choice is None:
                continue
            horizon = int(choice["horizon_hours"])
            outcome = next((r for r in group if int(r["horizon_hours"]) == horizon), None)
            if outcome is None:
                continue
            selected.append(outcome)
            audit.append({
                "observed_at": outcome["observed_at"],
                "market": outcome["market"],
                "selected_horizon_hours": horizon,
                "history_observations": choice["history_observations"],
                "estimated_normal_net_bps": choice["mean_normal_net_bps"],
                "estimated_lcb80_bps": choice["lcb80_normal_net_bps"],
                "cost_fraction_of_mean_winner": choice["cost_fraction_of_mean_winner"],
                "realized_normal_net_bps": outcome["normal_net_bps"],
                "label_matured_at": outcome["matured_at"],
                "lookahead_check": bool(outcome["matured_dt"] > decision_at),
            })

        normal = np.asarray([r["normal_net_bps"] for r in selected], dtype=float)
        stressed = np.asarray([r["stressed_net_bps"] for r in selected], dtype=float)
        gross = np.asarray([r["gross_return_bps"] for r in selected], dtype=float)
        costs = np.asarray([r["normal_cost_bps"] for r in selected], dtype=float)
        normal_summary = _summary(normal)
        stressed_summary = _summary(stressed)
        quality = return_quality(normal, gross, costs)
        posterior = bayesian_mean_posterior(
            normal,
            draws=max(3000, self.bootstrap_draws),
            seed=2605,
        ) if len(normal) else {"status": "INSUFFICIENT", "observations": 0}
        stochastic = self._stochastic(normal, stressed)
        swing_fraction = (
            float(np.mean([r["horizon_hours"] >= 24 for r in selected]))
            if selected else 0.0
        )
        checks = {
            "minimum_total_observations": len({r["observation_id"] for r in rows}) >= self.minimum_total,
            "minimum_oos_selected": len(selected) >= self.minimum_oos,
            "positive_normal_mean": float(normal_summary.get("mean_bps") or -1e12) > 0.0,
            "positive_stressed_mean": float(stressed_summary.get("mean_bps") or -1e12) > 0.0,
            "positive_fraction": float(normal_summary.get("positive_fraction") or 0.0) >= self.minimum_positive_fraction,
            "profit_factor": float(normal_summary.get("profit_factor") or 0.0) >= self.minimum_profit_factor,
            "payoff_ratio": float(normal_summary.get("payoff_ratio") or 0.0) >= self.minimum_payoff_ratio,
            "cost_burden": (
                quality.get("cost_fraction_of_mean_winner") is not None
                and float(quality["cost_fraction_of_mean_winner"]) <= self.maximum_cost_fraction
            ),
            "bayesian_probability": float(posterior.get("probability_mean_positive") or 0.0) >= self.minimum_probability_positive,
            "bayesian_p05": float(posterior.get("mean_p05_bps") or -1e12) > 0.0,
            "stochastic": bool(stochastic.get("passed", False)),
            "swing_horizon_fraction": swing_fraction >= 0.80,
            "delayed_label_audit": all(bool(row.get("lookahead_check")) for row in audit),
        }
        return {
            "status": "QUALIFIED" if all(checks.values()) else "COLLECTING",
            "qualified": all(checks.values()),
            "oos_selected": len(selected),
            "normal_net": normal_summary,
            "stressed_net": stressed_summary,
            "gross": _summary(gross),
            "cost_fraction_of_mean_winner": quality.get("cost_fraction_of_mean_winner"),
            "bayesian": posterior,
            "stochastic_validation": stochastic,
            "swing_horizon_fraction": swing_fraction,
            "selected_horizon_counts": {
                str(h): sum(int(r["horizon_hours"]) == h for r in selected)
                for h in self.swing_horizons
            },
            "checks": checks,
            "recent_audit": audit[-25:],
            "lookahead_policy": "HORIZON_SELECTED_ONLY_FROM_LABELS_MATURED_BEFORE_DECISION",
        }

    def refresh(self, *, force: bool = False) -> dict[str, Any]:
        now = time.time()
        if not force and self._latest and now - self._last_refresh < self.refresh_seconds:
            return dict(self._latest)
        rows = self._load_rows()
        geometry = self._geometry(rows)
        horizon_table = self._horizon_table(rows)
        oos = self._sequential_oos(rows)
        geometry_ok = bool(geometry.get("structurally_qualified", False))
        qualified = bool(oos.get("qualified", False) and geometry_ok)
        checks = {
            **dict(oos.get("checks") or {}),
            "geometry_structurally_qualified": geometry_ok,
        }
        payload = {
            "schema_version": self.SCHEMA,
            "generated_at": _now(),
            "status": "QUALIFIED" if qualified else "COLLECTING",
            "qualified": qualified,
            "observations": len({r["observation_id"] for r in rows}),
            "outcome_rows": len(rows),
            "horizons_hours": list(self.horizons),
            "swing_horizons_hours": list(self.swing_horizons),
            "anchor_horizon_hours": self.anchor_horizon,
            "cost_policy": {
                "full_costs_always_deducted": True,
                "maximum_cost_fraction_of_mean_winner": self.maximum_cost_fraction,
                "interpretation": (
                    "Swing tolerance is relative to expected move magnitude; fees, spread "
                    "and slippage are never removed from net-return evidence."
                ),
            },
            "reference_cost_burdens": self._reference_cost_burdens(horizon_table),
            "horizon_economics": horizon_table,
            "geometry": geometry,
            "sequential_oos": {**oos, "checks": checks, "qualified": qualified},
            "shadow_decision_influence": bool(qualified and self.mode != "live"),
            "live_decision_influence": False,
            "automatic_live_promotion": False,
            "orders_submitted": 0,
            "artifact_path": str(self.latest_path),
        }
        self.latest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        self._latest = dict(payload)
        self._last_refresh = now
        return payload

    @staticmethod
    def _reference_cost_burdens(horizon_table: Mapping[str, Any]) -> dict[str, Any]:
        costs = []
        for row in horizon_table.values():
            value = _f((row or {}).get("mean_cost_bps"), None)
            if value is not None:
                costs.append(value)
        mean_cost = float(np.mean(costs)) if costs else None
        return {
            "mean_observed_round_trip_cost_bps": mean_cost,
            "at_2pct_move": (mean_cost / 200.0 if mean_cost is not None else None),
            "at_3pct_move": (mean_cost / 300.0 if mean_cost is not None else None),
            "at_5pct_move": (mean_cost / 500.0 if mean_cost is not None else None),
            "diagnostic_only": True,
        }

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
        model = dict(active.get("geometry") or {})
        score = score_geometry(model, feature_vector(context))
        qualified = bool(active.get("qualified", False))
        passes = bool(score.get("passes", False) and qualified and self.mode != "live")
        return {
            "schema_version": "crypto_ai_swing_geometry_market_gate_v1",
            **score,
            "geometry_qualified": bool(model.get("structurally_qualified", False)),
            "swing_geometry_qualified": qualified,
            "passes": passes,
            "shadow_decision_influence": passes,
            "live_decision_influence": False,
            "authority": "ADVISORY_ONLY",
        }

    @staticmethod
    def policy_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
        oos = dict(payload.get("sequential_oos") or {})
        geometry = dict(payload.get("geometry") or {})
        return {
            "status": payload.get("status", "COLLECTING"),
            "qualified": bool(payload.get("qualified", False)),
            "observations": payload.get("observations"),
            "oos_selected": oos.get("oos_selected"),
            "normal_mean_bps": (oos.get("normal_net") or {}).get("mean_bps"),
            "stressed_mean_bps": (oos.get("stressed_net") or {}).get("mean_bps"),
            "profit_factor": (oos.get("normal_net") or {}).get("profit_factor"),
            "payoff_ratio": (oos.get("normal_net") or {}).get("payoff_ratio"),
            "cost_fraction_of_mean_winner": oos.get("cost_fraction_of_mean_winner"),
            "effective_rank": geometry.get("effective_rank"),
            "fisher_stability": geometry.get("fisher_bootstrap_cosine_median"),
            "reason_codes": [
                name.upper() + "_FAILED"
                for name, passed in dict(oos.get("checks") or {}).items()
                if not passed
            ],
            "artifact_path": str(payload.get("artifact_path") or ""),
            "live_decision_influence": False,
        }
