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

from crypto_ai_swing.agents.component_features import extract_descriptors
from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.research.performance_attribution import (
    PerformanceAttributionEngine,
    bayesian_mean_posterior,
)
from crypto_ai_swing.research.swing_geometry import (
    BASE_FEATURES,
    feature_vector,
    fit_shrinkage_geometry,
    score_geometry,
)


ZERO = 0.0


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


def selector_feature_vector(context: Mapping[str, Any]) -> dict[str, float | None]:
    base = feature_vector(context)
    desc = extract_descriptors(context)

    def val(name: str) -> float | None:
        return _f(base.get(name), None)

    mtf = val("mtf")
    technical = val("technical")
    execution = val("execution")
    macro = val("mtf_macro")
    trend = val("mtf_trend")
    setup = val("mtf_setup")
    trigger = val("mtf_trigger")
    spread = _f(desc.get("spread_bps"), None)
    state = str(desc.get("breakout_state") or "")
    family = str(desc.get("strategy_family") or "")

    result = dict(base)
    result.update(
        {
            "interaction_mtf_technical": (
                mtf * technical
                if mtf is not None and technical is not None
                else None
            ),
            "interaction_mtf_execution": (
                mtf * execution
                if mtf is not None and execution is not None
                else None
            ),
            "interaction_macro_trend": (
                macro * trend
                if macro is not None and trend is not None
                else None
            ),
            "interaction_setup_trigger": (
                setup * trigger
                if setup is not None and trigger is not None
                else None
            ),
            "spread_log1p_scaled": (
                float(np.log1p(max(0.0, spread)) / np.log(21.0))
                if spread is not None
                else None
            ),
            "not_overextended": 0.0 if bool(desc.get("overextended")) else 1.0,
            "state_donchian55": 1.0 if state == "DONCHIAN_55_CONFIRMED" else 0.0,
            "state_trend_continuation": 1.0 if state == "TREND_CONTINUATION" else 0.0,
            "state_trend_pullback": 1.0 if state == "TREND_PULLBACK" else 0.0,
            "family_breakout": 1.0 if family == "BREAKOUT_CONFIRMATION" else 0.0,
            "family_rally": 1.0 if family == "RALLY_ACCELERATION" else 0.0,
            "family_pullback": 1.0 if family == "TREND_PULLBACK" else 0.0,
        }
    )
    return result


SELECTOR_FEATURES = tuple(BASE_FEATURES) + (
    "interaction_mtf_technical",
    "interaction_mtf_execution",
    "interaction_macro_trend",
    "interaction_setup_trigger",
    "spread_log1p_scaled",
    "not_overextended",
    "state_donchian55",
    "state_trend_continuation",
    "state_trend_pullback",
    "family_breakout",
    "family_rally",
    "family_pullback",
)


def _ridge_multiresponse(
    design: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    """Solve multi-response ridge without explicitly inverting X'X."""
    x = np.asarray(design, dtype=float)
    y = np.asarray(targets, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("design/targets shape mismatch")
    penalty = np.eye(x.shape[1], dtype=float) * float(ridge)
    penalty[0, 0] = 0.0
    lhs = x.T @ x + penalty
    rhs = x.T @ y
    return np.linalg.solve(lhs, rhs)


def fit_factor_expectancy(
    matrix: np.ndarray,
    labels: np.ndarray,
    targets: np.ndarray,
    *,
    feature_names: Sequence[str],
    minimum_feature_coverage: float = 0.60,
    ridge: float = 1.0,
    maximum_factors: int = 8,
    bootstrap_draws: int = 100,
    seed: int = 2701,
) -> dict[str, Any]:
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(labels, dtype=int).reshape(-1)
    target = np.asarray(targets, dtype=float)
    if x.ndim != 2 or target.ndim != 2 or len(x) != len(y) or len(x) != len(target):
        raise ValueError("matrix/labels/targets shape mismatch")
    if np.sum(y > 0) < 3 or np.sum(y <= 0) < 3:
        return {
            "status": "COLLECTING",
            "observations": int(len(y)),
            "positive_class": int(np.sum(y > 0)),
            "negative_class": int(np.sum(y <= 0)),
            "reason": "INSUFFICIENT_CLASS_SUPPORT",
        }

    geometry = fit_shrinkage_geometry(
        x,
        y,
        feature_names=feature_names,
        minimum_feature_coverage=minimum_feature_coverage,
        bootstrap_draws=max(50, min(300, int(bootstrap_draws))),
        seed=seed,
    )
    if geometry.get("status") != "READY":
        return {
            "status": "COLLECTING",
            "observations": int(len(y)),
            "geometry": geometry,
            "reason": "GEOMETRY_NOT_READY",
        }

    names = list(geometry["feature_names"])
    index = [list(feature_names).index(name) for name in names]
    raw = x[:, index]
    center = np.asarray(geometry["center"], dtype=float)
    scale = np.asarray(geometry["scale"], dtype=float)
    raw = np.where(np.isfinite(raw), raw, center)
    z = (raw - center) / scale

    covariance = np.asarray(geometry["covariance"], dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    variance_fraction = eigenvalues / max(float(eigenvalues.sum()), 1e-12)
    cumulative = np.cumsum(variance_fraction)
    k90 = int(np.searchsorted(cumulative, 0.90) + 1)
    factor_count = max(
        1,
        min(
            int(maximum_factors),
            int(k90),
            int(len(eigenvalues)),
            max(1, len(z) // 5),
        ),
    )

    # v0.27.1: retain supervised predictive directions before appending
    # unsupervised principal components. Pure PCA truncation can discard a
    # low-variance but highly predictive swing direction.
    basis: list[np.ndarray] = []

    def add_direction(vector: np.ndarray) -> None:
        candidate = np.asarray(vector, dtype=float).reshape(-1).copy()
        if not np.all(np.isfinite(candidate)):
            return
        for existing in basis:
            candidate -= float(candidate @ existing) * existing
        norm = float(np.linalg.norm(candidate))
        if norm > 1e-8:
            basis.append(candidate / norm)

    regularized_covariance = (
        covariance + float(ridge) * np.eye(covariance.shape[0])
    )

    # Supervised normal-net, stressed-net, MFE and MAE directions.
    for target_idx in range(target.shape[1]):
        target_column = target[:, target_idx]
        centered_target = target_column - float(np.mean(target_column))
        target_scale = float(np.std(centered_target, ddof=1))
        if not np.isfinite(target_scale) or target_scale <= 1e-8:
            continue
        cross_covariance = (
            z.T @ (centered_target / target_scale)
        ) / max(1, len(z) - 1)
        try:
            direction = np.linalg.solve(
                regularized_covariance,
                cross_covariance,
            )
        except np.linalg.LinAlgError:
            continue
        add_direction(direction)
        if len(basis) >= factor_count:
            break

    # Preserve Fisher winner/loser separation if it adds an independent axis.
    if len(basis) < factor_count:
        fisher = np.asarray(
            geometry.get("fisher_direction") or [],
            dtype=float,
        )
        if fisher.shape == (z.shape[1],):
            add_direction(fisher)

    # Complete the subspace with descending-variance principal components.
    for component_idx in range(eigenvectors.shape[1]):
        if len(basis) >= factor_count:
            break
        add_direction(eigenvectors[:, component_idx])

    if not basis:
        return {
            "status": "COLLECTING",
            "observations": int(len(y)),
            "geometry": geometry,
            "reason": "NO_STABLE_FACTOR_BASIS",
        }

    projection = np.column_stack(basis[:factor_count])
    factor_count = int(projection.shape[1])
    factor_variance_fraction = [
        float(
            projection[:, idx].T
            @ covariance
            @ projection[:, idx]
            / max(float(np.trace(covariance)), 1e-12)
        )
        for idx in range(factor_count)
    ]
    factors = z @ projection
    design = np.column_stack([np.ones(len(factors)), factors])
    coefficients = _ridge_multiresponse(
        design,
        target,
        ridge=float(ridge),
    )
    fitted = design @ coefficients
    residuals = target - fitted
    residual_std = np.std(residuals, axis=0, ddof=1) if len(residuals) > 1 else np.zeros(target.shape[1])

    direction = coefficients[1:, 0]
    direction_norm = float(np.linalg.norm(direction))
    stability: list[float] = []
    if direction_norm > 0 and bootstrap_draws > 0 and len(factors) >= 20:
        reference = direction / direction_norm
        rng = np.random.default_rng(seed + 11)
        for _ in range(min(int(bootstrap_draws), 300)):
            idx = rng.integers(0, len(factors), size=len(factors))
            xb = design[idx]
            yb = target[idx]
            try:
                coef_b = _ridge_multiresponse(xb, yb, ridge=float(ridge))
            except np.linalg.LinAlgError:
                continue
            candidate = coef_b[1:, 0]
            norm = float(np.linalg.norm(candidate))
            if norm <= 0:
                continue
            cosine = float(np.dot(reference, candidate / norm))
            stability.append(abs(cosine))

    return {
        "status": "READY",
        "observations": int(len(y)),
        "feature_names": names,
        "geometry": geometry,
        "projection": projection.tolist(),
        "factor_count": factor_count,
        "factor_variance_fraction": factor_variance_fraction,
        "ridge": float(ridge),
        "coefficients": coefficients.tolist(),
        "target_residual_std": np.asarray(residual_std, dtype=float).tolist(),
        "return_direction_bootstrap_cosine_median": (
            float(np.median(stability)) if stability else None
        ),
        "return_direction_bootstrap_cosine_p05": (
            float(np.quantile(stability, 0.05)) if stability else None
        ),
        "linear_algebra_contract": {
            "covariance": "LEDOIT_WOLF_SHRINKAGE",
            "factorization": "SYMMETRIC_EIGENDECOMPOSITION",
            "factor_selection": "SUPERVISED_SIGNAL_DIRECTIONS_THEN_ORTHOGONAL_PCA_90PCT_CAPPED",
            "expectancy_model": "MULTIRESPONSE_RIDGE_ON_SUPERVISED_ORTHOGONAL_FACTORS",
            "solver": "LINEAR_SOLVE_NO_EXPLICIT_MATRIX_INVERSE",
            "ood": "MAHALANOBIS",
        },
    }


def score_factor_expectancy(
    model: Mapping[str, Any],
    values: Mapping[str, Any],
) -> dict[str, Any]:
    if str(model.get("status")) != "READY":
        return {"status": "COLLECTING"}
    geometry = dict(model.get("geometry") or {})
    geometry_score = score_geometry(geometry, values)
    names = list(model.get("feature_names") or [])
    center = np.asarray(geometry.get("center"), dtype=float)
    scale = np.asarray(geometry.get("scale"), dtype=float)
    raw = np.asarray([_f(values.get(name), np.nan) for name in names], dtype=float)
    raw = np.where(np.isfinite(raw), raw, center)
    z = (raw - center) / scale
    projection = np.asarray(model.get("projection"), dtype=float)
    factors = z @ projection
    design = np.concatenate(([1.0], factors))
    coefficients = np.asarray(model.get("coefficients"), dtype=float)
    prediction = design @ coefficients
    residual_std = np.asarray(model.get("target_residual_std"), dtype=float)
    return {
        "status": "READY",
        "expected_normal_net_bps": float(prediction[0]),
        "expected_stressed_net_bps": float(prediction[1]),
        "expected_mfe_bps": float(max(0.0, prediction[2])),
        "expected_abs_mae_bps": float(max(0.0, prediction[3])),
        "prediction_residual_std_bps": residual_std.tolist(),
        "geometry": geometry_score,
    }


class ProspectiveSwingEntrySelector:
    """Delayed-label-safe active-swing entry selector with abstention.

    The selector is research/paper/shadow only. Historical decisions are
    evaluated using only outcomes whose matured_at timestamp was available
    before that decision. Full costs remain deducted. ABSTAIN is a first-class
    action and is never converted into a BUY merely to increase trade count.
    """

    SCHEMA = "crypto_ai_swing_prospective_entry_selector_v1"

    def __init__(self, settings, *, mode: str = "paper") -> None:
        self.settings = settings
        self.mode = str(mode).lower()
        cfg = dict(
            (getattr(settings, "autonomy", {}) or {}).get(
                "entry_selector",
                {},
            )
            or {}
        )
        self.horizons = tuple(
            int(x)
            for x in cfg.get("horizons_hours", (24, 72, 168))
            if int(x) > 0
        )
        self.minimum_train = int(cfg.get("minimum_train_observations", 24))
        self.minimum_total = int(cfg.get("minimum_qualification_total", 120))
        self.minimum_oos_selected = int(cfg.get("minimum_oos_selected", 30))
        self.minimum_probability = float(cfg.get("minimum_geometry_probability", 0.62))
        self.minimum_expected_normal = float(cfg.get("minimum_expected_normal_net_bps", 20.0))
        self.minimum_expected_stressed = float(cfg.get("minimum_expected_stressed_net_bps", 0.0))
        self.minimum_mfe_cost_multiple = float(cfg.get("minimum_expected_mfe_cost_multiple", 2.0))
        self.minimum_excursion_ratio = float(cfg.get("minimum_expected_mfe_to_mae", 1.20))
        self.minimum_positive_fraction = float(cfg.get("minimum_positive_fraction", 0.35))
        self.minimum_profit_factor = float(cfg.get("minimum_profit_factor", 1.20))
        self.minimum_payoff_ratio = float(cfg.get("minimum_payoff_ratio", 1.10))
        self.minimum_probability_positive = float(cfg.get("minimum_probability_positive", 0.95))
        self.maximum_selection_fraction = float(cfg.get("maximum_selection_fraction", 0.45))
        self.minimum_direction_stability = float(
            cfg.get("minimum_return_direction_bootstrap_cosine_median", 0.55)
        )
        self.minimum_feature_coverage = float(cfg.get("minimum_feature_coverage", 0.60))
        self.ridge = float(cfg.get("ridge", 1.0))
        self.maximum_factors = int(cfg.get("maximum_factors", 8))
        self.bootstrap_draws = int(cfg.get("bootstrap_draws", 3000))
        self.refresh_seconds = float(cfg.get("refresh_seconds", 900))
        self.attribution = PerformanceAttributionEngine(settings, mode=mode)
        self.bridge = CryptoLibraryBridge(settings.crypto_repo_root)
        self.root = (
            Path(settings.project_root)
            / "output/crypto_ai_swing/research/entry_selector"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.root / "latest.json"
        self.history_path = self.root / "history.jsonl"
        self._latest: dict[str, Any] = {}
        self._last_refresh = 0.0

    def _forward_database_path(self) -> Path:
        cfg = dict(
            (getattr(self.settings, "autonomy", {}) or {}).get(
                "forward_evidence",
                {},
            )
            or {}
        )
        return Path(self.settings.project_root) / cfg.get(
            "path",
            "output/crypto_ai_swing/forward/forward.sqlite",
        )

    def _load_rows(self) -> list[dict[str, Any]]:
        path = self._forward_database_path()
        if not path.is_file():
            return []
        conn = sqlite3.connect(path)
        try:
            rows = conn.execute(
                """
                SELECT
                    s.observation_id,
                    s.observed_at,
                    s.market,
                    s.context,
                    o.horizon_hours,
                    o.matured_at,
                    o.return_bps,
                    o.mfe_bps,
                    o.mae_bps
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

        result = []
        for oid, observed_at, market, raw_context, horizon, matured_at, ret, mfe, mae in rows:
            try:
                context = json.loads(raw_context or "{}")
                gross = float(ret)
            except Exception:
                continue
            observed_dt = _dt(observed_at)
            matured_dt = _dt(matured_at)
            if (
                not isinstance(context, dict)
                or observed_dt is None
                or matured_dt is None
                or not np.isfinite(gross)
            ):
                continue
            normal_cost, stressed_cost = self.attribution._row_costs(context)
            result.append(
                {
                    "observation_id": str(oid),
                    "observed_at": str(observed_at),
                    "observed_dt": observed_dt,
                    "market": str(market),
                    "context": context,
                    "horizon_hours": int(horizon),
                    "matured_at": str(matured_at),
                    "matured_dt": matured_dt,
                    "gross_return_bps": gross,
                    "mfe_bps": float(_f(mfe, 0.0) or 0.0),
                    "mae_bps": float(_f(mae, 0.0) or 0.0),
                    "normal_cost_bps": float(normal_cost),
                    "stressed_cost_bps": float(stressed_cost),
                    "normal_net_bps": gross - float(normal_cost),
                    "stressed_net_bps": gross - float(stressed_cost),
                }
            )
        return result

    def _fit_model(self, rows: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
        selected = [row for row in rows if int(row["horizon_hours"]) == int(horizon)]
        if len(selected) < self.minimum_train:
            return {
                "status": "COLLECTING",
                "horizon_hours": int(horizon),
                "observations": len(selected),
                "required": self.minimum_train,
            }
        vectors = [selector_feature_vector(row["context"]) for row in selected]
        matrix = np.asarray(
            [
                [vector.get(name, np.nan) for name in SELECTOR_FEATURES]
                for vector in vectors
            ],
            dtype=float,
        )
        labels = np.asarray(
            [1 if row["normal_net_bps"] > 0.0 else 0 for row in selected],
            dtype=int,
        )
        targets = np.asarray(
            [
                [
                    row["normal_net_bps"],
                    row["stressed_net_bps"],
                    row["mfe_bps"],
                    abs(row["mae_bps"]),
                ]
                for row in selected
            ],
            dtype=float,
        )
        model = fit_factor_expectancy(
            matrix,
            labels,
            targets,
            feature_names=SELECTOR_FEATURES,
            minimum_feature_coverage=self.minimum_feature_coverage,
            ridge=self.ridge,
            maximum_factors=self.maximum_factors,
            bootstrap_draws=min(300, max(50, self.bootstrap_draws // 10)),
            seed=2700 + int(horizon),
        )
        model["horizon_hours"] = int(horizon)
        model["training_cutoff_matured_at"] = max(
            row["matured_dt"] for row in selected
        ).isoformat()
        return model

    def _decision(
        self,
        model: Mapping[str, Any],
        context: Mapping[str, Any],
        *,
        normal_cost_bps: float,
    ) -> dict[str, Any]:
        if model.get("status") != "READY":
            return {
                "status": "COLLECTING",
                "action": "ABSTAIN",
                "passes": False,
                "reason_codes": ["MODEL_NOT_READY"],
            }
        values = selector_feature_vector(context)
        prediction = score_factor_expectancy(model, values)
        geometry = dict(prediction.get("geometry") or {})
        expected_normal = float(prediction.get("expected_normal_net_bps") or -1e12)
        expected_stressed = float(prediction.get("expected_stressed_net_bps") or -1e12)
        expected_mfe = float(prediction.get("expected_mfe_bps") or 0.0)
        expected_mae = float(prediction.get("expected_abs_mae_bps") or 0.0)
        probability = float(geometry.get("fisher_probability_like") or 0.0)
        ood = bool(geometry.get("out_of_distribution", True))
        excursion_ratio = (
            expected_mfe / expected_mae if expected_mae > 1e-9 else None
        )
        cost_multiple = (
            expected_mfe / float(normal_cost_bps)
            if normal_cost_bps > 1e-9
            else None
        )
        stability = _f(
            model.get("return_direction_bootstrap_cosine_median"),
            None,
        )
        checks = {
            "not_ood": not ood,
            "geometry_probability": probability >= self.minimum_probability,
            "positive_normal_expectancy": expected_normal >= self.minimum_expected_normal,
            "positive_stressed_expectancy": expected_stressed >= self.minimum_expected_stressed,
            "mfe_cost_cover": (
                cost_multiple is not None
                and cost_multiple >= self.minimum_mfe_cost_multiple
            ),
            "excursion_asymmetry": (
                excursion_ratio is not None
                and excursion_ratio >= self.minimum_excursion_ratio
            ),
            "direction_stability": (
                stability is None
                or stability >= self.minimum_direction_stability
            ),
        }
        passes = all(checks.values())
        risk_scale = max(expected_mae, 50.0)
        utility = (
            expected_stressed / risk_scale
            + 0.10 * expected_normal / risk_scale
            + 0.05 * max(0.0, expected_mfe - normal_cost_bps) / risk_scale
        )
        return {
            "status": "READY",
            "action": (
                f"SELECT_{int(model.get('horizon_hours') or 0)}H"
                if passes
                else "ABSTAIN"
            ),
            "passes": passes,
            "horizon_hours": int(model.get("horizon_hours") or 0),
            "expected_normal_net_bps": expected_normal,
            "expected_stressed_net_bps": expected_stressed,
            "expected_mfe_bps": expected_mfe,
            "expected_abs_mae_bps": expected_mae,
            "expected_mfe_to_mae": excursion_ratio,
            "expected_mfe_cost_multiple": cost_multiple,
            "geometry_probability_like": probability,
            "mahalanobis_distance": geometry.get("mahalanobis_distance"),
            "out_of_distribution": ood,
            "utility": float(utility),
            "checks": checks,
            "reason_codes": [
                name.upper() + "_FAILED"
                for name, passed in checks.items()
                if not passed
            ],
        }

    def _walk_forward(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_observation: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_observation[row["observation_id"]].append(row)
        observations = sorted(
            by_observation.values(),
            key=lambda group: (
                group[0]["observed_dt"],
                group[0]["observation_id"],
            ),
        )
        selected_rows: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        total_scored = 0

        for group in observations:
            decision_time = group[0]["observed_dt"]
            context = group[0]["context"]
            current_by_horizon = {
                int(row["horizon_hours"]): row for row in group
            }
            candidates = []
            prior_counts: dict[str, int] = {}

            for horizon in self.horizons:
                current = current_by_horizon.get(int(horizon))
                if current is None:
                    continue
                prior = [
                    row
                    for row in rows
                    if int(row["horizon_hours"]) == int(horizon)
                    and row["matured_dt"] <= decision_time
                    and row["observation_id"] != current["observation_id"]
                ]
                prior_counts[str(horizon)] = len(prior)
                if len(prior) < self.minimum_train:
                    continue
                model = self._fit_model(prior, int(horizon))
                decision = self._decision(
                    model,
                    context,
                    normal_cost_bps=current["normal_cost_bps"],
                )
                if decision.get("status") == "READY":
                    total_scored += 1
                if decision.get("passes"):
                    candidates.append((float(decision["utility"]), decision, current))

            chosen = max(candidates, key=lambda item: item[0]) if candidates else None
            if chosen is not None:
                _, decision, current = chosen
                selected_rows.append(
                    {
                        **current,
                        "selector_decision": decision,
                    }
                )
                action = decision["action"]
            else:
                action = "ABSTAIN"

            audits.append(
                {
                    "observation_id": group[0]["observation_id"],
                    "market": group[0]["market"],
                    "decision_at": decision_time.isoformat(),
                    "prior_counts": prior_counts,
                    "action": action,
                }
            )

        normal = np.asarray(
            [row["normal_net_bps"] for row in selected_rows],
            dtype=float,
        )
        stressed = np.asarray(
            [row["stressed_net_bps"] for row in selected_rows],
            dtype=float,
        )
        selected_horizons: dict[str, int] = defaultdict(int)
        for row in selected_rows:
            selected_horizons[str(row["horizon_hours"])] += 1

        return {
            "status": "READY" if total_scored else "COLLECTING",
            "lookahead_policy": (
                "DECISION_T_USES_ONLY_LABELS_WITH_MATURED_AT_LE_DECISION_T"
            ),
            "total_observations": len(observations),
            "scored_horizon_decisions": total_scored,
            "oos_selected": len(selected_rows),
            "selection_fraction": (
                len(selected_rows) / len(observations)
                if observations
                else 0.0
            ),
            "abstention_fraction": (
                1.0 - len(selected_rows) / len(observations)
                if observations
                else 1.0
            ),
            "selected_horizon_counts": dict(selected_horizons),
            "normal_net": _summary(normal),
            "stressed_net": _summary(stressed),
            "recent_audit": audits[-25:],
            "_normal_array": normal,
            "_stressed_array": stressed,
        }

    def _stochastic(
        self,
        normal: np.ndarray,
        stressed: np.ndarray,
    ) -> dict[str, Any]:
        if len(normal) < self.minimum_oos_selected:
            return {
                "status": "COLLECTING",
                "passed": False,
                "observation_count": int(len(normal)),
                "required": self.minimum_oos_selected,
            }
        try:
            module = self.bridge.import_module("research.stochastic_validation")
            policy = module.StochasticValidationPolicy(
                simulations=10_000,
                expected_block_length=max(3, min(10, len(normal) // 6)),
                maximum_drawdown=0.15,
                maximum_drawdown_breach_probability=0.01,
                maximum_terminal_loss_probability=0.05,
                minimum_p05_total_return=0.0,
                dirichlet_blocks=8,
                dirichlet_concentrations=(0.5, 1.0, 5.0),
                minimum_observations=self.minimum_oos_selected,
                confidence_level=0.95,
                seed=2707,
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

    def _current_models(
        self,
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            str(horizon): self._fit_model(rows, int(horizon))
            for horizon in self.horizons
        }

    def refresh(self, *, force: bool = False) -> dict[str, Any]:
        now = time.time()
        if (
            not force
            and self._latest
            and now - self._last_refresh < self.refresh_seconds
        ):
            return dict(self._latest)

        rows = self._load_rows()
        walk = self._walk_forward(rows)
        normal = np.asarray(walk.pop("_normal_array"), dtype=float)
        stressed = np.asarray(walk.pop("_stressed_array"), dtype=float)
        posterior = bayesian_mean_posterior(
            normal,
            draws=max(3000, self.bootstrap_draws),
            seed=2708,
        )
        stochastic = self._stochastic(normal, stressed)
        normal_summary = dict(walk.get("normal_net") or {})
        stressed_summary = dict(walk.get("stressed_net") or {})
        selection_fraction = float(walk.get("selection_fraction") or 0.0)

        checks = {
            "minimum_total_observations": (
                int(walk.get("total_observations") or 0)
                >= self.minimum_total
            ),
            "minimum_oos_selected": (
                int(walk.get("oos_selected") or 0)
                >= self.minimum_oos_selected
            ),
            "selective_enough": (
                0.0 < selection_fraction <= self.maximum_selection_fraction
            ),
            "positive_normal_mean": float(
                normal_summary.get("mean_bps") or -1e12
            ) > 0.0,
            "positive_stressed_mean": float(
                stressed_summary.get("mean_bps") or -1e12
            ) > 0.0,
            "positive_fraction": float(
                normal_summary.get("positive_fraction") or 0.0
            ) >= self.minimum_positive_fraction,
            "profit_factor": float(
                normal_summary.get("profit_factor") or 0.0
            ) >= self.minimum_profit_factor,
            "payoff_ratio": float(
                normal_summary.get("payoff_ratio") or 0.0
            ) >= self.minimum_payoff_ratio,
            "bayesian_probability": float(
                posterior.get("probability_mean_positive") or 0.0
            ) >= self.minimum_probability_positive,
            "bayesian_p05": float(
                posterior.get("mean_p05_bps") or -1e12
            ) > 0.0,
            "stochastic": bool(stochastic.get("passed", False)),
        }
        qualified = all(checks.values())
        models = self._current_models(rows)
        config_contract = {
            "horizons": self.horizons,
            "minimum_train": self.minimum_train,
            "minimum_probability": self.minimum_probability,
            "minimum_expected_normal": self.minimum_expected_normal,
            "minimum_expected_stressed": self.minimum_expected_stressed,
            "minimum_mfe_cost_multiple": self.minimum_mfe_cost_multiple,
            "minimum_excursion_ratio": self.minimum_excursion_ratio,
            "ridge": self.ridge,
            "maximum_factors": self.maximum_factors,
            "features": SELECTOR_FEATURES,
        }
        payload = {
            "schema_version": self.SCHEMA,
            "generated_at": _now(),
            "status": "QUALIFIED" if qualified else "COLLECTING",
            "qualified": qualified,
            "observations": int(walk.get("total_observations") or 0),
            "horizons_hours": list(self.horizons),
            "oos_validation": walk,
            "bayesian": posterior,
            "stochastic_validation": stochastic,
            "checks": checks,
            "models": models,
            "model_contract_hash": hashlib.sha256(
                json.dumps(
                    config_contract,
                    sort_keys=True,
                    default=str,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "abstention_is_first_class": True,
            "full_costs_always_deducted": True,
            "delayed_label_safe": True,
            "hyperparameter_selection_on_oos": False,
            "paper_shadow_influence": qualified and self.mode != "live",
            "live_decision_influence": False,
            "automatic_live_promotion": False,
            "reason_codes": [
                name.upper() + "_FAILED"
                for name, passed in checks.items()
                if not passed
            ],
        }
        self._persist(payload)
        return payload

    def _persist(self, payload: dict[str, Any]) -> None:
        self.latest_path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(payload, sort_keys=True, default=str) + "\n"
            )
        self._latest = dict(payload)
        self._last_refresh = time.time()

    def _read_latest(self) -> dict[str, Any]:
        try:
            value = json.loads(self.latest_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def evaluate_context(
        self,
        context: Mapping[str, Any],
        *,
        policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        active = dict(policy or self._latest or self._read_latest())
        if not bool(active.get("qualified", False)):
            return {
                "schema_version": "crypto_ai_swing_entry_selector_decision_v1",
                "status": str(active.get("status") or "COLLECTING"),
                "action": "ABSTAIN",
                "passes": False,
                "qualified_policy_required": True,
                "live_decision_influence": False,
                "authority": "ADVISORY_ONLY",
            }

        models = dict(active.get("models") or {})
        candidates = []
        normal_cost, _ = self.attribution._row_costs(context)
        for horizon in self.horizons:
            model = dict(models.get(str(horizon)) or {})
            decision = self._decision(
                model,
                context,
                normal_cost_bps=float(normal_cost),
            )
            if decision.get("passes"):
                candidates.append(decision)
        chosen = (
            max(candidates, key=lambda row: float(row.get("utility") or -1e12))
            if candidates
            else None
        )
        if chosen is None:
            return {
                "schema_version": "crypto_ai_swing_entry_selector_decision_v1",
                "status": "READY",
                "action": "ABSTAIN",
                "passes": False,
                "live_decision_influence": False,
                "authority": "ADVISORY_ONLY",
            }
        return {
            "schema_version": "crypto_ai_swing_entry_selector_decision_v1",
            **chosen,
            "live_decision_influence": False,
            "authority": "ADVISORY_ONLY",
        }

    @staticmethod
    def policy_summary(
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        oos = dict(payload.get("oos_validation") or {})
        normal = dict(oos.get("normal_net") or {})
        stressed = dict(oos.get("stressed_net") or {})
        return {
            "status": payload.get("status", "COLLECTING"),
            "qualified": bool(payload.get("qualified", False)),
            "observations": payload.get("observations"),
            "oos_selected": oos.get("oos_selected"),
            "selection_fraction": oos.get("selection_fraction"),
            "abstention_fraction": oos.get("abstention_fraction"),
            "mean_normal_net_bps": normal.get("mean_bps"),
            "mean_stressed_net_bps": stressed.get("mean_bps"),
            "selected_horizon_counts": oos.get("selected_horizon_counts"),
            "reason_codes": payload.get("reason_codes"),
            "paper_shadow_influence": bool(
                payload.get("paper_shadow_influence", False)
            ),
            "live_decision_influence": False,
        }
