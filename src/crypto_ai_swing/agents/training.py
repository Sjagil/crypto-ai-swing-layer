from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_pinball_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crypto_ai_swing.agents.dataset import build_agent_dataset, purged_chronological_split
from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.universe.runtime import UniverseManager


@dataclass(frozen=True)
class TrainingResult:
    status: str
    artifact_path: Path
    pointer_path: Path
    metrics: dict[str, Any]
    dataset_id: str
    row_count: int
    markets: tuple[str, ...]
    horizon_bars: int


def _hist_classifier() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=350,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=2.0,
            random_state=17,
        )),
    ])


def _logistic_classifier() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=0.35,
            max_iter=2000,
            class_weight="balanced",
            random_state=17,
        )),
    ])


def _extra_trees_classifier() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=17,
        )),
    ])


def _regressor(*, quantile: float | None = None) -> Pipeline:
    kwargs: dict[str, Any] = dict(
        learning_rate=0.04,
        max_iter=350,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=2.0,
        random_state=17,
    )
    if quantile is not None:
        kwargs.update(loss="quantile", quantile=float(quantile))
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(**kwargs)),
    ])


def _auc(y: pd.Series, p: np.ndarray) -> float | None:
    return float(roc_auc_score(y, p)) if y.nunique() >= 2 else None


def _finite_mean(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else None


def _conservative_mean(values: np.ndarray, *, z: float = 1.0) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean
    se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    return mean - z * se


def _selected_economics(
    selected: np.ndarray,
    realized: np.ndarray,
    markets: np.ndarray,
    *,
    cost_floor: float,
) -> dict[str, Any]:
    mask = np.asarray(selected, dtype=bool)
    if not mask.any():
        return {
            "count": 0,
            "market_count": 0,
            "mean_return": None,
            "mean_net": None,
            "conservative_mean_net": None,
            "market_balanced_mean_net": None,
            "positive_market_fraction": None,
        }
    selected_returns = np.asarray(realized, dtype=float)[mask]
    selected_net = selected_returns - float(cost_floor)
    selected_markets = np.asarray(markets, dtype=object)[mask]
    by_market: list[float] = []
    for market in sorted({str(x) for x in selected_markets}):
        market_net = selected_net[selected_markets == market]
        if len(market_net):
            by_market.append(float(np.mean(market_net)))
    return {
        "count": int(mask.sum()),
        "market_count": len(by_market),
        "mean_return": float(selected_returns.mean()),
        "mean_net": float(selected_net.mean()),
        "conservative_mean_net": _conservative_mean(selected_net),
        "market_balanced_mean_net": float(np.mean(by_market)) if by_market else None,
        "positive_market_fraction": (
            float(np.mean(np.asarray(by_market) > 0.0)) if by_market else None
        ),
    }


def _threshold_plan(
    probability: np.ndarray,
    realized: np.ndarray,
    markets: np.ndarray,
    *,
    cost_floor: float,
    minimum_selected: int = 40,
    minimum_markets: int = 5,
) -> dict[str, Any]:
    plans: list[dict[str, Any]] = []
    for threshold in np.arange(0.50, 0.751, 0.025):
        threshold = round(float(threshold), 3)
        econ = _selected_economics(
            probability >= threshold,
            realized,
            markets,
            cost_floor=cost_floor,
        )
        eligible = bool(
            econ["count"] >= minimum_selected
            and econ["market_count"] >= minimum_markets
            and econ["mean_net"] is not None
            and float(econ["mean_net"]) > 0.0
            and econ["market_balanced_mean_net"] is not None
            and float(econ["market_balanced_mean_net"]) > 0.0
        )
        plans.append({"threshold": threshold, **econ, "economically_eligible": eligible})

    eligible = [row for row in plans if row["economically_eligible"]]
    if eligible:
        best = max(
            eligible,
            key=lambda row: (
                float(row.get("conservative_mean_net") or -999.0),
                float(row.get("market_balanced_mean_net") or -999.0),
                float(row.get("mean_net") or -999.0),
                int(row.get("count") or 0),
            ),
        )
    else:
        sufficiently_sampled = [
            row for row in plans
            if row["count"] >= max(20, minimum_selected // 2)
            and row["market_count"] >= max(3, minimum_markets // 2)
        ]
        pool = sufficiently_sampled or plans
        best = max(
            pool,
            key=lambda row: (
                float(row.get("mean_net") or -999.0),
                int(row.get("market_count") or 0),
                int(row.get("count") or 0),
            ),
        )
    return {
        "chosen": best,
        "grid": plans,
        "minimum_selected": minimum_selected,
        "minimum_markets": minimum_markets,
    }


def _direction_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    if not mask.any():
        return 0.0
    return float(np.mean((a[mask] > 0.0) == (p[mask] > 0.0)))


def _regression_skill(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    model_mae = float(mean_absolute_error(a, p))
    zero_mae = float(mean_absolute_error(a, np.zeros_like(a)))
    skill = 1.0 - model_mae / max(zero_mae, 1e-12)
    return {
        "mae": model_mae,
        "zero_baseline_mae": zero_mae,
        "skill_vs_zero": float(skill),
        "direction_accuracy": _direction_accuracy(a, p),
    }


def _risk_skill(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    baseline_quantile: float,
    quantile: float = 0.75,
) -> dict[str, float]:
    a = np.asarray(actual, dtype=float)
    p = np.clip(np.asarray(predicted, dtype=float), 0.0, None)
    loss = float(mean_pinball_loss(a, p, alpha=quantile))
    baseline = np.full_like(a, float(baseline_quantile))
    baseline_loss = float(mean_pinball_loss(a, baseline, alpha=quantile))
    coverage = float(np.mean(a <= p)) if len(a) else 0.0
    return {
        "pinball_loss": loss,
        "baseline_pinball_loss": baseline_loss,
        "skill_vs_constant_quantile": float(1.0 - loss / max(baseline_loss, 1e-12)),
        "coverage": coverage,
        "mae": float(mean_absolute_error(a, p)),
    }


class AgentTrainer:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.crypto = CryptoLibraryBridge(settings.crypto_repo_root)
        self.universe = UniverseManager(settings)

    def train(
        self,
        *,
        markets: list[str] | None = None,
        timeframe: str = "1h",
        horizon_bars: int = 4,
        minimum_rows: int = 1200,
        minimum_net_move_bps: float = 65.0,
    ) -> TrainingResult:
        selected_markets = [str(x).upper() for x in (markets or []) if str(x).strip()]
        if not selected_markets:
            selected_markets = list(self.universe.current()["markets"])
        frames = self.crypto.ohlcv_many(
            selected_markets,
            timeframe,
            persist=False,
            concurrency=4,
        )
        frames = {
            market: frame
            for market, frame in frames.items()
            if frame is not None and not frame.empty
        }
        dataset = build_agent_dataset(
            frames,
            horizon_bars=horizon_bars,
            minimum_net_move_bps=minimum_net_move_bps,
        )
        if len(dataset.frame) < int(minimum_rows):
            raise ValueError(
                f"insufficient causal rows: {len(dataset.frame)} < {minimum_rows}"
            )
        train, validation, test = purged_chronological_split(dataset)
        features = dataset.feature_columns
        x_train = train.loc[:, features]
        y_train = train["target_alpha"].astype(int)
        if y_train.nunique() < 2:
            raise ValueError("alpha label has one class")
        if train["target_regime_persistence"].nunique() < 2:
            raise ValueError("regime label has one class")

        quality_cfg = dict(getattr(self.settings, "agents", {}) or {}).get(
            "qualification", {}
        ) or {}
        min_val_selected = int(quality_cfg.get("minimum_validation_selected", 40))
        min_test_selected = int(quality_cfg.get("minimum_test_selected", 60))
        min_val_markets = int(quality_cfg.get("minimum_validation_markets", 5))
        min_test_markets = int(quality_cfg.get("minimum_test_markets", 6))
        min_market_positive_fraction = float(
            quality_cfg.get("minimum_test_positive_market_fraction", 0.50)
        )

        x_val = validation.loc[:, features]
        y_val = validation["target_alpha"].astype(int)
        val_realized = validation["target_forward_return"].to_numpy(float)
        val_markets = validation["market"].astype(str).to_numpy()
        cost_floor = float(minimum_net_move_bps) / 10_000.0
        candidates = {
            "hist_gradient_boosting": _hist_classifier(),
            "logistic_balanced": _logistic_classifier(),
            "extra_trees": _extra_trees_classifier(),
        }
        tournament: list[dict[str, Any]] = []
        fitted: dict[str, Pipeline] = {}
        for name, model in candidates.items():
            model.fit(x_train, y_train)
            probability = model.predict_proba(x_val)[:, 1]
            threshold_plan = _threshold_plan(
                probability,
                val_realized,
                val_markets,
                cost_floor=cost_floor,
                minimum_selected=min_val_selected,
                minimum_markets=min_val_markets,
            )
            chosen = dict(threshold_plan["chosen"])
            auc = _auc(y_val, probability)
            brier = float(brier_score_loss(y_val, probability))
            tournament.append({
                "model": name,
                "validation_auc": auc,
                "validation_brier": brier,
                "threshold_plan": threshold_plan,
                "validation_selected_count": chosen["count"],
                "validation_selected_market_count": chosen["market_count"],
                "validation_selected_mean_net": chosen["mean_net"],
                "validation_market_balanced_mean_net": chosen["market_balanced_mean_net"],
                "validation_positive_market_fraction": chosen["positive_market_fraction"],
                "validation_conservative_mean_net": chosen["conservative_mean_net"],
                "validation_economically_eligible": chosen["economically_eligible"],
            })
            fitted[name] = model

        def rank(row: dict[str, Any]):
            return (
                1 if row.get("validation_economically_eligible") else 0,
                float(row.get("validation_conservative_mean_net") or -999.0),
                float(row.get("validation_market_balanced_mean_net") or -999.0),
                float(row.get("validation_selected_mean_net") or -999.0),
                float(row.get("validation_auc") or 0.0),
                -float(row.get("validation_brier") or 999.0),
            )

        winner_row = max(tournament, key=rank)
        winner_name = str(winner_row["model"])
        alpha = fitted[winner_name]
        threshold = float(winner_row["threshold_plan"]["chosen"]["threshold"])

        regime = _hist_classifier().fit(
            x_train, train["target_regime_persistence"].astype(int)
        )
        ret = _regressor().fit(x_train, train["target_forward_return"])
        risk_quantile = 0.75
        risk = _regressor(quantile=risk_quantile).fit(x_train, train["target_mae"])

        x_test = test.loc[:, features]
        alpha_p = alpha.predict_proba(x_test)[:, 1]
        test_selected = alpha_p >= threshold
        realized = test["target_forward_return"].to_numpy(float)
        test_economics = _selected_economics(
            test_selected,
            realized,
            test["market"].astype(str).to_numpy(),
            cost_floor=cost_floor,
        )
        test_auc = _auc(test["target_alpha"].astype(int), alpha_p)
        val_auc = winner_row.get("validation_auc")
        positive_oos_net = bool(
            test_economics["count"] >= min_test_selected
            and test_economics["mean_net"] is not None
            and float(test_economics["mean_net"]) > 0.0
        )
        alpha_qualified = bool(
            positive_oos_net
            and bool(winner_row.get("validation_economically_eligible"))
            and int(winner_row.get("validation_selected_count") or 0) >= min_val_selected
            and int(winner_row.get("validation_selected_market_count") or 0) >= min_val_markets
            and test_economics["count"] >= min_test_selected
            and test_economics["market_count"] >= min_test_markets
            and test_economics["market_balanced_mean_net"] is not None
            and float(test_economics["market_balanced_mean_net"]) > 0.0
            and test_economics["positive_market_fraction"] is not None
            and float(test_economics["positive_market_fraction"])
                >= min_market_positive_fraction
            and val_auc is not None
            and float(val_auc) >= float(quality_cfg.get("minimum_validation_alpha_auc", 0.52))
            and test_auc is not None
            and float(test_auc) >= float(quality_cfg.get("minimum_test_alpha_auc", 0.50))
        )

        regime_val_p = regime.predict_proba(x_val)[:, 1]
        regime_test_p = regime.predict_proba(x_test)[:, 1]
        regime_val_auc = _auc(
            validation["target_regime_persistence"].astype(int), regime_val_p
        )
        regime_test_auc = _auc(
            test["target_regime_persistence"].astype(int), regime_test_p
        )
        regime_qualified = bool(
            regime_val_auc is not None
            and regime_test_auc is not None
            and regime_val_auc >= float(quality_cfg.get("minimum_validation_regime_auc", 0.52))
            and regime_test_auc >= float(quality_cfg.get("minimum_test_regime_auc", 0.50))
        )

        ret_val_p = ret.predict(x_val)
        ret_test_p = ret.predict(x_test)
        return_validation = _regression_skill(val_realized, ret_val_p)
        return_test = _regression_skill(realized, ret_test_p)
        return_qualified = bool(
            return_validation["skill_vs_zero"]
                >= float(quality_cfg.get("minimum_validation_return_skill", 0.02))
            and return_test["skill_vs_zero"]
                >= float(quality_cfg.get("minimum_test_return_skill", 0.0))
            and return_validation["direction_accuracy"] >= 0.52
            and return_test["direction_accuracy"] >= 0.50
        )

        baseline_quantile = float(
            np.quantile(train["target_mae"].to_numpy(float), risk_quantile)
        )
        risk_val_p = np.clip(risk.predict(x_val), 0.0, None)
        risk_test_p = np.clip(risk.predict(x_test), 0.0, None)
        risk_validation = _risk_skill(
            validation["target_mae"].to_numpy(float),
            risk_val_p,
            baseline_quantile=baseline_quantile,
            quantile=risk_quantile,
        )
        risk_test = _risk_skill(
            test["target_mae"].to_numpy(float),
            risk_test_p,
            baseline_quantile=baseline_quantile,
            quantile=risk_quantile,
        )
        risk_qualified = bool(
            risk_validation["skill_vs_constant_quantile"] > 0.0
            and risk_test["skill_vs_constant_quantile"] > 0.0
            and 0.65 <= risk_validation["coverage"] <= 0.85
            and 0.65 <= risk_test["coverage"] <= 0.85
        )

        head_qualifications = {
            "alpha": alpha_qualified,
            "regime": regime_qualified,
            "return": return_qualified,
            "risk": risk_qualified,
            "execution": True,
        }
        directional_qualified = bool(alpha_qualified or return_qualified)

        metrics = {
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "market_count": len(dataset.markets),
            "alpha_model": winner_name,
            "alpha_probability_threshold": threshold,
            "alpha_auc": test_auc,
            "validation_alpha_auc": val_auc,
            "alpha_brier": float(
                brier_score_loss(test["target_alpha"].astype(int), alpha_p)
            ),
            "alpha_accuracy_at_0_5": float(
                accuracy_score(test["target_alpha"].astype(int), alpha_p >= 0.5)
            ),
            "selected_count": int(test_economics["count"]),
            "selected_market_count": int(test_economics["market_count"]),
            "selected_mean_forward_return": test_economics["mean_return"],
            "selected_mean_net_after_cost_floor": test_economics["mean_net"],
            "selected_market_balanced_mean_net": test_economics["market_balanced_mean_net"],
            "selected_positive_market_fraction": test_economics["positive_market_fraction"],
            "selected_conservative_mean_net": test_economics["conservative_mean_net"],
            "positive_oos_net_proxy": positive_oos_net,
            "regime_auc": regime_test_auc,
            "validation_regime_auc": regime_val_auc,
            "return_mae": return_test["mae"],
            "return_validation": return_validation,
            "return_test": return_test,
            "risk_mae": risk_test["mae"],
            "risk_validation": risk_validation,
            "risk_test": risk_test,
            "head_qualifications": head_qualifications,
            "shadow_decision_qualified": directional_qualified,
            "model_tournament": tournament,
            "chronological": True,
            "purged": True,
            "purge_bars": int(dataset.horizon_bars),
            "point_in_time_features": True,
            "future_only_labels": True,
            "current_universe_selection_not_point_in_time": True,
            "automatic_live_promotion": False,
        }

        trained_at = datetime.now(timezone.utc)
        payload = {
            "schema_version": "swing_agent_bundle_v3",
            "status": "SHADOW",
            "shadow_decision_qualified": directional_qualified,
            "head_qualifications": head_qualifications,
            "live_decision_influence": False,
            "dataset_id": dataset.dataset_id,
            "dataset_rows": int(len(dataset.frame)),
            "feature_columns": list(features),
            "markets": list(dataset.markets),
            "timeframe": timeframe,
            "horizon_bars": int(horizon_bars),
            "minimum_net_move_bps": float(minimum_net_move_bps),
            "alpha_probability_threshold": threshold,
            "trained_at": trained_at.isoformat(),
            "expires_at": (trained_at + timedelta(days=30)).isoformat(),
            "train_range": [
                pd.Timestamp(train["feature_time"].min()).isoformat(),
                pd.Timestamp(train["feature_time"].max()).isoformat(),
            ],
            "validation_range": [
                pd.Timestamp(validation["feature_time"].min()).isoformat(),
                pd.Timestamp(validation["feature_time"].max()).isoformat(),
            ],
            "test_range": [
                pd.Timestamp(test["feature_time"].min()).isoformat(),
                pd.Timestamp(test["feature_time"].max()).isoformat(),
            ],
            "metrics": metrics,
            "models": {"alpha": alpha, "regime": regime, "return": ret, "risk": risk},
        }
        root = self.settings.project_root / "output/crypto_ai_swing/agents"
        identity = sha256(json.dumps(
            {k: v for k, v in payload.items() if k != "models"},
            sort_keys=True,
            default=str,
        ).encode()).hexdigest()
        artdir = root / "artifacts" / identity
        artdir.mkdir(parents=True, exist_ok=True)
        artifact = artdir / "bundle.joblib"
        joblib.dump(payload, artifact)
        artifact_hash = sha256(artifact.read_bytes()).hexdigest()
        manifest = {k: v for k, v in payload.items() if k != "models"}
        manifest.update(
            artifact_hash=artifact_hash,
            artifact_path=str(artifact.resolve()),
            model_names=sorted(payload["models"]),
        )
        manifest_path = artdir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        pointer = root / "latest.pointer.json"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(json.dumps({
            "schema_version": "swing_agent_pointer_v3",
            "artifact_path": str(artifact.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "artifact_hash": artifact_hash,
            "status": "SHADOW",
            "shadow_decision_qualified": directional_qualified,
            "head_qualifications": head_qualifications,
            "live_decision_influence": False,
            "updated_at": trained_at.isoformat(),
        }, indent=2), encoding="utf-8")
        return TrainingResult(
            "SHADOW",
            artifact,
            pointer,
            metrics,
            dataset.dataset_id,
            len(dataset.frame),
            dataset.markets,
            horizon_bars,
        )
