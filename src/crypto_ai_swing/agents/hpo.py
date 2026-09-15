from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_pinball_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crypto_ai_swing.agents.canonical_features import select_train_only_features
from crypto_ai_swing.agents.feature_denoising import (
    NOISE_CONTROL_VERSION,
    feature_group_counts,
)
from crypto_ai_swing.agents.dataset import build_agent_dataset, purged_chronological_split
from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.universe.runtime import UniverseManager

SCHEMA = "crypto_ai_swing_hpo_v1"
HPO_SEARCH_SPACE_VERSION = "round47a2_v1"


def _now() -> datetime:
    return datetime.now(UTC)


def _safe_auc(y, p) -> float:
    y = pd.Series(y).astype(int)
    if y.nunique() < 2:
        return 0.5
    return float(roc_auc_score(y, np.asarray(p, dtype=float)))


def _classification_score(y, p) -> float:
    probability = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    auc = _safe_auc(y, probability)
    brier = float(brier_score_loss(pd.Series(y).astype(int), probability))
    return float((auc - 0.5) * 2.0 - 0.35 * brier)


def _economic_sample_weight(
    frame: pd.DataFrame,
    *,
    cost_floor: float,
) -> np.ndarray:
    realized = frame["target_forward_return"].to_numpy(float)
    mae = frame["target_mae"].to_numpy(float)
    mfe = frame["target_mfe"].to_numpy(float)
    net = realized - float(cost_floor)

    scale = np.clip(
        np.abs(net) / max(float(cost_floor), 1e-4),
        0.0,
        3.0,
    )
    path_quality = np.clip(
        (mfe - mae) / (mfe + mae + 1e-9),
        0.0,
        1.0,
    )

    return np.clip(
        1.0
        + 0.35 * np.sqrt(scale)
        + 0.25 * path_quality * (net > 0.0),
        0.75,
        2.50,
    )


def _fit_weighted(model, x, y, weights):
    weights = np.asarray(weights, dtype=float)

    if hasattr(model, "steps") and getattr(model, "steps", None):
        step_name = model.steps[-1][0]
        try:
            return model.fit(
                x,
                y,
                **{f"{step_name}__sample_weight": weights},
            )
        except (TypeError, ValueError):
            pass

    try:
        return model.fit(
            x,
            y,
            sample_weight=weights,
        )
    except (TypeError, ValueError):
        return model.fit(x, y)


def _economic_alpha_score(
    y,
    probability,
    realized,
    markets,
    *,
    cost_floor: float,
    minimum_selected: int = 40,
    minimum_markets: int = 5,
) -> float:
    p = np.clip(
        np.asarray(probability, dtype=float),
        1e-6,
        1.0 - 1e-6,
    )
    r = np.asarray(realized, dtype=float)
    m = np.asarray(markets, dtype=object)

    auc = _safe_auc(y, p)
    brier = float(
        brier_score_loss(
            pd.Series(y).astype(int),
            p,
        )
    )

    best = -10.0

    for threshold in np.arange(0.50, 0.901, 0.025):
        mask = p >= float(threshold)

        if int(mask.sum()) < int(minimum_selected):
            continue

        selected_markets = m[mask]
        market_names = sorted(
            {str(value) for value in selected_markets}
        )

        if len(market_names) < int(minimum_markets):
            continue

        net = r[mask] - float(cost_floor)

        if not len(net):
            continue

        mean_net = float(np.mean(net))
        conservative = mean_net

        if len(net) > 1:
            conservative -= float(
                np.std(net, ddof=1)
                / np.sqrt(len(net))
            )

        market_means = np.asarray(
            [
                float(
                    np.mean(
                        net[selected_markets == market]
                    )
                )
                for market in market_names
            ],
            dtype=float,
        )

        balanced = float(np.mean(market_means))
        positive_market_fraction = float(
            np.mean(market_means > 0.0)
        )

        score = (
            0.40 * np.tanh(conservative * 10_000.0 / 50.0)
            + 0.25 * np.tanh(balanced * 10_000.0 / 50.0)
            + 0.15 * (2.0 * positive_market_fraction - 1.0)
            + 0.15 * ((auc - 0.5) * 2.0)
            - 0.05 * brier
        )

        best = max(best, float(score))

    if best <= -9.0:
        return float(
            -1.0
            + 0.20 * ((auc - 0.5) * 2.0)
            - 0.05 * brier
        )

    return float(best)


def _return_score(actual, predicted) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    model_mae = float(mean_absolute_error(a, p))
    baseline_mae = float(mean_absolute_error(a, np.zeros_like(a)))
    skill = 1.0 - model_mae / max(baseline_mae, 1e-12)
    direction = float(np.mean((a > 0.0) == (p > 0.0)))
    rank_ic = pd.Series(a).corr(
        pd.Series(p),
        method="spearman",
    )
    if rank_ic is None or not np.isfinite(rank_ic):
        rank_ic = 0.0
    return float(
        skill
        + 0.30 * (direction - 0.5)
        + 0.25 * float(rank_ic)
    )


def _risk_score(actual, predicted, *, quantile: float = 0.75) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.clip(np.asarray(predicted, dtype=float), 0.0, None)
    baseline_value = float(np.quantile(a, quantile))
    baseline = np.full_like(a, baseline_value)
    loss = float(mean_pinball_loss(a, p, alpha=quantile))
    baseline_loss = float(mean_pinball_loss(a, baseline, alpha=quantile))
    skill = 1.0 - loss / max(baseline_loss, 1e-12)
    coverage = float(np.mean(a <= p))
    return float(skill - 0.50 * abs(coverage - quantile))


def _classifier(spec: dict[str, Any], *, seed: int = 43):
    family = str(spec.get("family") or "hist_gradient_boosting")
    p = dict(spec.get("params") or {})

    if family == "hist_gradient_boosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=float(p.get("learning_rate", 0.04)),
                max_iter=int(p.get("max_iter", 350)),
                max_leaf_nodes=int(p.get("max_leaf_nodes", 15)),
                min_samples_leaf=int(p.get("min_samples_leaf", 30)),
                l2_regularization=float(p.get("l2_regularization", 2.0)),
                random_state=seed,
            )),
        ])
    if family == "extra_trees":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=int(p.get("n_estimators", 500)),
                max_depth=None if p.get("max_depth") in (None, "None") else int(p["max_depth"]),
                min_samples_leaf=int(p.get("min_samples_leaf", 12)),
                max_features=p.get("max_features", "sqrt"),
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            )),
        ])
    if family == "logistic":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=float(p.get("C", 0.35)),
                max_iter=3000,
                class_weight="balanced",
                random_state=seed,
            )),
        ])
    if family == "mlp":
        hidden = tuple(int(v) for v in p.get("hidden_layer_sizes", [96, 48]))
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=hidden,
                activation=str(p.get("activation", "relu")),
                solver="adam",
                alpha=float(p.get("alpha", 1e-4)),
                batch_size=int(p.get("batch_size", 128)),
                learning_rate_init=float(p.get("learning_rate_init", 5e-4)),
                max_iter=int(p.get("max_iter", 350)),
                early_stopping=False,
                shuffle=False,
                random_state=seed,
            )),
        ])
    raise ValueError(f"unsupported classifier family: {family}")


def _regressor(spec: dict[str, Any], *, seed: int = 43, quantile: float | None = None):
    family = str(spec.get("family") or "hist_gradient_boosting")
    p = dict(spec.get("params") or {})
    if family == "hist_gradient_boosting":
        kwargs = dict(
            learning_rate=float(p.get("learning_rate", 0.04)),
            max_iter=int(p.get("max_iter", 350)),
            max_leaf_nodes=int(p.get("max_leaf_nodes", 15)),
            min_samples_leaf=int(p.get("min_samples_leaf", 30)),
            l2_regularization=float(p.get("l2_regularization", 2.0)),
            random_state=seed,
        )
        if quantile is not None:
            kwargs.update(loss="quantile", quantile=float(quantile))
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(**kwargs)),
        ])
    if family == "extra_trees" and quantile is None:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=int(p.get("n_estimators", 500)),
                max_depth=None if p.get("max_depth") in (None, "None") else int(p["max_depth"]),
                min_samples_leaf=int(p.get("min_samples_leaf", 10)),
                max_features=p.get("max_features", 1.0),
                n_jobs=-1,
                random_state=seed,
            )),
        ])
    if family == "mlp" and quantile is None:
        hidden = tuple(int(v) for v in p.get("hidden_layer_sizes", [96, 48]))
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=hidden,
                activation=str(p.get("activation", "relu")),
                solver="adam",
                alpha=float(p.get("alpha", 1e-4)),
                batch_size=int(p.get("batch_size", 128)),
                learning_rate_init=float(p.get("learning_rate_init", 5e-4)),
                max_iter=int(p.get("max_iter", 350)),
                early_stopping=False,
                shuffle=False,
                random_state=seed,
            )),
        ])
    return _regressor({"family": "hist_gradient_boosting", "params": p}, seed=seed, quantile=quantile)


def _storage(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.resolve()}"


def _best_spec(
    study,
    *,
    default_family: str | None = None,
) -> dict[str, Any]:
    trial = study.best_trial
    params = dict(trial.params)

    family = params.pop("family", None)

    if not family:
        family = dict(
            getattr(trial, "user_attrs", {}) or {}
        ).get("family")

    if not family:
        family = default_family

    if not family:
        if "C" in params:
            family = "logistic"
        elif "hidden_1" in params:
            family = "mlp"
        elif "n_estimators" in params:
            family = "extra_trees"
        else:
            family = "hist_gradient_boosting"

    family = str(family)

    if "hidden_1" in params:
        hidden = [int(params.pop("hidden_1"))]
        hidden_2 = int(params.pop("hidden_2", 0))
        if hidden_2 > 0:
            hidden.append(hidden_2)
        params["hidden_layer_sizes"] = hidden

    return {
        "family": family,
        "params": params,
        "objective": float(study.best_value),
        "trial_number": int(trial.number),
        "study_name": study.study_name,
        "search_space_version": HPO_SEARCH_SPACE_VERSION,
    }


@dataclass(frozen=True)
class HPORunResult:
    status: str
    state_path: Path
    dataset_id: str | None
    payload: dict[str, Any]


class HPOService:
    """Optuna HPO. It only sees train/validation; the final test stays untouched."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self.cfg = dict(settings.agents.get("hpo", {}) or {})
        self.crypto = CryptoLibraryBridge(settings.crypto_repo_root)
        self.universe = UniverseManager(settings)
        self.root = settings.project_root / "output/crypto_ai_swing/hpo"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "best.json"
        self.storage_path = self.root / "studies.sqlite3"

    def _optuna(self):
        try:
            import optuna
        except Exception as exc:
            raise RuntimeError("Optuna missing. Install with: pip install -e '.[research]'") from exc
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return optuna

    def _study(self, head: str, dataset_id: str):
        optuna = self._optuna()

        identity = hashlib.sha256(
            (
                HPO_SEARCH_SPACE_VERSION
                + "|"
                + str(head)
                + "|"
                + str(dataset_id)
            ).encode("utf-8")
        ).hexdigest()[:16]

        study_name = (
            HPO_SEARCH_SPACE_VERSION
            + "_"
            + str(head)
            + "_"
            + identity
        )

        return optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=_storage(self.storage_path),
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(
                seed=int(self.cfg.get("seed", 43)),
                multivariate=True,
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5
            ),
        )

    @staticmethod
    def _classifier_spec(trial, *, allow_mlp: bool):
        families = ["hist_gradient_boosting", "extra_trees", "logistic"] + (["mlp"] if allow_mlp else [])
        family = trial.suggest_categorical("family", families)
        if family == "hist_gradient_boosting":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
                "max_iter": trial.suggest_int("max_iter", 180, 500, step=40),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 31, step=4),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 70, step=5),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.1, 8.0, log=True),
            }
        elif family == "extra_trees":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 250, 750, step=100),
                "max_depth": trial.suggest_int("max_depth", 5, 14),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 8, 36, step=4),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.7, 1.0]),
            }
        elif family == "logistic":
            params = {"C": trial.suggest_float("C", 0.03, 3.0, log=True)}
        else:
            params = {
                "hidden_1": trial.suggest_categorical("hidden_1", [48, 64, 96, 128, 160]),
                "hidden_2": trial.suggest_categorical("hidden_2", [0, 24, 32, 48, 64]),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 5e-5, 3e-3, log=True),
                "max_iter": trial.suggest_int("max_iter", 220, 460, step=40),
            }
        spec = {"family": family, "params": params}
        if family == "mlp":
            spec["params"]["hidden_layer_sizes"] = [int(params["hidden_1"])] + ([int(params["hidden_2"])] if int(params["hidden_2"]) > 0 else [])
        return spec

    @staticmethod
    def _return_spec(trial, *, allow_mlp: bool):
        families = ["hist_gradient_boosting", "extra_trees"] + (["mlp"] if allow_mlp else [])
        family = trial.suggest_categorical("family", families)
        if family == "hist_gradient_boosting":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
                "max_iter": trial.suggest_int("max_iter", 180, 500, step=40),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 31, step=4),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 70, step=5),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.1, 8.0, log=True),
            }
        elif family == "extra_trees":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 250, 750, step=100),
                "max_depth": trial.suggest_int("max_depth", 5, 14),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 6, 30, step=3),
                "max_features": trial.suggest_categorical("max_features", [0.5, 0.7, 1.0]),
            }
        else:
            params = {
                "hidden_1": trial.suggest_categorical("hidden_1", [48, 64, 96, 128, 160]),
                "hidden_2": trial.suggest_categorical("hidden_2", [0, 24, 32, 48, 64]),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 5e-5, 3e-3, log=True),
                "max_iter": trial.suggest_int("max_iter", 220, 460, step=40),
            }
        spec = {"family": family, "params": params}
        if family == "mlp":
            spec["params"]["hidden_layer_sizes"] = [int(params["hidden_1"])] + ([int(params["hidden_2"])] if int(params["hidden_2"]) > 0 else [])
        return spec

    @staticmethod
    def _risk_spec(trial):
        return {
            "family": "hist_gradient_boosting",
            "params": {
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.10, log=True),
                "max_iter": trial.suggest_int("max_iter", 180, 500, step=40),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 27, step=4),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 70, step=5),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.1, 8.0, log=True),
            },
        }

    def run_supervised(self, *, markets=None, timeframe="1h", horizon_bars=4, minimum_rows=8000) -> HPORunResult:
        selected = [str(v).upper() for v in (markets or []) if str(v).strip()] or list(self.universe.current().get("markets", []))
        frames = self.crypto.ohlcv_many(selected, timeframe, persist=False, concurrency=int(self.cfg.get("fetch_concurrency", 4)))
        frames = {m: f for m, f in frames.items() if f is not None and not f.empty}
        dataset = build_agent_dataset(
            frames,
            horizon_bars=int(horizon_bars),
            minimum_net_move_bps=float(self.settings.agents.get("minimum_net_move_bps", 65.0)),
            feature_columns=None,
            canonical_bridge=self.crypto,
        )
        if len(dataset.frame) < int(minimum_rows):
            raise ValueError(f"HPO insufficient causal rows: {len(dataset.frame)} < {minimum_rows}")
        train, validation, _ = purged_chronological_split(dataset)
        features = select_train_only_features(
            train, dataset.feature_columns, target=train["target_forward_return"], maximum_features=int(self.cfg.get("maximum_features", 96))
        )
        if len(features) < 8:
            raise ValueError(f"HPO selected only {len(features)} features")
        x_train, x_val = train.loc[:, features], validation.loc[:, features]
        allow_mlp = bool(self.cfg.get("backprop_mlp_enabled", True))
        trials = dict(self.cfg.get("trials_per_run", {}) or {})
        timeout = int(self.cfg.get("timeout_seconds_per_head", 900))
        heads = {}

        training_cfg = dict(self.settings.agents.get("training", {}) or {})

        cost_floor = float(training_cfg.get("minimum_net_move_bps", self.settings.agents.get("minimum_net_move_bps", 65.0))) / 10_000.0

        alpha_train_weights = _economic_sample_weight(train, cost_floor=cost_floor)

        return_low = float(train["target_forward_return"].quantile(0.005))

        return_high = float(train["target_forward_return"].quantile(0.995))

        robust_return_train = train["target_forward_return"].clip(return_low, return_high)


        def tune_classifier(head, target):
            study = self._study(head, dataset.dataset_id)
            y_train, y_val = train[target].astype(int), validation[target].astype(int)
            def objective(trial):
                model = _classifier(
                    self._classifier_spec(
                        trial,
                        allow_mlp=allow_mlp,
                    )
                )

                if head == "alpha":
                    _fit_weighted(
                        model,
                        x_train,
                        y_train,
                        alpha_train_weights,
                    )
                else:
                    model.fit(x_train, y_train)

                probability = model.predict_proba(x_val)[:, 1]

                if head == "alpha":
                    qualification = dict(
                        self.settings.agents.get(
                            "qualification",
                            {},
                        )
                        or {}
                    )
                    return _economic_alpha_score(
                        y_val,
                        probability,
                        validation[
                            "target_forward_return"
                        ].to_numpy(float),
                        validation[
                            "market"
                        ].astype(str).to_numpy(),
                        cost_floor=cost_floor,
                        minimum_selected=int(
                            qualification.get(
                                "minimum_validation_selected",
                                40,
                            )
                        ),
                        minimum_markets=int(
                            qualification.get(
                                "minimum_validation_markets",
                                5,
                            )
                        ),
                    )

                return _classification_score(
                    y_val,
                    probability,
                )
            study.optimize(objective, n_trials=int(trials.get(head, 12)), timeout=timeout, gc_after_trial=True, show_progress_bar=False)
            heads[head] = _best_spec(study)

        tune_classifier("alpha", "target_alpha")
        tune_classifier("regime", "target_regime_persistence")

        study = self._study("return", dataset.dataset_id)
        def return_objective(trial):
            model = _regressor(self._return_spec(trial, allow_mlp=allow_mlp))
            model.fit(x_train, robust_return_train)
            return _return_score(validation["target_forward_return"], model.predict(x_val))
        study.optimize(return_objective, n_trials=int(trials.get("return", 12)), timeout=timeout, gc_after_trial=True, show_progress_bar=False)
        heads["return"] = _best_spec(study)

        study = self._study("risk", dataset.dataset_id)
        def risk_objective(trial):
            model = _regressor(self._risk_spec(trial), quantile=0.75)
            model.fit(x_train, train["target_mae"])
            return _risk_score(validation["target_mae"], model.predict(x_val))
        study.optimize(risk_objective, n_trials=int(trials.get("risk", 10)), timeout=timeout, gc_after_trial=True, show_progress_bar=False)
        heads["risk"] = _best_spec(study)

        existing = {}
        if self.state_path.is_file():
            try:
                existing = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        payload = {
            **existing,
            "schema_version": SCHEMA,
            "generated_at": _now().isoformat(),
            "dataset_id": dataset.dataset_id,
            "dataset_rows": len(dataset.frame),
            "markets": list(dataset.markets),
            "timeframe": timeframe,
            "horizon_bars": int(horizon_bars),
            "feature_count": len(features),
            "feature_columns": list(features),
            "heads": heads,
            "backprop_mlp_enabled": allow_mlp,
            "final_test_touched_by_hpo": False,
            "hpo_scope": "TRAIN_VALIDATION_ONLY",
        }
        self._atomic(payload)
        return HPORunResult("READY", self.state_path, dataset.dataset_id, payload)

    def run_rl(self, *, markets=None, timeframe="1h", minimum_rows_per_market=900) -> HPORunResult:
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
            from crypto_ai_swing.agents.rl_multi_market import (
                RL_STATE_VERSION,
                _env,
                _metrics,
                _normalize,
                _prepare,
            )
        except Exception as exc:
            raise RuntimeError("RL HPO requires .[ai,research] extras") from exc

        selected = [str(v).upper() for v in (markets or []) if str(v).strip()] or list(self.universe.current().get("markets", []))
        frames = self.crypto.ohlcv_many(selected, timeframe, persist=False, concurrency=int(self.cfg.get("fetch_concurrency", 4)))
        raw = {}
        for market, frame in frames.items():
            if frame is None or frame.empty:
                continue
            x, r = _prepare(self.crypto, market, frame)
            if len(x) >= int(minimum_rows_per_market):
                raw[market] = (x, r)
        if len(raw) < 6:
            raise ValueError(f"RL HPO usable markets {len(raw)} < 6")

        train_raw, val_raw, signature = {}, {}, []
        for market, (x, r) in raw.items():
            n = len(x)
            a, b = int(n * 0.60), int(n * 0.80)
            if a < 400 or b - a < 150:
                continue
            train_raw[market] = (x.iloc[:a].copy(), r.iloc[:a].copy())
            val_raw[market] = (x.iloc[a:b].copy(), r.iloc[a:b].copy())
            signature.append((market, n, str(x.index.min()), str(x.index.max())))
        if len(train_raw) < 6:
            raise ValueError(f"RL HPO split-usable markets {len(train_raw)} < 6")

        selection_blocks = []
        for market in sorted(train_raw):
            x, r = train_raw[market]
            block = x.copy()
            block["__round47f1_target"] = pd.to_numeric(
                r.reindex(x.index),
                errors="coerce",
            )
            selection_blocks.append(block)

        selection_frame = pd.concat(
            selection_blocks,
            axis=0,
            sort=False,
        ).sort_index(kind="stable")
        train_target_full = selection_frame.pop(
            "__round47f1_target"
        )
        full = selection_frame

        features = select_train_only_features(
            full,
            tuple(full.columns),
            target=train_target_full,
            maximum_features=int(
                self.cfg.get(
                    "rl_maximum_features",
                    64,
                )
            ),
        )

        if len(features) < 8:
            raise ValueError(
                f"RL HPO selected only {len(features)} features"
            )

        train_raw = {
            m: (
                x.reindex(
                    columns=list(features)
                ),
                r,
            )
            for m, (x, r) in train_raw.items()
        }
        val_raw = {
            m: (
                x.reindex(
                    columns=list(features)
                ),
                r,
            )
            for m, (x, r) in val_raw.items()
        }
        matrix = pd.concat([x for x, _ in train_raw.values()], axis=0)
        mean, std = matrix.mean(axis=0), matrix.std(axis=0, ddof=0).replace(0.0, 1.0)
        train = {m: (_normalize(x, mean, std), r.copy()) for m, (x, r) in train_raw.items()}
        validation = {m: (_normalize(x, mean, std), r.copy()) for m, (x, r) in val_raw.items()}

        dataset_id = hashlib.sha256(
            json.dumps(
                {
                    "signature": signature,
                    "rl_state_version": RL_STATE_VERSION,
                    "noise_control_version": NOISE_CONTROL_VERSION,
                    "feature_columns": list(features),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        study = self._study("rl_ppo", dataset_id)
        trial_timesteps = int(self.cfg.get("rl_trial_timesteps", 15000))

        def objective(trial):
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 5e-5, 8e-4, log=True),
                "gamma": trial.suggest_float("gamma", 0.97, 0.9995),
                "gae_lambda": trial.suggest_float("gae_lambda", 0.85, 0.99),
                "ent_coef": trial.suggest_float("ent_coef", 1e-5, 0.02, log=True),
                "vf_coef": trial.suggest_float("vf_coef", 0.3, 0.8),
                "clip_range": trial.suggest_float("clip_range", 0.10, 0.30),
                "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 15]),
                "net_arch": trial.suggest_categorical(
                    "net_arch", ["64x64", "128x64", "128x128"]
                ),
                "n_steps": trial.suggest_categorical("n_steps", [256, 512, 1024]),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            }
            factories = [(lambda x=x.copy(), r=r.copy(): _env(x, r)) for x, r in (train[m] for m in sorted(train))]
            architecture = {
                "64x64": [64, 64],
                "128x64": [128, 64],
                "128x128": [128, 128],
            }.get(str(params.get("net_arch")), [64, 64])
            model = PPO(
                "MlpPolicy",
                DummyVecEnv(factories),
                policy_kwargs={"net_arch": architecture},
                learning_rate=float(params["learning_rate"]),
                n_steps=int(params["n_steps"]),
                batch_size=int(params["batch_size"]),
                gamma=float(params["gamma"]),
                gae_lambda=float(params["gae_lambda"]),
                ent_coef=float(params["ent_coef"]),
                vf_coef=float(params["vf_coef"]),
                clip_range=float(params["clip_range"]),
                n_epochs=int(params["n_epochs"]),
                max_grad_norm=0.5,
                seed=int(self.cfg.get("seed", 43)),
                verbose=0,
            )
            model.learn(total_timesteps=trial_timesteps)
            metrics = _metrics(model, validation)
            metrics.pop("portfolio_step_returns", None)
            return float(
                metrics["mean_return"]
                + 0.35 * metrics["mean_excess_vs_buy_hold"]
                + 0.20 * metrics["median_return"]
                - 0.50 * metrics["worst_maximum_drawdown"]
                - 0.20 * metrics.get("mean_turnover_rate", 0.0)
                - 0.10 * metrics.get("mean_invalid_action_rate", 0.0)
                - 0.10 * metrics.get("worst_trade_giveback", 0.0)
            )

        study.optimize(
            objective,
            n_trials=int(self.cfg.get("rl_trials_per_run", 8)),
            timeout=int(self.cfg.get("rl_timeout_seconds", 3600)),
            gc_after_trial=True,
            show_progress_bar=False,
        )
        existing = {}
        if self.state_path.is_file():
            try:
                existing = json.loads(self.state_path.read_text())
            except Exception:
                pass
        payload = {
            **existing,
            "schema_version": SCHEMA,
            "generated_at": _now().isoformat(),
            "rl": {
                "dataset_id": dataset_id,
                "timeframe": timeframe,
                "markets": sorted(train),
                "feature_columns": list(features),
                "feature_group_counts": feature_group_counts(features),
                "noise_control_version": NOISE_CONTROL_VERSION,
                "feature_selection_target": "TRAIN_NEXT_RETURN_ONLY",
                "params": dict(study.best_trial.params),
                "objective": float(study.best_value),
                "trial_number": int(study.best_trial.number),
                "study_name": study.study_name,
                "observation_state_version": RL_STATE_VERSION,
                "final_test_touched_by_hpo": False,
            },
        }
        self._atomic(payload)
        return HPORunResult("READY", self.state_path, dataset_id, payload)

    def _atomic(self, payload):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        tmp.replace(self.state_path)


class HPOModelFactory:
    def __init__(self, settings, *, timeframe="1h", horizon_bars=4):
        self.settings = settings
        self.path = settings.project_root / "output/crypto_ai_swing/hpo/best.json"
        self.state = self._load()
        max_age = float((settings.agents.get("hpo", {}) or {}).get("maximum_state_age_hours", 168))
        raw = self.state.get("generated_at")
        if raw:
            try:
                stamp = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=UTC)
                if _now() - stamp.astimezone(UTC) > timedelta(hours=max_age):
                    self.state = {}
            except Exception:
                self.state = {}

    def _load(self):
        try:
            value = json.loads(self.path.read_text())
            return dict(value) if isinstance(value, dict) else {}
        except Exception:
            return {}

    def summary(self):
        return {
            "enabled": bool(self.state),
            "state_path": str(self.path),
            "dataset_id": self.state.get("dataset_id"),
            "generated_at": self.state.get("generated_at"),
            "backprop_mlp_enabled": self.state.get("backprop_mlp_enabled"),
            "heads": {
                k: {"family": (v or {}).get("family"), "objective": (v or {}).get("objective")}
                for k, v in dict(self.state.get("heads") or {}).items()
            },
            "rl": {"objective": dict(self.state.get("rl") or {}).get("objective")},
        }

    def alpha_candidate(self):
        spec = dict(self.state.get("heads") or {}).get("alpha")
        return _classifier(spec) if spec else None

    def regime_model(self):
        spec = dict(self.state.get("heads") or {}).get("regime")
        return _classifier(spec) if spec else None

    def return_model(self):
        spec = dict(self.state.get("heads") or {}).get("return")
        return _regressor(spec) if spec else None

    def risk_model(self, *, quantile=0.75):
        spec = dict(self.state.get("heads") or {}).get("risk")
        return _regressor(spec, quantile=quantile) if spec else None

    def rl_params(self):
        return dict(dict(self.state.get("rl") or {}).get("params") or {})
