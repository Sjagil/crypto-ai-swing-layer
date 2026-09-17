
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline


SPLIT_PROVENANCE = {
    "repository": "Sjagil/stocks-quant-agent-final-2",
    "path": "src/stocks/rl/splits.py",
    "reference_commit": "9f8c996202baf0cd2cfd486bac6bb629c8bce600",
    "algorithm": "PURGED_CHRONOLOGICAL_TRAIN_VALIDATION_UNTOUCHED_TEST",
}
CANDIDATE_SEEDS = (17, 29, 43, 71)


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _load_rows(path: Path, horizon_hours: int) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            """
            SELECT s.observation_id,s.observed_at,s.market,s.side,s.blocked,
                   s.context,o.return_bps,o.mae_bps,o.mfe_bps
            FROM signal_observations s
            JOIN forward_outcomes_v2 o ON o.observation_id=s.observation_id
            WHERE o.horizon_hours=?
            ORDER BY s.observed_at,s.observation_id
            """,
            (int(horizon_hours),),
        ).fetchall()
    finally:
        conn.close()
    output: list[dict[str, Any]] = []
    for observation_id, observed_at, market, side, blocked, context_raw, ret, mae, mfe in rows:
        try:
            context = json.loads(context_raw)
            vector = (context.get("round44") or {}).get("learning_vector") or {}
        except Exception:
            continue
        selected: dict[str, float] = {}
        for key, value in vector.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                selected[str(key)] = number
        if not selected:
            continue
        output.append(
            {
                **selected,
                "_observation_id": str(observation_id),
                "_observed_at": str(observed_at),
                "_market": str(market),
                "_side": str(side),
                "_blocked": int(bool(blocked)),
                "_return_bps": float(ret),
                "_mae_bps": float(mae),
                "_mfe_bps": float(mfe),
            }
        )
    return pd.DataFrame(output)


def _dataset_hash(frame: pd.DataFrame, horizon_hours: int) -> str:
    if frame.empty:
        return "EMPTY"
    columns = sorted(str(column) for column in frame.columns)
    identity = frame.reindex(columns=columns).copy()
    raw = pd.util.hash_pandas_object(identity, index=False).to_numpy().tobytes()
    schema = "|".join(columns).encode("utf-8")
    return hashlib.sha256(str(horizon_hours).encode() + schema + raw).hexdigest()


def _purged_split(
    frame: pd.DataFrame,
    *,
    purge_buckets: int = 4,
    embargo_buckets: int = 4,
    final_test_fraction: float = 0.20,
) -> Split:
    times = pd.Index(pd.to_datetime(frame["_observed_at"], utc=True).unique()).sort_values()
    if len(times) < 100:
        raise ValueError("INSUFFICIENT_TIMESTAMPS_FOR_PURGED_SPLIT")
    test_size = max(8, int(len(times) * final_test_fraction))
    test_start = len(times) - test_size
    development_end = max(1, test_start - embargo_buckets)
    train_end = max(1, int(development_end * 0.75) - purge_buckets)
    validation_start = min(development_end, train_end + purge_buckets)
    train_times = times[:train_end]
    validation_times = times[validation_start:development_end]
    test_times = times[test_start:]
    observed = pd.to_datetime(frame["_observed_at"], utc=True)
    train = frame.loc[observed.isin(train_times)].copy()
    validation = frame.loc[observed.isin(validation_times)].copy()
    test = frame.loc[observed.isin(test_times)].copy()
    if min(len(train), len(validation), len(test)) < 20:
        raise ValueError("PURGED_SPLIT_EMPTY_OR_TOO_SMALL")
    return Split(train=train, validation=validation, test=test)


def _select_features(
    train: pd.DataFrame,
    *,
    maximum_features: int,
    minimum_coverage: float = 0.70,
) -> list[str]:
    features = [name for name in train.columns if not name.startswith("_")]
    target = pd.to_numeric(train["_return_bps"], errors="coerce")
    scored: list[tuple[str, float, float]] = []
    for name in features:
        values = pd.to_numeric(train[name], errors="coerce")
        coverage = float(values.notna().mean()) if len(values) else 0.0
        if coverage < minimum_coverage or values.nunique(dropna=True) < 3:
            continue
        valid = values.notna() & target.notna()
        association = 0.0
        if int(valid.sum()) >= 20:
            corr = values[valid].rank().corr(target[valid].rank())
            if pd.notna(corr):
                association = abs(float(corr))
        scored.append((name, association, coverage))
    scored.sort(key=lambda item: (-item[1], -item[2], item[0]))
    candidates = [name for name, _, _ in scored[: max(maximum_features * 3, maximum_features)]]
    if not candidates:
        return []
    numeric = train[candidates].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="spearman").abs()
    selected: list[str] = []
    for name in candidates:
        if any(
            pd.notna(corr.loc[name, other]) and float(corr.loc[name, other]) >= 0.95
            for other in selected
        ):
            continue
        selected.append(name)
        if len(selected) >= maximum_features:
            break
    return selected


def _classifier(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.04,
                    max_iter=300,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    l2_regularization=2.0,
                    random_state=seed,
                ),
            ),
        ]
    )


def _regressor(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.04,
                    max_iter=300,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    l2_regularization=2.0,
                    random_state=seed,
                ),
            ),
        ]
    )


def _auc(y: pd.Series, p: np.ndarray) -> float | None:
    return float(roc_auc_score(y, p)) if y.nunique() >= 2 else None


def _selection_economics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    threshold: float,
    normal_cost_bps: float,
    stress_extra_bps: float,
) -> dict[str, Any]:
    mask = np.asarray(probabilities, dtype=float) >= float(threshold)
    selected = frame.loc[mask].copy()
    if selected.empty:
        return {
            "count": 0,
            "market_count": 0,
            "gross_mean_bps": None,
            "normal_net_mean_bps": None,
            "stressed_net_mean_bps": None,
            "positive_market_fraction_normal": None,
        }
    gross = pd.to_numeric(selected["_return_bps"], errors="coerce")
    normal = gross - float(normal_cost_bps)
    stressed = normal - float(stress_extra_bps)
    per_market = pd.DataFrame(
        {"market": selected["_market"].astype(str), "normal": normal.to_numpy(float)}
    ).groupby("market")["normal"].mean()
    return {
        "count": int(mask.sum()),
        "market_count": int(per_market.size),
        "gross_mean_bps": float(gross.mean()),
        "normal_net_mean_bps": float(normal.mean()),
        "stressed_net_mean_bps": float(stressed.mean()),
        "positive_market_fraction_normal": float((per_market > 0.0).mean()),
    }


class ProspectiveContextAgent:
    """Research-only learner over real Round44 15m PIT snapshots."""

    SCHEMA = "round44_prospective_context_agent_v2"

    def __init__(self, settings) -> None:
        self.settings = settings
        self.root = settings.project_root / "output/crypto_ai_swing/agents/round44_context"
        self.pointer = self.root / "latest.pointer.json"
        self._bundle: dict[str, Any] | None = None
        self._mtime: float | None = None

    def _current_pointer(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.pointer.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def train(
        self,
        database_path: Path,
        *,
        horizon_hours: int = 4,
        minimum_rows: int = 150,
        minimum_markets: int = 5,
        maximum_features: int = 72,
        minimum_net_move_bps: float = 65.0,
        normal_cost_bps: float = 35.0,
        stress_extra_bps: float = 25.0,
    ) -> dict[str, Any]:
        frame = _load_rows(database_path, horizon_hours)
        if len(frame) < minimum_rows:
            return self._status("COLLECTING", rows=len(frame), minimum_rows=minimum_rows)
        if int(frame["_market"].nunique()) < minimum_markets:
            return self._status(
                "COLLECTING_MARKET_BREADTH",
                rows=len(frame),
                market_count=int(frame["_market"].nunique()),
            )
        dataset_hash = _dataset_hash(frame, horizon_hours)
        pointer = self._current_pointer()
        if pointer.get("dataset_hash") == dataset_hash:
            return self._status(
                "NO_NEW_EVIDENCE",
                rows=len(frame),
                dataset_hash=dataset_hash,
                artifact_path=pointer.get("artifact_path"),
            )
        try:
            split = _purged_split(frame)
        except ValueError as exc:
            return self._status(str(exc), rows=len(frame), dataset_hash=dataset_hash)
        selected = _select_features(split.train, maximum_features=maximum_features)
        if len(selected) < 8:
            return self._status(
                "INSUFFICIENT_NONREDUNDANT_FEATURES",
                rows=len(frame),
                selected_features=len(selected),
                dataset_hash=dataset_hash,
            )

        threshold = float(minimum_net_move_bps)
        y_train = (split.train["_return_bps"] >= threshold).astype(int)
        y_validation = (split.validation["_return_bps"] >= threshold).astype(int)
        y_test = (split.test["_return_bps"] >= threshold).astype(int)
        if y_train.nunique() < 2:
            return self._status("ONE_CLASS_TRAIN", rows=len(frame), dataset_hash=dataset_hash)

        candidate_rows: list[dict[str, Any]] = []
        models: dict[int, Pipeline] = {}
        for seed in CANDIDATE_SEEDS:
            model = _classifier(seed)
            model.fit(split.train[selected], y_train)
            probability = model.predict_proba(split.validation[selected])[:, 1]
            auc = _auc(y_validation, probability)
            brier = float(brier_score_loss(y_validation, probability))
            economics = _selection_economics(
                split.validation,
                probability,
                threshold=0.60,
                normal_cost_bps=normal_cost_bps,
                stress_extra_bps=stress_extra_bps,
            )
            objective = (
                (auc if auc is not None else 0.50)
                - 0.50 * brier
                + 0.001 * float(economics.get("stressed_net_mean_bps") or -100.0)
            )
            candidate_rows.append(
                {
                    "seed": seed,
                    "validation_auc": auc,
                    "validation_brier": brier,
                    "validation_economics": economics,
                    "objective": float(objective),
                }
            )
            models[seed] = model
        selected_candidate = max(candidate_rows, key=lambda row: float(row["objective"]))
        selected_seed = int(selected_candidate["seed"])
        classifier = models[selected_seed]
        regressor = _regressor(selected_seed)
        regressor.fit(split.train[selected], split.train["_return_bps"])

        # Untouched OOS test is evaluated once after validation model selection.
        p_test = classifier.predict_proba(split.test[selected])[:, 1]
        r_test = regressor.predict(split.test[selected])
        test_auc = _auc(y_test, p_test)
        test_brier = float(brier_score_loss(y_test, p_test))
        return_mae = float(mean_absolute_error(split.test["_return_bps"], r_test))
        economics = _selection_economics(
            split.test,
            p_test,
            threshold=0.60,
            normal_cost_bps=normal_cost_bps,
            stress_extra_bps=stress_extra_bps,
        )
        seed_auc = [
            row["validation_auc"]
            for row in candidate_rows
            if row.get("validation_auc") is not None
        ]
        seed_robust = bool(
            len(seed_auc) >= 3
            and float(np.median(seed_auc)) >= 0.51
            and max(seed_auc) - min(seed_auc) <= 0.12
        )
        qualification_checks = {
            "validation_auc": selected_candidate.get("validation_auc") is not None
            and float(selected_candidate["validation_auc"]) >= 0.52,
            "test_auc": test_auc is not None and test_auc >= 0.52,
            "test_brier": test_brier <= 0.25,
            "selected_count": int(economics["count"]) >= 20,
            "selected_market_count": int(economics["market_count"]) >= 3,
            "normal_net_positive": economics["normal_net_mean_bps"] is not None
            and float(economics["normal_net_mean_bps"]) > 0.0,
            "stressed_net_positive": economics["stressed_net_mean_bps"] is not None
            and float(economics["stressed_net_mean_bps"]) > 0.0,
            "market_breadth": economics["positive_market_fraction_normal"] is not None
            and float(economics["positive_market_fraction_normal"]) >= 0.50,
            "multi_seed_robustness": seed_robust,
        }
        qualified = all(qualification_checks.values())

        trained_at = datetime.now(UTC)
        self.root.mkdir(parents=True, exist_ok=True)
        artifact = self.root / f"context_agent_{trained_at:%Y%m%dT%H%M%SZ}.joblib"
        bundle = {
            "schema_version": self.SCHEMA,
            "status": "SHADOW_QUALIFIED" if qualified else "RESEARCH_ONLY",
            "trained_at": trained_at.isoformat(),
            "expires_at": (trained_at + timedelta(hours=24)).isoformat(),
            "dataset_hash": dataset_hash,
            "horizon_hours": int(horizon_hours),
            "feature_columns": selected,
            "classifier": classifier,
            "return_regressor": regressor,
            "selected_seed": selected_seed,
            "candidate_seeds": candidate_rows,
            "split_provenance": SPLIT_PROVENANCE,
            "split_rows": {
                "train": len(split.train),
                "validation": len(split.validation),
                "test": len(split.test),
            },
            "metrics": {
                "test_auc": test_auc,
                "test_brier": test_brier,
                "test_return_mae_bps": return_mae,
                "test_economics": economics,
                "qualification_checks": qualification_checks,
            },
            "qualified": qualified,
            "live_decision_influence": False,
            "automatic_live_promotion": False,
        }
        joblib.dump(bundle, artifact)
        pointer_payload = {
            "schema_version": "round44_context_pointer_v2",
            "artifact_path": str(artifact),
            "dataset_hash": dataset_hash,
            "trained_at": trained_at.isoformat(),
            "qualified": qualified,
            "live_decision_influence": False,
        }
        self.pointer.write_text(
            json.dumps(pointer_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._bundle = bundle
        self._mtime = self.pointer.stat().st_mtime
        return {
            **self._status(
                bundle["status"],
                rows=len(frame),
                dataset_hash=dataset_hash,
                artifact_path=str(artifact),
            ),
            "qualified": qualified,
            "selected_features": selected,
            "selected_seed": selected_seed,
            "candidate_seeds": candidate_rows,
            "metrics": bundle["metrics"],
            "split_provenance": SPLIT_PROVENANCE,
        }

    def _load(self) -> dict[str, Any] | None:
        if not self.pointer.is_file():
            self._bundle = None
            return None
        mtime = self.pointer.stat().st_mtime
        if self._bundle is not None and self._mtime == mtime:
            return self._bundle
        try:
            pointer = self._current_pointer()
            self._bundle = joblib.load(Path(str(pointer["artifact_path"])))
            self._mtime = mtime
        except Exception:
            self._bundle = None
            self._mtime = mtime
        return self._bundle

    def predict_vector(self, vector: dict[str, Any]) -> dict[str, Any]:
        bundle = self._load()
        if bundle is None:
            return {
                "status": "NOT_TRAINED",
                "score": None,
                "probability": None,
                "predicted_return_bps": None,
                "live_decision_influence": False,
            }
        expiry = pd.Timestamp(bundle.get("expires_at"))
        expiry = expiry.tz_localize("UTC") if expiry.tzinfo is None else expiry.tz_convert("UTC")
        if pd.Timestamp.now(tz="UTC") > expiry:
            return {
                "status": "EXPIRED",
                "score": None,
                "probability": None,
                "predicted_return_bps": None,
                "live_decision_influence": False,
            }
        features = list(bundle.get("feature_columns") or [])
        row = pd.DataFrame([{name: vector.get(name) for name in features}])
        try:
            probability = float(bundle["classifier"].predict_proba(row)[0, 1])
            predicted_return = float(bundle["return_regressor"].predict(row)[0])
        except Exception as exc:
            return {
                "status": "ERROR",
                "error": f"{type(exc).__name__}:{str(exc)[:200]}",
                "score": None,
                "probability": None,
                "predicted_return_bps": None,
                "live_decision_influence": False,
            }
        return {
            "status": str(bundle.get("status") or "RESEARCH_ONLY"),
            "qualified": bool(bundle.get("qualified", False)),
            "probability": probability,
            "score": float(np.clip(2.0 * probability - 1.0, -1.0, 1.0)),
            "predicted_return_bps": predicted_return,
            "dataset_hash": bundle.get("dataset_hash"),
            "authority": "ADVISORY_ONLY",
            "live_decision_influence": False,
        }

    def _status(self, status: str, **extra: Any) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA,
            "status": status,
            "generated_at": datetime.now(UTC).isoformat(),
            "split_provenance": SPLIT_PROVENANCE,
            "automatic_live_authority": False,
            "automatic_live_promotion": False,
            **extra,
        }
