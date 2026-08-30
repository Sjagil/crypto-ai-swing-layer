from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline


@dataclass
class SupervisedModel:
    feature_columns: list[str]
    pipeline: Pipeline

    @classmethod
    def create(cls, feature_columns: list[str]) -> "SupervisedModel":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=200,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=17,
                    ),
                ),
            ]
        )
        return cls(feature_columns=feature_columns, pipeline=model)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "SupervisedModel":
        self.pipeline.fit(x[self.feature_columns], y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> pd.Series:
        prob = self.pipeline.predict_proba(x[self.feature_columns])[:, 1]
        return pd.Series(prob, index=x.index, name="ml_probability")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"features": self.feature_columns, "pipeline": self.pipeline}, path)

    @classmethod
    def load(cls, path: Path) -> "SupervisedModel":
        raw = joblib.load(path)
        return cls(feature_columns=list(raw["features"]), pipeline=raw["pipeline"])
