from __future__ import annotations

import pandas as pd


def min_variance_weights(returns: pd.DataFrame) -> pd.Series:
    """Optional skfolio challenger. Imported only when explicitly called."""
    try:
        from skfolio.optimization import MeanRisk
        from skfolio.measures import RiskMeasure
    except Exception as exc:
        raise RuntimeError("skfolio is not installed or its API changed") from exc

    model = MeanRisk(risk_measure=RiskMeasure.VARIANCE)
    model.fit(returns)
    weights = getattr(model, "weights_", None)
    if weights is None:
        raise RuntimeError("skfolio optimizer did not expose weights_")
    return pd.Series(weights, index=returns.columns, dtype=float)
