from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_context_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize already-timestamped feed/scraper output without inventing publication times."""
    out = df.copy()
    if "usable_at" not in out.columns:
        raise ValueError("Context data requires causal usable_at timestamp")
    out["usable_at"] = pd.to_datetime(out["usable_at"], utc=True)
    if "score" not in out.columns:
        out["score"] = 0.0
    out["score"] = pd.to_numeric(out["score"], errors="coerce").clip(-1.0, 1.0)
    if "confidence" not in out.columns:
        out["confidence"] = 0.5
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").clip(0.0, 1.0)
    out["weighted_score"] = out["score"] * out["confidence"]
    return out.replace([np.inf, -np.inf], np.nan)
