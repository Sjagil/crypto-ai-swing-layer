"""Leakage-safe CoinMarketCap historical PIT feature bridge.

Only the canonical CMC feature mart's TRUE_HISTORICAL_SOURCE rows are joined.
Forward-only DEX, derivatives, metadata, trending, content and WebSocket data
are intentionally excluded from historical training.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXCLUDED_TOKENS = (
    "target_",
    "future_",
    "forward_",
    "label_",
    "outcome_",
    "pnl_",
)


def _feature_path(bridge: Any) -> Path:
    root_override = str(os.getenv("CMC_STARTUP_ROOT") or "").strip()
    if root_override:
        return Path(root_override).expanduser().resolve() / "features" / "pit_features.parquet"
    native = bridge.settings()
    return Path(native.paths.data_dir) / "coinmarketcap_startup" / "features" / "pit_features.parquet"


def cmc_feature_store_status(bridge: Any) -> dict[str, Any]:
    path = _feature_path(bridge)
    if not path.is_file():
        return {"status": "MISSING", "path": str(path), "rows": 0}
    try:
        frame = pd.read_parquet(path, columns=["market", "available_at", "data_classification"])
    except Exception as exc:
        return {"status": "INVALID", "path": str(path), "rows": 0, "error": f"{type(exc).__name__}:{str(exc)[:300]}"}
    valid = frame[frame["data_classification"].astype(str).eq("TRUE_HISTORICAL_SOURCE")]
    return {
        "status": "READY" if not valid.empty else "NO_TRUE_HISTORICAL_ROWS",
        "path": str(path),
        "rows": int(len(valid)),
        "markets": int(valid["market"].astype(str).nunique()) if not valid.empty else 0,
    }


def augment_cmc_pit_features(
    bridge: Any,
    frame: pd.DataFrame,
    *,
    market: str,
    maximum_age_days: float = 8.0,
) -> pd.DataFrame:
    """Join only information available at or before each feature timestamp."""
    if frame is None or frame.empty:
        return frame
    path = _feature_path(bridge)
    if not path.is_file():
        out = frame.copy()
        out.attrs["cmc_pit_status"] = "MISSING"
        return out

    market_name = str(market).upper().replace("/", "-")
    cmc = pd.read_parquet(path)
    required = {"market", "available_at", "data_classification"}
    if not required.issubset(cmc.columns):
        out = frame.copy()
        out.attrs["cmc_pit_status"] = "INVALID_SCHEMA"
        return out

    cmc = cmc[
        cmc["market"].astype(str).str.upper().eq(market_name)
        & cmc["data_classification"].astype(str).eq("TRUE_HISTORICAL_SOURCE")
    ].copy()
    if cmc.empty:
        out = frame.copy()
        out.attrs["cmc_pit_status"] = "NO_MARKET_HISTORY"
        return out

    cmc["available_at"] = pd.to_datetime(cmc["available_at"], utc=True, errors="coerce")
    cmc = cmc.dropna(subset=["available_at"]).sort_values("available_at")
    numeric = []
    for name in cmc.columns:
        lower = str(name).lower()
        if not lower.startswith("cmc_"):
            continue
        if any(token in lower for token in EXCLUDED_TOKENS):
            continue
        if pd.api.types.is_numeric_dtype(cmc[name]):
            numeric.append(str(name))
    if not numeric:
        out = frame.copy()
        out.attrs["cmc_pit_status"] = "NO_NUMERIC_FEATURES"
        return out

    left = frame.copy()
    original_index = left.index
    feature_time = pd.to_datetime(original_index, utc=True, errors="coerce")
    if feature_time.isna().any():
        raise ValueError("CMC PIT merge requires a valid DatetimeIndex")
    left["__cmc_feature_time"] = feature_time
    left["__cmc_original_position"] = np.arange(len(left), dtype=np.int64)
    left = left.sort_values("__cmc_feature_time")

    right = cmc[["available_at", *numeric]].copy()
    right = right.drop_duplicates("available_at", keep="last")
    right["__cmc_source_available_at"] = right["available_at"]

    joined = pd.merge_asof(
        left,
        right,
        left_on="__cmc_feature_time",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    age_hours = (
        joined["__cmc_feature_time"] - joined["__cmc_source_available_at"]
    ).dt.total_seconds() / 3600.0
    maximum_hours = float(maximum_age_days) * 24.0
    stale = age_hours > maximum_hours
    if stale.any():
        joined.loc[stale, numeric] = np.nan
    joined["cmc_history_age_hours"] = age_hours.where(~stale)
    joined["cmc_history_present"] = (~joined[numeric].isna().all(axis=1)).astype(np.float32)

    joined = joined.sort_values("__cmc_original_position")
    joined.index = original_index
    joined = joined.drop(
        columns=[
            "__cmc_feature_time",
            "__cmc_original_position",
            "available_at",
            "__cmc_source_available_at",
        ],
        errors="ignore",
    )
    joined.attrs.update(frame.attrs)
    joined.attrs["cmc_pit_status"] = "READY"
    joined.attrs["cmc_pit_point_in_time"] = True
    joined.attrs["cmc_forward_only_sources_excluded"] = True
    return joined


__all__ = ["augment_cmc_pit_features", "cmc_feature_store_status"]
