"""Research-only diagnostics for historical CMC PIT features."""
from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from crypto_ai_swing.agents.cmc_pit_features import augment_cmc_pit_features
from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge


def _atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _spearman(x: pd.Series, y: pd.Series) -> float | None:
    pair = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 80 or pair.iloc[:, 0].nunique() < 4 or pair.iloc[:, 1].nunique() < 4:
        return None
    value = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
    return float(value) if value is not None and math.isfinite(float(value)) else None


def cmc_feature_research(
    settings,
    markets: Sequence[str],
    *,
    timeframe: str = "1d",
    horizon_bars: int = 7,
    maximum_features: int = 80,
) -> dict[str, Any]:
    """Measure PIT feature information coefficients without promotion authority."""
    bridge = CryptoLibraryBridge(settings.crypto_repo_root)
    frames = bridge.historical_ohlcv_many(
        [str(x).upper() for x in markets],
        timeframe,
        provider="bitvavo",
    )
    rows: list[dict[str, Any]] = []
    feature_names: set[str] = set()
    for market, raw in sorted(frames.items()):
        if raw is None or raw.empty or "close" not in raw:
            continue
        base = pd.DataFrame(index=raw.index)
        augmented = augment_cmc_pit_features(bridge, base, market=market)
        cmc_cols = [
            c for c in augmented.columns
            if str(c).startswith("cmc_")
            and pd.api.types.is_numeric_dtype(augmented[c])
        ][: int(maximum_features)]
        if not cmc_cols:
            continue
        close = pd.to_numeric(raw["close"], errors="coerce")
        future = close.shift(-int(horizon_bars)) / close - 1.0
        for name in cmc_cols:
            ic = _spearman(pd.to_numeric(augmented[name], errors="coerce"), future)
            if ic is None:
                continue
            valid = pd.concat([augmented[name], future], axis=1).dropna()
            rows.append(
                {
                    "market": str(market).upper(),
                    "feature": str(name),
                    "spearman_ic": ic,
                    "abs_ic": abs(ic),
                    "rows": int(len(valid)),
                    "timeframe": timeframe,
                    "horizon_bars": int(horizon_bars),
                }
            )
            feature_names.add(str(name))

    table = pd.DataFrame(rows)
    aggregate: list[dict[str, Any]] = []
    if not table.empty:
        for feature, group in table.groupby("feature"):
            values = group["spearman_ic"].astype(float)
            aggregate.append(
                {
                    "feature": str(feature),
                    "market_count": int(group["market"].nunique()),
                    "median_ic": float(values.median()),
                    "mean_ic": float(values.mean()),
                    "median_abs_ic": float(values.abs().median()),
                    "positive_fraction": float((values > 0).mean()),
                }
            )
        aggregate.sort(key=lambda x: x["median_abs_ic"], reverse=True)

    payload = {
        "schema_version": "cmc_pit_feature_research_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "timeframe": timeframe,
        "horizon_bars": int(horizon_bars),
        "market_count": int(table["market"].nunique()) if not table.empty else 0,
        "feature_count": len(feature_names),
        "pair_count": len(rows),
        "aggregate": aggregate,
        "per_market": rows,
        "point_in_time_only": True,
        "forward_only_sources_excluded": True,
        "authority": "RESEARCH_ONLY",
        "automatic_live_promotion": False,
        "orders_submitted": 0,
    }
    target = Path(settings.project_root) / "output/crypto_ai_swing/research/cmc_pit_feature_research.json"
    _atomic(target, payload)
    return payload


__all__ = ["cmc_feature_research"]
