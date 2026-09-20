from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

NOISE_CONTROL_VERSION = "round47f_noise_control_v1"
DEFAULT_STABILITY_BLOCKS = 4
MAX_FEATURE_SELECTION_ROWS = 100_000


def _bounded_train_selection_sample(
    train: pd.DataFrame,
    target: pd.Series | None,
    *,
    maximum_rows: int = MAX_FEATURE_SELECTION_ROWS,
) -> tuple[pd.DataFrame, pd.Series | None]:
    if len(train) <= int(maximum_rows):
        selected_target = (
            _as_numeric_positional(target, length=len(train)) if target is not None else None
        )
        return train, selected_target

    maximum_rows = max(1_000, int(maximum_rows))
    positions: list[int] = []
    if "market" in train.columns:
        markets = train["market"].astype(str).to_numpy()
        names = sorted(set(markets))
        quota = max(50, maximum_rows // max(1, len(names)))
        for market in names:
            available = np.flatnonzero(markets == market)
            if len(available) <= quota:
                positions.extend(int(v) for v in available)
            else:
                local = np.linspace(0, len(available) - 1, quota, dtype=int)
                positions.extend(int(available[index]) for index in local)

    if len(positions) < maximum_rows:
        already = set(positions)
        for value in np.linspace(0, len(train) - 1, maximum_rows, dtype=int):
            index = int(value)
            if index not in already:
                positions.append(index)
                already.add(index)
            if len(positions) >= maximum_rows:
                break

    positions = sorted(set(positions))[:maximum_rows]
    sampled = train.iloc[positions]
    sampled_target = None
    if target is not None:
        numeric_target = _as_numeric_positional(target, length=len(train))
        sampled_target = numeric_target.iloc[positions].reset_index(drop=True)
    return sampled, sampled_target


_GROUP_WEIGHTS = {
    "pattern": 0.13,
    "strategy": 0.07,
    "vwap": 0.05,
    "index": 0.06,
    "structure": 0.06,
    "volume_liquidity": 0.06,
    "volatility": 0.05,
    "cmc": 0.06,
    "core": 0.06,
    "mtf_1h": 0.08,
    "mtf_2h": 0.07,
    "mtf_4h": 0.08,
    "mtf_1d": 0.07,
    "mtf_1w": 0.03,
    "mtf_cross": 0.07,
}

_GROUP_ORDER = (
    "pattern",
    "strategy",
    "vwap",
    "index",
    "structure",
    "volume_liquidity",
    "volatility",
    "cmc",
    "core",
    "mtf_1h",
    "mtf_2h",
    "mtf_4h",
    "mtf_1d",
    "mtf_1w",
    "mtf_cross",
)


@dataclass(frozen=True)
class FeatureScore:
    name: str
    group: str
    coverage: float
    unique_count: int
    binary: bool
    activity_quality: float
    global_ic: float
    median_abs_block_ic: float
    sign_consistency: float
    block_presence: float
    stability_score: float
    unsupervised_quality: float


def feature_group(name: str) -> str:
    token = str(name).lower()

    for timeframe in ("1h", "2h", "4h", "1d", "1w"):
        if token.startswith(f"mtf_{timeframe}__") or token.startswith(f"htf_{timeframe}_"):
            return f"mtf_{timeframe}"
    if token.startswith("mtf_cross__"):
        return "mtf_cross"

    if token.startswith("cmc_"):
        return "cmc"

    if token.startswith("pattern_"):
        return "pattern"
    if (
        token.startswith("crypto_strategy_")
        or token.startswith("crypto_tactical_")
        or token.startswith("strategy_family_")
    ):
        return "strategy"
    if (
        token.startswith("crypto_vwap")
        or token.startswith("crypto_anchored_vwap")
        or "vwap" in token
    ):
        return "vwap"
    if token.startswith("crypto_idx_"):
        return "index"

    structure_tokens = (
        "bos",
        "choch",
        "fractal",
        "swing_",
        "range_position",
        "donchian",
        "breakout",
        "breakdown",
        "support",
        "resistance",
        "market_structure",
    )
    if any(value in token for value in structure_tokens):
        return "structure"

    volume_tokens = (
        "volume",
        "obv",
        "chaikin",
        "money_flow",
        "mfi_",
        "force_index",
        "liquidity",
        "orderflow",
        "order_flow",
        "cvd",
        "ofi",
        "vzo",
    )
    if any(value in token for value in volume_tokens):
        return "volume_liquidity"

    volatility_tokens = (
        "atr",
        "volatility",
        "realized_vol",
        "bollinger",
        "keltner",
        "choppiness",
        "variance",
        "garch",
        "std_",
        "_std",
    )
    if any(value in token for value in volatility_tokens):
        return "volatility"

    return "core"


def _as_numeric_positional(
    values: pd.Series | Iterable[float],
    *,
    length: int,
) -> pd.Series:
    if isinstance(values, pd.Series):
        numeric = pd.to_numeric(values, errors="coerce").reset_index(drop=True)
    else:
        numeric = pd.to_numeric(
            pd.Series(list(values)),
            errors="coerce",
        ).reset_index(drop=True)

    if len(numeric) != int(length):
        raise ValueError(f"target length {len(numeric)} != train length {length}")
    return numeric


def _activity_quality(values: pd.Series) -> tuple[bool, float]:
    clean = values.dropna()
    if clean.empty:
        return False, 0.0

    unique = int(clean.nunique())
    if unique == 2:
        counts = clean.value_counts(normalize=True)
        minority = float(counts.min()) if len(counts) >= 2 else 0.0
        return True, float(np.clip(2.0 * minority, 0.0, 1.0))

    q05 = float(clean.quantile(0.05))
    q95 = float(clean.quantile(0.95))
    spread = abs(q95 - q05)
    scale = max(
        abs(float(clean.median())),
        float(clean.abs().median()),
        1e-9,
    )
    ratio = spread / scale
    quality = float(np.tanh(max(0.0, ratio)))
    return False, quality


def _winsorized(values: pd.Series) -> pd.Series:
    clean = values.dropna()
    if len(clean) < 20:
        return values
    lo = float(clean.quantile(0.01))
    hi = float(clean.quantile(0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return values
    return values.clip(lo, hi)


def _rank_ic(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 20:
        return 0.0

    x_valid = x[valid]
    y_valid = y[valid]
    if x_valid.nunique(dropna=True) < 2 or y_valid.nunique(dropna=True) < 2:
        return 0.0

    value = x_valid.rank(method="average").corr(y_valid.rank(method="average"))
    return float(value) if pd.notna(value) and np.isfinite(value) else 0.0


def _feature_score(
    train: pd.DataFrame,
    name: str,
    *,
    target: pd.Series | None,
    blocks: int,
    minimum_coverage: float,
) -> FeatureScore | None:
    if name not in train.columns:
        return None

    values = pd.to_numeric(
        train[name],
        errors="coerce",
    ).reset_index(drop=True)
    n = len(values)
    coverage = float(values.notna().mean()) if n else 0.0
    if coverage < float(minimum_coverage):
        return None

    unique_count = int(values.nunique(dropna=True))
    if unique_count < 2:
        return None

    binary, activity_quality = _activity_quality(values)
    if binary:
        minority_count = int(values.dropna().value_counts().min())
        if minority_count < max(8, int(0.002 * max(1, n))):
            return None
    elif activity_quality <= 1e-8:
        return None

    unsupervised_quality = float(
        coverage * max(activity_quality, 0.05) * min(1.0, np.log1p(unique_count) / np.log(21.0))
    )

    if target is None:
        return FeatureScore(
            name=str(name),
            group=feature_group(name),
            coverage=coverage,
            unique_count=unique_count,
            binary=binary,
            activity_quality=activity_quality,
            global_ic=0.0,
            median_abs_block_ic=0.0,
            sign_consistency=0.0,
            block_presence=0.0,
            stability_score=0.0,
            unsupervised_quality=unsupervised_quality,
        )

    x = _winsorized(values)
    global_ic = _rank_ic(x, target)

    block_values: list[float] = []
    for indices in np.array_split(np.arange(n), max(2, int(blocks))):
        if len(indices) < 20:
            continue
        block_values.append(
            _rank_ic(
                x.iloc[indices].reset_index(drop=True),
                target.iloc[indices].reset_index(drop=True),
            )
        )

    nonzero = [value for value in block_values if abs(value) >= 1e-6]
    median_abs = float(np.median(np.abs(nonzero))) if nonzero else 0.0

    if nonzero:
        reference_sign = np.sign(global_ic)
        if reference_sign == 0.0:
            reference_sign = np.sign(np.median(nonzero))
        if reference_sign == 0.0:
            sign_consistency = 0.0
        else:
            sign_consistency = float(
                np.mean([np.sign(value) == reference_sign for value in nonzero])
            )
        block_presence = float(np.mean([abs(value) >= 0.01 for value in block_values]))
    else:
        sign_consistency = 0.0
        block_presence = 0.0

    raw_signal = 0.45 * abs(global_ic) + 0.55 * median_abs
    stability_score = float(
        raw_signal * (0.25 + 0.75 * sign_consistency) * (0.50 + 0.50 * block_presence) * coverage
    )

    return FeatureScore(
        name=str(name),
        group=feature_group(name),
        coverage=coverage,
        unique_count=unique_count,
        binary=binary,
        activity_quality=activity_quality,
        global_ic=float(global_ic),
        median_abs_block_ic=median_abs,
        sign_consistency=sign_consistency,
        block_presence=block_presence,
        stability_score=stability_score,
        unsupervised_quality=unsupervised_quality,
    )


def _group_budgets(total: int) -> dict[str, int]:
    total = max(1, int(total))
    budgets = {
        group: max(1, int(round(total * weight))) for group, weight in _GROUP_WEIGHTS.items()
    }

    while sum(budgets.values()) > total:
        reducible = [group for group in reversed(_GROUP_ORDER) if budgets[group] > 1]
        if not reducible:
            break
        budgets[reducible[0]] -= 1

    while sum(budgets.values()) < total:
        budgets["core"] += 1

    return budgets


def feature_selection_diagnostics(
    train: pd.DataFrame,
    candidates: Iterable[str],
    *,
    target: pd.Series | None = None,
    minimum_coverage: float = 0.70,
    stability_blocks: int = DEFAULT_STABILITY_BLOCKS,
) -> list[dict[str, object]]:
    target_numeric = (
        _as_numeric_positional(
            target,
            length=len(train),
        )
        if target is not None
        else None
    )

    rows: list[FeatureScore] = []
    for name in dict.fromkeys(str(value) for value in candidates):
        scored = _feature_score(
            train,
            name,
            target=target_numeric,
            blocks=stability_blocks,
            minimum_coverage=minimum_coverage,
        )
        if scored is not None:
            rows.append(scored)

    rows.sort(
        key=lambda row: (
            -row.stability_score,
            -row.unsupervised_quality,
            -row.coverage,
            row.name,
        )
    )
    return [asdict(row) for row in rows]


def denoised_candidate_columns(
    frame: pd.DataFrame,
    *,
    minimum_coverage: float = 0.70,
    maximum_candidates: int = 180,
) -> tuple[str, ...]:
    diagnostics = feature_selection_diagnostics(
        frame,
        tuple(str(name) for name in frame.columns),
        target=None,
        minimum_coverage=minimum_coverage,
    )
    if not diagnostics:
        return ()

    by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in diagnostics:
        by_group[str(row["group"])].append(row)

    budgets = _group_budgets(maximum_candidates)
    selected: list[str] = []
    used: set[str] = set()

    for group in _GROUP_ORDER:
        ordered = sorted(
            by_group.get(group, []),
            key=lambda row: (
                -float(row["unsupervised_quality"]),
                -float(row["coverage"]),
                str(row["name"]),
            ),
        )
        for row in ordered[: budgets[group]]:
            name = str(row["name"])
            if name not in used:
                selected.append(name)
                used.add(name)

    if len(selected) < int(maximum_candidates):
        remaining = sorted(
            diagnostics,
            key=lambda row: (
                -float(row["unsupervised_quality"]),
                -float(row["coverage"]),
                str(row["name"]),
            ),
        )
        for row in remaining:
            name = str(row["name"])
            if name in used:
                continue
            selected.append(name)
            used.add(name)
            if len(selected) >= int(maximum_candidates):
                break

    return tuple(selected[: int(maximum_candidates)])


def select_stable_train_features(
    train: pd.DataFrame,
    candidates: Iterable[str],
    *,
    target: pd.Series | None = None,
    maximum_features: int = 96,
    minimum_coverage: float = 0.70,
    maximum_abs_correlation: float = 0.95,
    stability_blocks: int = DEFAULT_STABILITY_BLOCKS,
) -> tuple[str, ...]:
    scoring_train, scoring_target = _bounded_train_selection_sample(
        train,
        target,
        maximum_rows=MAX_FEATURE_SELECTION_ROWS,
    )
    diagnostics = feature_selection_diagnostics(
        scoring_train,
        candidates,
        target=scoring_target,
        minimum_coverage=minimum_coverage,
        stability_blocks=stability_blocks,
    )
    if not diagnostics:
        return ()

    by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in diagnostics:
        by_group[str(row["group"])].append(row)

    ordered_names = [str(row["name"]) for row in diagnostics]
    numeric = scoring_train.reindex(columns=ordered_names).apply(pd.to_numeric, errors="coerce")
    for name in numeric.columns:
        if pd.api.types.is_float_dtype(numeric[name]):
            numeric[name] = numeric[name].astype(np.float32, copy=False)
    corr = numeric.corr(method="spearman").abs()

    def redundant(name: str, selected: list[str]) -> bool:
        for other in selected:
            if name not in corr.index or other not in corr.columns:
                continue
            value = corr.loc[name, other]
            if pd.notna(value) and float(value) >= float(maximum_abs_correlation):
                return True
        return False

    budgets = _group_budgets(maximum_features)
    selected: list[str] = []
    for group in _GROUP_ORDER:
        rows = sorted(
            by_group.get(group, []),
            key=lambda row: (
                -float(row["stability_score"]),
                -float(row["unsupervised_quality"]),
                -float(row["coverage"]),
                str(row["name"]),
            ),
        )
        if not rows:
            continue
        budget = budgets[group]
        exploration_floor = max(1, budget // 4)
        accepted = 0
        for rank, row in enumerate(rows):
            name = str(row["name"])
            stable = float(row["stability_score"]) >= 0.002
            exploratory = rank < exploration_floor
            if scoring_target is not None and not (stable or exploratory):
                continue
            if redundant(name, selected):
                continue
            selected.append(name)
            accepted += 1
            if accepted >= budget:
                break

    if len(selected) < int(maximum_features):
        for row in diagnostics:
            name = str(row["name"])
            if name in selected or redundant(name, selected):
                continue
            if scoring_target is not None:
                if float(row["stability_score"]) < 0.001 and len(selected) >= max(
                    8, int(maximum_features * 0.75)
                ):
                    continue
            selected.append(name)
            if len(selected) >= int(maximum_features):
                break

    return tuple(selected[: int(maximum_features)])


def feature_group_counts(features: Iterable[str]) -> dict[str, int]:
    counts = Counter(feature_group(str(name)) for name in features)
    return {group: int(counts.get(group, 0)) for group in _GROUP_ORDER}


__all__ = [
    "NOISE_CONTROL_VERSION",
    "denoised_candidate_columns",
    "feature_group",
    "feature_group_counts",
    "feature_selection_diagnostics",
    "select_stable_train_features",
]
