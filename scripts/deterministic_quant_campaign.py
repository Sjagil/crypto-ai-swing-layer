#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.research.autonomous_strategy_director import AutonomousStrategyDirector
from crypto_ai_swing.research.native import NativeResearchBridge
from crypto_ai_swing.settings import Settings
from crypto_ai_swing.universe.runtime import UniverseManager


SYMMETRY_SCHEMA = "deterministic_strategy_symmetry_v1"
SYMMETRY_HORIZONS_HOURS = (4, 24, 72, 168)
SYMMETRY_RANDOM_CONTROLS = 32


def now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return None


def market_list(settings: Settings, raw: str | None) -> list[str]:
    if raw:
        return list(
            dict.fromkeys(
                item.strip().upper()
                for item in raw.split(",")
                if item.strip()
            )
        )
    current = UniverseManager(settings).current()
    return [
        str(item).upper()
        for item in current.get("markets", [])
        if str(item).strip()
    ]


def historical_coverage(
    bridge: CryptoLibraryBridge,
    markets: list[str],
    *,
    timeframes: tuple[str, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for timeframe in timeframes:
        frames = bridge.historical_ohlcv_many(
            markets,
            timeframe,
            provider="bitvavo",
        )
        rows: dict[str, Any] = {}
        for market in markets:
            frame = frames.get(market)
            if frame is None or frame.empty:
                rows[market] = {
                    "rows": 0,
                    "source_path": None,
                    "start": None,
                    "end": None,
                }
                continue
            rows[market] = {
                "rows": int(len(frame)),
                "source_path": frame.attrs.get("source_path"),
                "start": pd.Timestamp(frame.index.min()).isoformat(),
                "end": pd.Timestamp(frame.index.max()).isoformat(),
            }
        result[timeframe] = rows
    return result


def cmc_pit_audit() -> dict[str, Any]:
    root_raw = os.getenv("CMC_STARTUP_ROOT", "").strip()
    if not root_raw:
        return {"status": "CMC_STARTUP_ROOT_NOT_SET"}

    root = Path(root_raw).expanduser()
    candidates = (
        root / "features" / "pit_features.parquet",
        root / "pit_features.parquet",
    )
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return {
            "status": "PIT_FEATURE_FILE_NOT_FOUND",
            "candidates": [str(p) for p in candidates],
        }

    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        return {
            "status": "READ_ERROR",
            "path": str(path),
            "error": f"{type(exc).__name__}:{str(exc)[:500]}",
        }

    time_column = next(
        (
            name
            for name in (
                "available_at",
                "timestamp",
                "date",
                "feature_time",
                "observed_at",
            )
            if name in frame.columns
        ),
        None,
    )
    start = end = None
    if time_column:
        values = pd.to_datetime(
            frame[time_column],
            utc=True,
            errors="coerce",
        ).dropna()
        if len(values):
            start = values.min().isoformat()
            end = values.max().isoformat()

    return {
        "status": "READY",
        "path": str(path.resolve()),
        "rows": int(len(frame)),
        "columns": list(map(str, frame.columns)),
        "column_count": int(len(frame.columns)),
        "time_column": time_column,
        "start": start,
        "end": end,
    }


def capability_audit(
    bridge: CryptoLibraryBridge,
    native: NativeResearchBridge,
    markets: list[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}

    native_status = native.status()
    output["native_research"] = {
        "ready": native_status.ready,
        "imported_modules": native_status.imported_modules,
        "required_modules": native_status.required_modules,
        "promotion_states": list(native_status.promotion_states),
    }

    indicator_module = bridge.import_module("research.indicator_registry")
    registry = indicator_module.indicator_registry()
    definitions = list(registry.definitions())
    output["indicator_registry"] = {
        "definitions": len(definitions),
        "families": dict(
            sorted(Counter(str(item.family) for item in definitions).items())
        ),
        "status_counts": dict(
            sorted(Counter(str(item.status) for item in definitions).items())
        ),
        "tradable": sum(bool(item.tradable) for item in definitions),
        "combinable": sum(bool(item.combinable) for item in definitions),
        "repainting": sum(bool(item.repaints) for item in definitions),
    }

    simple_module = bridge.import_module("research.simple_strategy_lab")
    blocks = dict(simple_module.registry_driven_signal_blocks())
    output["signal_blocks"] = {
        "registered": len(blocks),
        "families": dict(
            sorted(Counter(str(item.family) for item in blocks.values()).items())
        ),
        "roles": dict(
            sorted(Counter(str(item.role) for item in blocks.values()).items())
        ),
    }

    tactical_module = bridge.import_module("research.tactical_multitimeframe")
    tactical_specs = list(tactical_module.tactical_strategy_specs())
    output["tactical_strategies"] = [
        {
            "strategy_id": item.strategy_id,
            "family": item.family,
            "timeframe": item.timeframe,
            "confirmation_timeframe": item.confirmation_timeframe,
            "regime_timeframe": item.regime_timeframe,
            "mechanism": item.mechanism,
            "dna_hash": item.dna_hash,
        }
        for item in tactical_specs
    ]

    mechanics_module = bridge.import_module("research.gex_orderflow_strategies")
    mechanics_specs = list(mechanics_module.market_mechanics_strategy_specs())
    output["gex_orderflow_strategies"] = [
        {
            "strategy_id": item.strategy_id,
            "family": item.family,
            "mechanism": item.mechanism,
            "gex_policy": item.gex_policy,
            "market_scope": item.market_scope,
            "entry_timeframe": item.entry_timeframe,
            "dna_hash": item.dna_hash,
        }
        for item in mechanics_specs
    ]

    macro_module = bridge.import_module("research.macro_context")
    output["macro_context"] = {
        "groups": list(getattr(macro_module, "GROUPS", ())),
        "point_in_time_alignment": hasattr(macro_module, "causal_align"),
    }

    derivatives_module = bridge.import_module("data.derivatives_context")
    output["derivatives_context"] = {
        "gex_analyzer": hasattr(derivatives_module, "CryptoGEXAnalyzer"),
        "options_contract": hasattr(derivatives_module, "OptionsContract"),
        "funding_collector": hasattr(derivatives_module, "FundingRateCollector"),
        "full_greeks_delta_theta_vega_native": all(
            hasattr(derivatives_module, name)
            for name in ("delta", "theta", "vega")
        ),
        "note": (
            "Canonical repo has gamma/GEX context. Full delta/theta/vega are "
            "not claimed unless native functions are actually present."
        ),
    }

    structure_module = bridge.import_module(
        "legacy.market_structure_patterns"
    )
    structure_builder = getattr(
        structure_module,
        "build_market_structure_features",
        None,
    )
    structure_rows: dict[str, Any] = {}
    if callable(structure_builder):
        frames = bridge.historical_ohlcv_many(
            markets[: min(5, len(markets))],
            "1h",
            provider="bitvavo",
        )
        for market, frame in frames.items():
            if frame is None or frame.empty:
                structure_rows[market] = {"status": "NO_DATA"}
                continue
            sample = frame.iloc[-5000:].copy()
            try:
                features = structure_builder(sample)
                bool_columns = [
                    str(column)
                    for column in features.columns
                    if str(features[column].dtype)
                    in {"bool", "boolean"}
                ]
                structure_rows[market] = {
                    "status": "READY",
                    "rows": int(len(features)),
                    "feature_count": int(len(features.columns)),
                    "boolean_feature_count": len(bool_columns),
                    "boolean_events": {
                        column: int(
                            pd.Series(features[column])
                            .astype("boolean")
                            .fillna(False)
                            .astype(bool)
                            .sum()
                        )
                        for column in bool_columns[:80]
                    },
                }
            except Exception as exc:
                structure_rows[market] = {
                    "status": "ERROR",
                    "error": (
                        f"{type(exc).__name__}:{str(exc)[:500]}"
                    ),
                }
    output["market_structure_fractals"] = structure_rows

    output["cmc_pit"] = cmc_pit_audit()
    return output


def materialize_simple_lab(
    bridge: CryptoLibraryBridge,
    *,
    batch_size: int,
) -> dict[str, Any]:
    module = bridge.import_module("research.simple_strategy_lab")
    factory = module.SimpleStrategyResearchFactory(bridge.settings())
    result = factory.materialize_batch(
        batch_size=int(batch_size),
        complexities=(1, 2, 3, 4, 5),
        timeframes=("15m", "1h", "2h", "4h", "1d", "1W"),
        resume=True,
    )
    return {
        "materialize": result,
        "queue_status": factory.queue_status(),
        "output_dir": str(factory.output_dir.resolve()),
        "queue_path": str(factory.queue_path.resolve()),
        "cursor_path": str(factory.cursor_path.resolve()),
    }


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _profit_factor(returns: np.ndarray) -> float | None:
    positive = float(returns[returns > 0].sum())
    negative = float(abs(returns[returns < 0].sum()))
    if negative <= 0:
        return None
    return positive / negative


def _maximum_drawdown(returns: np.ndarray) -> float | None:
    if not len(returns):
        return None
    equity = np.cumprod(1.0 + returns)
    if not np.isfinite(equity).all():
        return None
    peaks = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peaks, 1e-12) - 1.0
    return float(drawdown.min(initial=0.0))


def _outcome_summary(returns: Sequence[float]) -> dict[str, Any]:
    values = _finite(returns)
    if not len(values):
        return {
            "observations": 0,
            "mean_net_bps": None,
            "median_net_bps": None,
            "positive_fraction": None,
            "profit_factor": None,
            "maximum_drawdown": None,
        }
    return {
        "observations": int(len(values)),
        "mean_net_bps": float(values.mean() * 10_000.0),
        "median_net_bps": float(np.median(values) * 10_000.0),
        "positive_fraction": float(np.mean(values > 0.0)),
        "profit_factor": _profit_factor(values),
        "maximum_drawdown": _maximum_drawdown(values),
    }


def _seed(label: str) -> int:
    return int(
        hashlib.sha256(label.encode("utf-8")).hexdigest()[:8],
        16,
    )


def _signal_outcomes(
    frame: pd.DataFrame,
    signal: pd.Series,
    *,
    horizon_bars: int,
    cost_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Next-open fixed-horizon long and hypothetical inverse outcomes.

    The inverse path is a research-only directional diagnostic:
        inverse_gross = -original_gross
    and receives the same conservative round-trip cost drag.

    It is NOT a shorting implementation and never receives execution authority.
    """

    if horizon_bars < 1:
        raise ValueError("horizon_bars must be positive")

    raw = signal.reindex(frame.index).fillna(False).astype(bool)
    entry = pd.to_numeric(frame["open"], errors="coerce").shift(-1)
    exit_price = pd.to_numeric(
        frame["open"],
        errors="coerce",
    ).shift(-(1 + horizon_bars))

    valid = (
        raw
        & entry.gt(0)
        & exit_price.gt(0)
        & entry.notna()
        & exit_price.notna()
    )
    gross = (
        exit_price.loc[valid].to_numpy(dtype=float)
        / entry.loc[valid].to_numpy(dtype=float)
        - 1.0
    )
    original = gross - float(cost_fraction)
    inverse = -gross - float(cost_fraction)
    return original, inverse


def _delayed_signal(signal: pd.Series, bars: int = 1) -> pd.Series:
    if bars < 0:
        raise ValueError("delay cannot be negative")
    return (
        signal.shift(bars, fill_value=False)
        .fillna(False)
        .astype(bool)
    )


def _random_control_means(
    frame: pd.DataFrame,
    signal: pd.Series,
    *,
    horizon_bars: int,
    cost_fraction: float,
    draws: int,
    seed: int,
) -> list[float]:
    """Deterministic random-timestamp null preserving signal count.

    The control samples entry timestamps only from rows that have a valid
    next-open entry and complete fixed-horizon outcome. It never becomes a
    deployable strategy and is used only to check whether the observed signal
    mean is distinguishable from arbitrary timing.
    """

    if draws < 1 or len(frame) < 10:
        return []
    rng = np.random.default_rng(seed)
    source = (
        signal.reindex(frame.index)
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    signal_count = int(source.sum())
    valid_count = len(frame) - horizon_bars - 1
    if signal_count < 1 or valid_count < 2:
        return []

    sample_size = min(signal_count, valid_count)
    eligible_positions = np.arange(valid_count, dtype=int)
    result: list[float] = []
    for _ in range(draws):
        positions = rng.choice(
            eligible_positions,
            size=sample_size,
            replace=False,
        )
        randomized_values = np.zeros(len(frame), dtype=bool)
        randomized_values[positions] = True
        randomized = pd.Series(
            randomized_values,
            index=frame.index,
        )
        normal, _ = _signal_outcomes(
            frame,
            randomized,
            horizon_bars=horizon_bars,
            cost_fraction=cost_fraction,
        )
        if len(normal):
            result.append(float(np.mean(normal)))
    return result


def _classification(
    original: Mapping[str, Any],
    inverse: Mapping[str, Any],
) -> tuple[str, str]:
    original_mean = original.get("mean_net_bps")
    inverse_mean = inverse.get("mean_net_bps")
    if original_mean is None or inverse_mean is None:
        return "NOT_EVALUABLE", "NO_DIRECTIONAL_DECISION"

    original_positive = float(original_mean) > 0.0
    inverse_positive = float(inverse_mean) > 0.0

    if original_positive:
        return "POSITIVE_DIRECTIONAL_ALPHA", "KEEP_ORIGINAL_FOR_DEEP_VALIDATION"
    if inverse_positive:
        return (
            "NEGATIVE_ALPHA_INFORMATION",
            "RETAIN_AS_AVOIDANCE_EXIT_OR_RISK_OFF_CANDIDATE",
        )
    return (
        "BOTH_DIRECTIONS_NET_NEGATIVE_AFTER_COSTS",
        "DO_NOT_PROMOTE_DIRECTION_BUT_KEEP_FOR_FILTER_ABLATION_IF_USEFUL",
    )


def directional_symmetry_for_candidate(
    *,
    bridge: CryptoLibraryBridge,
    candidate: Mapping[str, Any],
    markets: Sequence[str],
    timeframe: str,
    maximum_rows: int,
    random_controls: int,
) -> dict[str, Any]:
    """Run a cheap point-in-time directional symmetry screen.

    This deliberately does not create a short-capable Strategy or modify the
    canonical BacktestEngine. Original entries are evaluated at next open.
    The inverse is a hypothetical sign-flipped outcome path used only to detect
    negative directional information. If the inverse is robust, the spot-safe
    downstream use is an avoidance/exit/risk-off candidate, not a short order.
    """

    combo_module = bridge.import_module("research.combinatorial_lab")
    feature_module = bridge.import_module("research.features")
    economics_module = bridge.import_module("core.economics")

    registry = dict(
        bridge.import_module(
            "research.simple_strategy_lab"
        ).registry_driven_signal_blocks()
    )
    logic_mode = combo_module.LogicMode(
        str(candidate.get("logic_mode") or "LAYERED")
    )
    combination = combo_module.CombinationGenerator(
        registry
    ).materialize_membership(
        tuple(str(value) for value in candidate.get("block_ids") or ()),
        logic_mode=logic_mode,
        mode=combo_module.GenerationMode.FAMILY_AWARE,
        timeframes=(timeframe,),
    )
    if str(combination.eligibility_status) not in {
        "GENERATED",
        "CombinationState.GENERATED",
    } and getattr(
        combination.eligibility_status,
        "value",
        "",
    ) != "GENERATED":
        return {
            "status": "NOT_EVALUABLE",
            "reason": (
                f"COMBINATION_{getattr(combination.eligibility_status, 'value', combination.eligibility_status)}"
            ),
            "strategy_dna_hash": candidate.get(
                "strategy_dna_hash"
            ),
        }

    strategy = combo_module.CombinatorialStrategy(
        combination,
        registry,
    )
    timeframe_seconds = int(
        bridge.import_module("config.settings").TIMEFRAME_SECONDS[
            timeframe
        ]
    )
    canonical_costs = economics_module.CanonicalCostModel.from_settings(
        bridge.settings()
    )
    cost_fraction = float(
        canonical_costs.conservative_roundtrip_fraction
    )

    frames = bridge.historical_ohlcv_many(
        list(markets),
        timeframe,
        provider="bitvavo",
    )

    market_rows: dict[str, Any] = {}
    aggregate: dict[int, dict[str, list[float]]] = {
        horizon: {
            "original": [],
            "inverse": [],
            "delayed": [],
        }
        for horizon in SYMMETRY_HORIZONS_HOURS
    }
    missing_context: dict[str, str] = {}

    for market in markets:
        raw = frames.get(market)
        if raw is None or raw.empty:
            market_rows[market] = {
                "status": "NO_DATA",
            }
            continue
        selected = raw.iloc[-maximum_rows:].copy()
        try:
            features = feature_module.FeaturePipeline().build(
                selected,
                market=market,
            )
            output = strategy.generate(features)
        except Exception as exc:
            message = f"{type(exc).__name__}:{str(exc)[:500]}"
            missing_context[market] = message
            market_rows[market] = {
                "status": "NOT_EVALUABLE_MISSING_OR_UNBUILT_CONTEXT",
                "error": message,
            }
            continue

        usable_entry = (
            output.entry
            & ~output.avoid
            & output.size_multiplier.gt(0)
        )
        delayed = _delayed_signal(usable_entry, 1)
        per_horizon: dict[str, Any] = {}

        for horizon_hours in SYMMETRY_HORIZONS_HOURS:
            horizon_bars = max(
                1,
                int(
                    round(
                        horizon_hours * 3600
                        / timeframe_seconds
                    )
                ),
            )
            original_values, inverse_values = _signal_outcomes(
                selected,
                usable_entry,
                horizon_bars=horizon_bars,
                cost_fraction=cost_fraction,
            )
            delayed_values, _ = _signal_outcomes(
                selected,
                delayed,
                horizon_bars=horizon_bars,
                cost_fraction=cost_fraction,
            )
            null_means = _random_control_means(
                selected,
                usable_entry,
                horizon_bars=horizon_bars,
                cost_fraction=cost_fraction,
                draws=random_controls,
                seed=_seed(
                    f"{combination.strategy_dna_hash}:{market}:{timeframe}:{horizon_hours}"
                ),
            )

            aggregate[horizon_hours]["original"].extend(
                original_values.tolist()
            )
            aggregate[horizon_hours]["inverse"].extend(
                inverse_values.tolist()
            )
            aggregate[horizon_hours]["delayed"].extend(
                delayed_values.tolist()
            )

            original_summary = _outcome_summary(
                original_values
            )
            inverse_summary = _outcome_summary(
                inverse_values
            )
            classification, spot_use = _classification(
                original_summary,
                inverse_summary,
            )
            null_array = _finite(null_means)
            per_horizon[str(horizon_hours)] = {
                "horizon_hours": horizon_hours,
                "horizon_bars": horizon_bars,
                "original": original_summary,
                "inverse_directional_diagnostic": (
                    inverse_summary
                ),
                "delayed_one_bar": _outcome_summary(
                    delayed_values
                ),
                "randomized_null": {
                    "draws": int(len(null_array)),
                    "mean_of_randomized_mean_net_bps": (
                        float(null_array.mean() * 10_000.0)
                        if len(null_array)
                        else None
                    ),
                    "p95_randomized_mean_net_bps": (
                        float(
                            np.quantile(
                                null_array,
                                0.95,
                            )
                            * 10_000.0
                        )
                        if len(null_array)
                        else None
                    ),
                },
                "directional_edge_bps": (
                    (
                        original_summary["mean_net_bps"]
                        - inverse_summary["mean_net_bps"]
                    )
                    if (
                        original_summary["mean_net_bps"]
                        is not None
                        and inverse_summary["mean_net_bps"]
                        is not None
                    )
                    else None
                ),
                "edge_strength_bps": (
                    abs(
                        original_summary["mean_net_bps"]
                        - inverse_summary["mean_net_bps"]
                    )
                    if (
                        original_summary["mean_net_bps"]
                        is not None
                        and inverse_summary["mean_net_bps"]
                        is not None
                    )
                    else None
                ),
                "classification": classification,
                "spot_safe_research_action": spot_use,
            }

        market_rows[market] = {
            "status": "READY",
            "source_path": selected.attrs.get(
                "source_path"
            ),
            "rows": int(len(selected)),
            "entry_signals": int(
                usable_entry.fillna(False).sum()
            ),
            "horizons": per_horizon,
        }

    aggregate_rows: dict[str, Any] = {}
    inverse_signal_detected = False
    for horizon_hours, values in aggregate.items():
        original_summary = _outcome_summary(
            values["original"]
        )
        inverse_summary = _outcome_summary(
            values["inverse"]
        )
        delayed_summary = _outcome_summary(
            values["delayed"]
        )
        classification, spot_use = _classification(
            original_summary,
            inverse_summary,
        )
        inverse_signal_detected |= (
            classification == "NEGATIVE_ALPHA_INFORMATION"
        )
        aggregate_rows[str(horizon_hours)] = {
            "original": original_summary,
            "inverse_directional_diagnostic": (
                inverse_summary
            ),
            "delayed_one_bar": delayed_summary,
            "directional_edge_bps": (
                (
                    original_summary["mean_net_bps"]
                    - inverse_summary["mean_net_bps"]
                )
                if (
                    original_summary["mean_net_bps"]
                    is not None
                    and inverse_summary["mean_net_bps"]
                    is not None
                )
                else None
            ),
            "edge_strength_bps": (
                abs(
                    original_summary["mean_net_bps"]
                    - inverse_summary["mean_net_bps"]
                )
                if (
                    original_summary["mean_net_bps"]
                    is not None
                    and inverse_summary["mean_net_bps"]
                    is not None
                )
                else None
            ),
            "classification": classification,
            "spot_safe_research_action": spot_use,
        }

    return {
        "schema_version": SYMMETRY_SCHEMA,
        "status": (
            "READY"
            if any(
                row.get("status") == "READY"
                for row in market_rows.values()
            )
            else "NOT_EVALUABLE"
        ),
        "strategy_dna_hash": combination.strategy_dna_hash,
        "source_strategy_dna_hash": candidate.get(
            "strategy_dna_hash"
        ),
        "combination_id": combination.combination_id,
        "block_ids": list(combination.block_ids),
        "families": list(combination.families),
        "logic_mode": combination.logic_mode.value,
        "timeframe": timeframe,
        "cost_model_version": (
            canonical_costs.cost_model_version
        ),
        "conservative_roundtrip_cost_bps": (
            cost_fraction * 10_000.0
        ),
        "horizons": aggregate_rows,
        "markets": market_rows,
        "missing_context": missing_context,
        "inverse_directional_information_detected": (
            inverse_signal_detected
        ),
        "inverse_execution_authority": False,
        "inverse_short_orders_permitted": False,
        "spot_safe_use_of_negative_alpha": (
            "AVOIDANCE_EXIT_OR_RISK_OFF_FILTER_ONLY"
        ),
        "selection_rule": (
            "POLARITY_MUST_BE_SELECTED_BEFORE_FINAL_HOLDOUT; "
            "INVERSE_COUNTS_AS_SEPARATE_HYPOTHESIS"
        ),
        "randomized_control_role": (
            "NULL_DIAGNOSTIC_ONLY_NOT_DEPLOYABLE"
        ),
        "delayed_control_role": (
            "TIMING_ROBUSTNESS_ONLY_NOT_A_NEW_STRATEGY"
        ),
    }


def run_symmetry_screen(
    bridge: CryptoLibraryBridge,
    *,
    markets: list[str],
    limit: int,
    timeframe: str,
    maximum_rows: int,
    random_controls: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Screen queued deterministic DNA for original vs inverse information."""

    if limit < 1:
        raise ValueError("symmetry-screen limit must be positive")

    simple_module = bridge.import_module(
        "research.simple_strategy_lab"
    )
    factory = simple_module.SimpleStrategyResearchFactory(
        bridge.settings()
    )
    candidates = factory.queued_strategies(limit=limit)

    results: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    inverse_candidates: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates, start=1):
        try:
            row = directional_symmetry_for_candidate(
                bridge=bridge,
                candidate=candidate,
                markets=markets,
                timeframe=timeframe,
                maximum_rows=maximum_rows,
                random_controls=random_controls,
            )
        except Exception as exc:
            row = {
                "schema_version": SYMMETRY_SCHEMA,
                "status": "ERROR",
                "strategy_dna_hash": candidate.get(
                    "strategy_dna_hash"
                ),
                "error": (
                    f"{type(exc).__name__}:{str(exc)[:800]}"
                ),
            }
        row["screen_sequence"] = index
        results.append(row)
        counts[str(row.get("status") or "UNKNOWN")] += 1

        for horizon, metrics in (
            row.get("horizons") or {}
        ).items():
            if (
                metrics.get("classification")
                == "NEGATIVE_ALPHA_INFORMATION"
            ):
                inverse_candidates.append(
                    {
                        "strategy_dna_hash": row.get(
                            "strategy_dna_hash"
                        ),
                        "timeframe": row.get("timeframe"),
                        "horizon_hours": int(horizon),
                        "classification": (
                            "NEGATIVE_ALPHA_INFORMATION"
                        ),
                        "spot_safe_research_action": (
                            "RETAIN_AS_AVOIDANCE_EXIT_OR_RISK_OFF_CANDIDATE"
                        ),
                        "original": metrics.get("original"),
                        "inverse_directional_diagnostic": (
                            metrics.get(
                                "inverse_directional_diagnostic"
                            )
                        ),
                    }
                )

    payload = {
        "schema_version": SYMMETRY_SCHEMA,
        "generated_at": now(),
        "timeframe": timeframe,
        "candidate_count": len(candidates),
        "market_count": len(markets),
        "markets": markets,
        "maximum_rows_per_market": maximum_rows,
        "horizons_hours": list(
            SYMMETRY_HORIZONS_HOURS
        ),
        "random_controls_per_horizon": random_controls,
        "status_counts": dict(sorted(counts.items())),
        "effective_directional_hypotheses": (
            len(candidates) * 2
        ),
        "multiple_testing_note": (
            "ORIGINAL_AND_INVERSE_ARE_COUNTED_AS_SEPARATE_DIRECTIONAL_HYPOTHESES"
        ),
        "inverse_candidate_count": len(
            inverse_candidates
        ),
        "inverse_candidates": inverse_candidates,
        "results": results,
        "authority": "RESEARCH_ONLY",
        "shorting_enabled": False,
        "derivatives_execution_enabled": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "symmetry_latest.json", payload)
    with (
        output_dir / "symmetry_history.jsonl"
    ).open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                payload,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic quant research orchestrator. Reuses the existing "
            "Sjagil/crypto research stack; it does not add ML/AI authority."
        )
    )
    parser.add_argument("--markets", default=None)
    parser.add_argument(
        "--mode",
        choices=("audit", "stage0", "exact"),
        default="audit",
    )
    parser.add_argument(
        "--materialize-batch",
        type=int,
        default=0,
        help=(
            "Materialize this many registry-driven strategy DNA candidates."
        ),
    )
    parser.add_argument(
        "--timeframes",
        default="15m,1h,4h,1d",
    )
    parser.add_argument(
        "--symmetry-screen",
        type=int,
        default=0,
        help=(
            "Evaluate N queued deterministic DNA with original/inverse "
            "directional diagnostics, one-bar delay, and randomized null "
            "controls. Research-only; no short execution."
        ),
    )
    parser.add_argument(
        "--symmetry-timeframe",
        default="1h",
    )
    parser.add_argument(
        "--symmetry-maximum-rows",
        type=int,
        default=20_000,
    )
    parser.add_argument(
        "--symmetry-random-controls",
        type=int,
        default=SYMMETRY_RANDOM_CONTROLS,
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    settings = Settings.load(root)
    markets = market_list(settings, args.markets)
    if not markets:
        raise SystemExit("No runtime markets found.")

    bridge = CryptoLibraryBridge(
        settings.crypto_repo_root
    )
    native = NativeResearchBridge(
        settings.crypto_repo_root
    )

    stamp = datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    out_root = (
        root
        / "output"
        / "crypto_ai_swing"
        / "research"
        / "deterministic_quant"
    )
    run_root = out_root / "runs" / stamp
    run_root.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": (
            "deterministic_quant_campaign_v2"
        ),
        "started_at": now(),
        "mode": args.mode,
        "markets": markets,
        "swing_repo": str(root.resolve()),
        "swing_git_head": git_head(root),
        "crypto_repo": str(
            Path(settings.crypto_repo_root).resolve()
        ),
        "crypto_git_head": git_head(
            Path(settings.crypto_repo_root)
        ),
        "ai_used_for_signal_generation": False,
        "ai_used_for_model_selection": False,
        "inverse_testing_enabled": (
            args.symmetry_screen > 0
        ),
        "inverse_short_execution_enabled": False,
        "execution_authority_changed": False,
        "capital_authority_changed": False,
        "orders_submitted": 0,
    }

    timeframes = tuple(
        item.strip()
        for item in args.timeframes.split(",")
        if item.strip()
    )

    payload["historical_coverage"] = (
        historical_coverage(
            bridge,
            markets,
            timeframes=timeframes,
        )
    )
    payload["capabilities"] = capability_audit(
        bridge,
        native,
        markets,
    )

    if args.materialize_batch > 0:
        payload["simple_strategy_lab"] = (
            materialize_simple_lab(
                bridge,
                batch_size=args.materialize_batch,
            )
        )

    if args.symmetry_screen > 0:
        symmetry_root = (
            out_root / "strategy_symmetry"
        )
        payload["strategy_symmetry"] = (
            run_symmetry_screen(
                bridge,
                markets=markets,
                limit=args.symmetry_screen,
                timeframe=args.symmetry_timeframe,
                maximum_rows=max(
                    500,
                    args.symmetry_maximum_rows,
                ),
                random_controls=max(
                    1,
                    args.symmetry_random_controls,
                ),
                output_dir=symmetry_root,
            )
        )

    if args.mode in {"stage0", "exact"}:
        director = AutonomousStrategyDirector(
            settings,
            mode="shadow",
        )
        payload["deterministic_director"] = (
            director.cycle(
                markets=markets,
                priorities=[],
                force=True,
                force_exact=(
                    args.mode == "exact"
                ),
            )
        )

    payload["completed_at"] = now()

    run_path = run_root / "campaign.json"
    atomic_json(run_path, payload)

    symmetry = payload.get(
        "strategy_symmetry"
    ) or {}
    latest = {
        "schema_version": payload[
            "schema_version"
        ],
        "generated_at": payload[
            "completed_at"
        ],
        "mode": args.mode,
        "run_path": str(run_path.resolve()),
        "markets": len(markets),
        "simple_strategy_lab": (
            (
                payload.get(
                    "simple_strategy_lab"
                )
                or {}
            ).get("queue_status", {})
        ),
        "symmetry_screen": {
            "candidate_count": symmetry.get(
                "candidate_count"
            ),
            "status_counts": symmetry.get(
                "status_counts"
            ),
            "inverse_candidate_count": (
                symmetry.get(
                    "inverse_candidate_count"
                )
            ),
            "effective_directional_hypotheses": (
                symmetry.get(
                    "effective_directional_hypotheses"
                )
            ),
            "shorting_enabled": False,
        },
        "deterministic_director_status": (
            (
                payload.get(
                    "deterministic_director"
                )
                or {}
            ).get("status")
        ),
        "orders_submitted": 0,
        "execution_authority_changed": False,
    }
    atomic_json(out_root / "latest.json", latest)

    print(
        json.dumps(
            latest,
            indent=2,
            default=str,
        )
    )
    print(f"RUN_ARTIFACT={run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
