from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
import json
from typing import Any

import numpy as np
import pandas as pd

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.data.features import build_features
from crypto_ai_swing.research.backtest import long_flat_backtest
from crypto_ai_swing.universe.runtime import UniverseManager


@dataclass(frozen=True)
class Candidate:
    family: str
    parameters: dict[str, Any]


def _forced_holding_exit(
    entries: pd.Series,
    base_exit: pd.Series,
    maximum_bars: int,
) -> pd.Series:
    entries = entries.astype(bool)
    base_exit = base_exit.astype(bool)
    result = pd.Series(False, index=entries.index, dtype=bool)
    active = False
    age = 0
    for i, _ in enumerate(entries.index):
        if not active and bool(entries.iloc[i]):
            active = True
            age = 0
            continue
        if active:
            age += 1
            if bool(base_exit.iloc[i]) or age >= int(maximum_bars):
                result.iloc[i] = True
                active = False
                age = 0
    return result


def _raw_signals(frame: pd.DataFrame, candidate: Candidate) -> tuple[pd.Series, pd.Series]:
    """Build full-history shifted signals so rolling features have warm-up history."""
    feat = build_features(frame)
    p = candidate.parameters
    if candidate.family == "TREND_PULLBACK":
        raw_entry = (
            (feat["trend_20_50"] > 0)
            & (feat["trend_50_200"] > -0.01)
            & (feat["rsi_14"] >= p["rsi_min"])
            & (feat["rsi_14"] <= p["rsi_max"])
            & (feat["bb_z"] <= p["bb_z_max"])
            & (feat["trend_8_20"] > 0)
        )
        raw_exit = (feat["trend_8_20"] < 0) | (feat["rsi_14"] > 75)
    elif candidate.family == "DONCHIAN_BREAKOUT":
        raw_entry = (
            (feat["breakout_20"] > p["breakout"])
            & (feat["trend_20_50"] > 0)
            & (feat["volume_z_48"] >= p["volume_z"])
        )
        raw_exit = (feat["trend_8_20"] < 0) | (feat["ret_8"] < -0.02)
    elif candidate.family == "MOMENTUM_24":
        raw_entry = (
            (feat["ret_24"] >= p["ret_24"])
            & (feat["trend_20_50"] > 0)
            & (feat["rsi_14"] <= p["rsi_max"])
        )
        raw_exit = (feat["ret_8"] < 0) | (feat["trend_8_20"] < 0)
    elif candidate.family == "MEAN_REVERSION_UPTREND":
        raw_entry = (
            (feat["trend_20_50"] > 0)
            & (feat["bb_z"] <= p["bb_z"])
            & (feat["rsi_14"] <= p["rsi_max"])
        )
        raw_exit = (feat["bb_z"] >= 0) | (feat["rsi_14"] >= 55)
    elif candidate.family == "VOL_EXPANSION":
        raw_entry = (
            (feat["breakout_20"] > 0)
            & (feat["volume_z_48"] >= p["volume_z"])
            & (feat["vol_regime"] >= p["vol_regime"])
            & (feat["trend_20_50"] > 0)
        )
        raw_exit = (feat["trend_8_20"] < 0) | (feat["ret_4"] < -0.015)
    else:
        raise ValueError(candidate.family)
    # A signal known after close t becomes executable on bar t+1.
    entries = raw_entry.astype(bool).shift(1, fill_value=False).astype(bool)
    exits = raw_exit.astype(bool).shift(1, fill_value=False).astype(bool)
    return entries, exits


def _piece_signals(
    full_entries: pd.Series,
    full_exits: pd.Series,
    piece: pd.DataFrame,
    maximum_bars: int,
) -> tuple[pd.Series, pd.Series]:
    entries = full_entries.reindex(piece.index, fill_value=False).astype(bool)
    base_exit = full_exits.reindex(piece.index, fill_value=False).astype(bool)
    exits = _forced_holding_exit(entries, base_exit, maximum_bars)
    return entries, exits


def _candidate_grid() -> list[Candidate]:
    """Pre-registered bounded classical grid with broader parameter neighborhoods."""
    result: list[Candidate] = []
    for rsi_max, bb in product((55, 60, 65), (-0.50, -0.25, 0.0)):
        result.append(Candidate(
            "TREND_PULLBACK",
            {"rsi_min": 38, "rsi_max": rsi_max, "bb_z_max": bb, "maximum_bars": 48},
        ))
    for breakout, volume in product((0.0, 0.001, 0.002, 0.004), (0.5, 1.0)):
        result.append(Candidate(
            "DONCHIAN_BREAKOUT",
            {"breakout": breakout, "volume_z": volume, "maximum_bars": 72},
        ))
    for ret24, rsi_max in product((0.01, 0.02, 0.04), (68, 75)):
        result.append(Candidate(
            "MOMENTUM_24",
            {"ret_24": ret24, "rsi_max": rsi_max, "maximum_bars": 48},
        ))
    for bb, rsi_max in product((-1.0, -1.5, -2.0), (30, 35)):
        result.append(Candidate(
            "MEAN_REVERSION_UPTREND",
            {"bb_z": bb, "rsi_max": rsi_max, "maximum_bars": 36},
        ))
    for volume, regime in product((0.75, 1.0, 1.5), (1.0, 1.10, 1.25)):
        result.append(Candidate(
            "VOL_EXPANSION",
            {"volume_z": volume, "vol_regime": regime, "maximum_bars": 48},
        ))
    return result

def _split(frame: pd.DataFrame, purge: int = 4) -> dict[str, pd.DataFrame]:
    n = len(frame)
    train_end = int(n * 0.60)
    val_start = min(n, train_end + purge)
    val_end = int(n * 0.80)
    test_start = min(n, val_end + purge)
    return {
        "train": frame.iloc[: max(0, train_end - purge)],
        "validation": frame.iloc[val_start: max(val_start, val_end - purge)],
        "test": frame.iloc[test_start:],
    }



def _walk_forward_slices(
    frame: pd.DataFrame,
    *,
    folds: int = 3,
    purge: int = 4,
    development_fraction: float = 0.80,
) -> list[dict[str, Any]]:
    """Sequential development-only windows. The final holdout is never touched."""
    n = len(frame)
    development_end = max(0, min(n, int(n * float(development_fraction))))
    if development_end < 120:
        return []
    fold_count = max(2, int(folds))
    validation_region_start = int(development_end * 0.50)
    available = max(0, development_end - validation_region_start)
    window = max(1, available // fold_count)
    result: list[dict[str, Any]] = []
    for fold in range(fold_count):
        val_start_raw = validation_region_start + fold * window
        val_end_raw = (
            development_end
            if fold == fold_count - 1
            else validation_region_start + (fold + 1) * window
        )
        train_end = max(0, val_start_raw - purge)
        val_start = min(development_end, val_start_raw + purge)
        val_end = max(val_start, val_end_raw - purge)
        train = frame.iloc[:train_end]
        validation = frame.iloc[val_start:val_end]
        result.append(
            {
                "fold": fold + 1,
                "train": train,
                "validation": validation,
                "train_end_fraction": train_end / n if n else 0.0,
                "validation_start_fraction": val_start / n if n else 0.0,
                "validation_end_fraction": val_end / n if n else 0.0,
            }
        )
    return result


def _walk_forward_summary(folds: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [
        row for row in folds
        if row.get("validation", {}).get("evaluated")
        and int(row.get("validation", {}).get("markets") or 0) > 0
    ]
    if not evaluated:
        return {
            "folds_requested": len(folds),
            "folds_evaluated": 0,
            "profitable_fold_fraction": 0.0,
            "median_fold_median_return": None,
            "worst_fold_median_return": None,
            "mean_positive_market_fraction": 0.0,
            "minimum_positive_market_fraction": 0.0,
            "worst_max_drawdown": None,
            "total_validation_trades": 0,
            "stability_pass": False,
        }
    medians = np.asarray([
        float(row["validation"].get("median_total_return") or 0.0)
        for row in evaluated
    ])
    breadth = np.asarray([
        float(row["validation"].get("positive_market_fraction") or 0.0)
        for row in evaluated
    ])
    drawdowns = np.asarray([
        float(row["validation"].get("worst_max_drawdown") or 0.0)
        for row in evaluated
    ])
    profitable = float(np.mean(medians > 0.0))
    total_trades = int(sum(
        int(row["validation"].get("trades") or 0) for row in evaluated
    ))
    stability = bool(
        len(evaluated) >= 2
        and profitable >= (2.0 / 3.0)
        and float(np.median(medians)) > 0.0
        and float(np.mean(breadth)) >= 0.50
        and float(np.max(drawdowns)) <= 0.30
        and total_trades >= 30
    )
    return {
        "folds_requested": len(folds),
        "folds_evaluated": len(evaluated),
        "profitable_fold_fraction": profitable,
        "median_fold_median_return": float(np.median(medians)),
        "worst_fold_median_return": float(np.min(medians)),
        "mean_positive_market_fraction": float(np.mean(breadth)),
        "minimum_positive_market_fraction": float(np.min(breadth)),
        "worst_max_drawdown": float(np.max(drawdowns)),
        "total_validation_trades": total_trades,
        "stability_pass": stability,
    }


def _diversified_shortlist(
    rows: list[dict[str, Any]],
    size: int,
    *,
    maximum_per_family: int = 2,
) -> list[dict[str, Any]]:
    """Prevent one parameter family from consuming the entire final holdout."""
    target = max(1, min(int(size), len(rows)))
    cap = max(1, int(maximum_per_family))
    selected: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    for row in rows:
        family = str(row["family"])
        if family_counts.get(family, 0) >= cap:
            continue
        selected.append(row)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= target:
            return selected
    # If diversity caps leave too few candidates, fill by development rank.
    selected_ids = {int(row["candidate_id"]) for row in selected}
    for row in rows:
        if int(row["candidate_id"]) in selected_ids:
            continue
        selected.append(row)
        if len(selected) >= target:
            break
    return selected


def _empty_aggregate(*, evaluated: bool) -> dict[str, Any]:
    return {
        "evaluated": evaluated,
        "markets": 0,
        "trades": 0,
        "mean_total_return": 0.0 if evaluated else None,
        "median_total_return": 0.0 if evaluated else None,
        "positive_market_fraction": 0.0 if evaluated else None,
        "worst_max_drawdown": 1.0 if evaluated else None,
        "mean_profit_factor": None,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("rows", 0) > 0]
    if not valid:
        return _empty_aggregate(evaluated=True)
    returns = np.asarray([float(row["total_return"]) for row in valid])
    pfs = [
        float(row["profit_factor"])
        for row in valid
        if np.isfinite(float(row["profit_factor"]))
    ]
    return {
        "evaluated": True,
        "markets": len(valid),
        "trades": int(sum(int(row["trades"]) for row in valid)),
        "mean_total_return": float(returns.mean()),
        "median_total_return": float(np.median(returns)),
        "positive_market_fraction": float((returns > 0).mean()),
        "worst_max_drawdown": float(
            max(float(row["max_drawdown"]) for row in valid)
        ),
        "mean_profit_factor": float(np.mean(pfs)) if pfs else None,
    }


def _validation_rank(row: dict[str, Any]) -> tuple[float, ...]:
    train = row["aggregate"]["train"]
    val = row["aggregate"]["validation"]
    wf = row.get("walk_forward_summary") or {}
    train_sane = bool(
        train["trades"] >= 20
        and train["worst_max_drawdown"] <= 0.55
        and train["median_total_return"] > -0.08
    )
    val_pf = val.get("mean_profit_factor")
    return (
        1.0 if bool(wf.get("stability_pass")) else 0.0,
        1.0 if train_sane else 0.0,
        float(wf.get("profitable_fold_fraction") or 0.0),
        float(wf.get("median_fold_median_return") or -999.0),
        float(wf.get("minimum_positive_market_fraction") or 0.0),
        -float(wf.get("worst_max_drawdown") or 999.0),
        float(val.get("positive_market_fraction") or 0.0),
        float(val.get("median_total_return") or -999.0),
        float(val_pf) if val_pf is not None else 0.0,
        -float(val.get("worst_max_drawdown") or 999.0),
        float(val.get("mean_total_return") or -999.0),
        float(val.get("trades") or 0),
    )

def _rejection_reasons(aggregate: dict[str, Any], walk_forward: dict[str, Any] | None = None) -> list[str]:
    train = aggregate["train"]
    val = aggregate["validation"]
    test = aggregate["test"]
    reasons: list[str] = []
    wf = dict(walk_forward or {})
    if int(wf.get("folds_evaluated") or 0) < 2:
        reasons.append("WALK_FORWARD_INSUFFICIENT_FOLDS")
    elif not bool(wf.get("stability_pass")):
        reasons.append("WALK_FORWARD_UNSTABLE")
    if train["trades"] < 30:
        reasons.append("TRAIN_SAMPLE_TOO_SMALL")
    if train["worst_max_drawdown"] > 0.50:
        reasons.append("TRAIN_DRAWDOWN_EXCESSIVE")
    if train["median_total_return"] <= -0.05:
        reasons.append("TRAIN_MEDIAN_NEGATIVE")
    if val["trades"] < 20:
        reasons.append("VALIDATION_SAMPLE_TOO_SMALL")
    if val["median_total_return"] <= 0:
        reasons.append("VALIDATION_MEDIAN_NOT_POSITIVE")
    if val["positive_market_fraction"] < 0.55:
        reasons.append("VALIDATION_BREADTH_WEAK")
    if val["worst_max_drawdown"] > 0.25:
        reasons.append("VALIDATION_DRAWDOWN_EXCESSIVE")
    if not test.get("evaluated"):
        reasons.append("FINAL_HOLDOUT_NOT_EVALUATED")
        return reasons
    if test["trades"] < 20:
        reasons.append("TEST_SAMPLE_TOO_SMALL")
    if test["median_total_return"] <= 0:
        reasons.append("TEST_MEDIAN_NOT_POSITIVE")
    if test["positive_market_fraction"] < 0.50:
        reasons.append("TEST_BREADTH_WEAK")
    if test["worst_max_drawdown"] > 0.30:
        reasons.append("TEST_DRAWDOWN_EXCESSIVE")
    pf = test.get("mean_profit_factor")
    if pf is None or float(pf) < 1.05:
        reasons.append("TEST_PROFIT_FACTOR_WEAK")
    return reasons


class ColdStartResearchRunner:
    """Bounded research for the period before canonical P0.5 family episodes exist.

    Candidate selection uses train and validation only. A bounded shortlist is then
    evaluated once on the final holdout. Current-universe membership and current
    spread costs are not point-in-time historical evidence, so this remains
    research-only and cannot grant paper or live authority.
    """

    def __init__(self, settings) -> None:
        self.settings = settings
        self.crypto = CryptoLibraryBridge(settings.crypto_repo_root)
        self.universe = UniverseManager(settings)

    def _cost_map(
        self,
        snapshot: dict[str, Any],
        *,
        uniform_override: float | None,
    ) -> dict[str, float]:
        if uniform_override is not None:
            return {
                str(market): float(uniform_override)
                for market in snapshot.get("markets", [])
            }
        execution = dict(getattr(self.settings, "execution", {}) or {})
        cost_cfg = dict(execution.get("costs", {}) or {})
        fee = float(cost_cfg.get("fee_bps_per_side", 25.0))
        base_slippage = float(cost_cfg.get("base_slippage_bps", 2.0))
        candidate_map = {
            str(row.get("market")): row
            for row in snapshot.get("candidates", [])
            if isinstance(row, dict) and row.get("market")
        }
        result: dict[str, float] = {}
        hard_spread = float(
            snapshot.get("policy", {}).get("hard_maximum_spread_bps", 35.0)
        )
        for market in snapshot.get("markets", []):
            row = candidate_map.get(str(market), {})
            try:
                spread = float(row.get("spread_bps"))
            except (TypeError, ValueError):
                spread = hard_spread
            if not np.isfinite(spread) or spread < 0:
                spread = hard_spread
            result[str(market)] = fee + base_slippage + 0.5 * spread
        return result

    @staticmethod
    def _evaluate_piece(
        market: str,
        piece: pd.DataFrame,
        entries: pd.Series,
        exits: pd.Series,
        *,
        cost_bps_per_side: float,
    ) -> dict[str, Any] | None:
        if len(piece) < 48:
            return None
        # Shifted signals are executed at the next bar open. This is still a
        # simplified cold-start proxy, but it no longer labels close execution
        # as next-open execution.
        result = long_flat_backtest(
            piece["open"],
            entries,
            exits,
            cost_bps_per_side=float(cost_bps_per_side),
        )
        metrics = dict(result["metrics"])
        return {
            "market": market,
            "rows": len(piece),
            "trades": int(metrics.get("trades") or 0),
            "total_return": float(metrics.get("total_return") or 0.0),
            "profit_factor": float(metrics.get("profit_factor") or 0.0),
            "expectancy": float(metrics.get("expectancy") or 0.0),
            "max_drawdown": float(metrics.get("max_drawdown") or 0.0),
            "psr": float(metrics.get("psr") or 0.0),
            "cost_bps_per_side": float(cost_bps_per_side),
        }

    def run(
        self,
        *,
        markets: list[str] | None = None,
        timeframe: str = "1h",
        cost_bps_per_side: float | None = None,
        final_holdout_shortlist_size: int = 5,
    ) -> dict[str, Any]:
        snapshot = self.universe.current()
        selected = [str(x).upper() for x in (markets or []) if str(x).strip()]
        if not selected:
            selected = list(snapshot["markets"])
        frames = self.crypto.ohlcv_many(
            selected, timeframe, persist=False, concurrency=4
        )
        frames = {
            m: f
            for m, f in frames.items()
            if f is not None and len(f) >= 300
        }
        if len(frames) < max(5, min(15, len(selected) // 2)):
            raise RuntimeError("insufficient market coverage for cold-start research")

        cost_map = self._cost_map(
            snapshot,
            uniform_override=cost_bps_per_side,
        )
        research_cfg = dict(
            (getattr(self.settings, "autonomy", {}) or {}).get("research", {})
        )
        walk_forward_folds = max(
            2, int(research_cfg.get("cold_start_walk_forward_folds", 3))
        )
        final_holdout_fraction = float(
            research_cfg.get("cold_start_final_holdout_fraction", 0.20)
        )
        development_fraction = max(
            0.60, min(0.90, 1.0 - final_holdout_fraction)
        )
        maximum_per_family = max(
            1, int(research_cfg.get("cold_start_max_finalists_per_family", 2))
        )

        prepared: dict[str, dict[str, Any]] = {}
        for market, frame in frames.items():
            prepared[market] = {
                "frame": frame,
                "parts": _split(frame, purge=4),
                "walk_forward": _walk_forward_slices(
                    frame,
                    folds=walk_forward_folds,
                    purge=4,
                    development_fraction=development_fraction,
                ),
            }

        candidate_results: list[dict[str, Any]] = []
        candidate_objects: dict[int, Candidate] = {}
        candidates = _candidate_grid()
        for idx, candidate in enumerate(candidates):
            split_rows: dict[str, list[dict[str, Any]]] = {
                "train": [],
                "validation": [],
            }
            fold_rows: list[list[dict[str, Any]]] = [
                [] for _ in range(walk_forward_folds)
            ]
            fold_metadata: list[dict[str, Any]] = [
                {
                    "fold": fold + 1,
                    "train_end_fraction": None,
                    "validation_start_fraction": None,
                    "validation_end_fraction": None,
                }
                for fold in range(walk_forward_folds)
            ]

            for market, prepared_row in prepared.items():
                frame = prepared_row["frame"]
                full_entries, full_exits = _raw_signals(frame, candidate)
                for split_name in ("train", "validation"):
                    piece = prepared_row["parts"][split_name]
                    entries, exits = _piece_signals(
                        full_entries,
                        full_exits,
                        piece,
                        int(candidate.parameters.get("maximum_bars", 48)),
                    )
                    result = self._evaluate_piece(
                        market,
                        piece,
                        entries,
                        exits,
                        cost_bps_per_side=cost_map.get(market, 32.5),
                    )
                    if result is not None:
                        split_rows[split_name].append(result)

                for fold_idx, fold in enumerate(prepared_row["walk_forward"]):
                    piece = fold["validation"]
                    entries, exits = _piece_signals(
                        full_entries,
                        full_exits,
                        piece,
                        int(candidate.parameters.get("maximum_bars", 48)),
                    )
                    result = self._evaluate_piece(
                        market,
                        piece,
                        entries,
                        exits,
                        cost_bps_per_side=cost_map.get(market, 32.5),
                    )
                    if result is not None:
                        fold_rows[fold_idx].append(result)
                    if fold_metadata[fold_idx]["train_end_fraction"] is None:
                        for key in (
                            "train_end_fraction",
                            "validation_start_fraction",
                            "validation_end_fraction",
                        ):
                            fold_metadata[fold_idx][key] = float(fold[key])

            walk_forward: list[dict[str, Any]] = []
            for fold_idx, rows in enumerate(fold_rows):
                row = dict(fold_metadata[fold_idx])
                row["validation"] = _aggregate(rows)
                walk_forward.append(row)
            wf_summary = _walk_forward_summary(walk_forward)
            aggregate = {
                "train": _aggregate(split_rows["train"]),
                "validation": _aggregate(split_rows["validation"]),
                "test": _empty_aggregate(evaluated=False),
            }
            row = {
                "candidate_id": idx,
                "family": candidate.family,
                "parameters": candidate.parameters,
                "aggregate": aggregate,
                "walk_forward": walk_forward,
                "walk_forward_summary": wf_summary,
                "final_holdout_evaluated": False,
                "survivor": False,
            }
            candidate_results.append(row)
            candidate_objects[idx] = candidate

        candidate_results.sort(key=_validation_rank, reverse=True)
        shortlist_size = max(
            1,
            min(int(final_holdout_shortlist_size), len(candidate_results)),
        )
        shortlist = _diversified_shortlist(
            candidate_results,
            shortlist_size,
            maximum_per_family=maximum_per_family,
        )
        shortlist_ids = {int(row["candidate_id"]) for row in shortlist}

        for row in shortlist:
            candidate = candidate_objects[int(row["candidate_id"])]
            test_rows: list[dict[str, Any]] = []
            for market, prepared_row in prepared.items():
                frame = prepared_row["frame"]
                full_entries, full_exits = _raw_signals(frame, candidate)
                piece = prepared_row["parts"]["test"]
                entries, exits = _piece_signals(
                    full_entries,
                    full_exits,
                    piece,
                    int(candidate.parameters.get("maximum_bars", 48)),
                )
                result = self._evaluate_piece(
                    market,
                    piece,
                    entries,
                    exits,
                    cost_bps_per_side=cost_map.get(market, 32.5),
                )
                if result is not None:
                    test_rows.append(result)
            row["aggregate"]["test"] = _aggregate(test_rows)
            row["final_holdout_evaluated"] = True
            reasons = _rejection_reasons(
                row["aggregate"],
                row.get("walk_forward_summary"),
            )
            row["rejection_reasons"] = reasons
            row["survivor"] = len(reasons) == 0

        for row in candidate_results:
            if int(row["candidate_id"]) not in shortlist_ids:
                row["rejection_reasons"] = ["FINAL_HOLDOUT_NOT_EVALUATED"]

        survivors = [row for row in shortlist if row["survivor"]]
        near_candidates = [row for row in shortlist if not row["survivor"]]
        family_counts: dict[str, int] = {}
        for row in shortlist:
            family = str(row["family"])
            family_counts[family] = family_counts.get(family, 0) + 1

        payload = {
            "schema_version": "crypto_ai_swing_cold_start_research_v3",
            "status": "COMPLETED",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "authority": "RESEARCH_ONLY",
            "universe": selected,
            "market_count_requested": len(selected),
            "market_count_loaded": len(frames),
            "timeframe": timeframe,
            "cost_model": {
                "mode": (
                    "UNIFORM_OVERRIDE"
                    if cost_bps_per_side is not None
                    else "CURRENT_PUBLIC_SPREAD_FLOOR"
                ),
                "uniform_cost_bps_per_side": cost_bps_per_side,
                "market_cost_bps_per_side": {
                    market: round(float(cost_map.get(market, 0.0)), 6)
                    for market in selected
                    if market in cost_map
                },
                "current_cost_snapshot_not_point_in_time": True,
                "volatility_and_participation_stress_deferred_to_exact_validation": True,
            },
            "next_open_execution_proxy": True,
            "closed_signal_shift_bars": 1,
            "rolling_feature_warmup_preserved_across_partitions": True,
            "current_universe_selection_not_point_in_time": True,
            "candidate_count": len(candidate_results),
            "candidate_grid_preregistered": True,
            "candidate_selection_source": (
                "TRAIN_VALIDATION_AND_WALK_FORWARD_DEVELOPMENT_ONLY"
            ),
            "test_used_for_candidate_selection": False,
            "walk_forward": {
                "folds": walk_forward_folds,
                "development_fraction": development_fraction,
                "final_holdout_fraction": final_holdout_fraction,
                "selection_uses_final_holdout": False,
                "minimum_stability_rule": (
                    "2_OF_3_POSITIVE_FOLDS_WITH_BREADTH_AND_DRAWDOWN_GATES"
                ),
            },
            "final_holdout_shortlist_size": shortlist_size,
            "final_holdout_multiple_testing_candidates": shortlist_size,
            "finalist_family_cap": maximum_per_family,
            "finalist_family_counts": family_counts,
            "holdout_tested_count": len(shortlist),
            "survivor_count": len(survivors),
            "survivors": survivors,
            "near_candidates": near_candidates,
            "top_candidates": candidate_results[:20],
            "automatic_live_promotion": False,
            "paper_authority_granted": False,
            "live_authority_granted": False,
            "orders_submitted": 0,
        }
        out = (
            self.settings.project_root
            / "output/crypto_ai_swing/research/bootstrap_latest.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        payload["output"] = str(out)
        return payload

