from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from crypto_ai_swing.agents.canonical_features import (
    candidate_columns,
    canonical_model_frame,
)

MTF_FUSION_VERSION = "round47h_final_canonical_mtf_v2"
MTF_FEATURE_SOURCE = "canonical_mtf_feature_pipeline_v2"

_TIMEFRAME_SECONDS = {
    "15m": 15 * 60,
    "1h": 60 * 60,
    "2h": 2 * 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
    "1w": 7 * 24 * 60 * 60,
    "1W": 7 * 24 * 60 * 60,
}


@dataclass(frozen=True)
class MTFFusionPolicy:
    enabled: bool = True
    base_timeframe: str = "15m"
    required_context_timeframes: tuple[str, ...] = ("1h", "2h", "4h", "1d")
    optional_context_timeframes: tuple[str, ...] = ("1w",)
    minimum_base_rows_per_market: int = 8000
    minimum_training_markets: int = 12
    maximum_context_features_per_timeframe: int = 36
    candidate_maximum_features: int = 256
    supervised_maximum_features: int = 128
    primary_horizon_hours: float = 4.0
    maximum_required_context_age_bars: float = 1.25
    minimum_selected_per_required_timeframe: int = 3
    minimum_selected_cross_timeframe: int = 4


def resolve_mtf_policy(config: Mapping[str, Any] | None = None) -> MTFFusionPolicy:
    raw = dict(config or {})
    return MTFFusionPolicy(
        enabled=bool(raw.get("enabled", True)),
        base_timeframe=str(raw.get("base_timeframe", "15m")),
        required_context_timeframes=tuple(
            str(v) for v in raw.get(
                "required_context_timeframes", ("1h", "2h", "4h", "1d")
            )
        ),
        optional_context_timeframes=tuple(
            str(v) for v in raw.get("optional_context_timeframes", ("1w",))
        ),
        minimum_base_rows_per_market=int(raw.get("minimum_base_rows_per_market", 8000)),
        minimum_training_markets=int(raw.get("minimum_training_markets", 12)),
        maximum_context_features_per_timeframe=int(
            raw.get("maximum_context_features_per_timeframe", 36)
        ),
        candidate_maximum_features=int(raw.get("candidate_maximum_features", 256)),
        supervised_maximum_features=int(raw.get("supervised_maximum_features", 128)),
        primary_horizon_hours=float(raw.get("primary_horizon_hours", 4.0)),
        maximum_required_context_age_bars=float(
            raw.get("maximum_required_context_age_bars", 1.25)
        ),
        minimum_selected_per_required_timeframe=int(
            raw.get("minimum_selected_per_required_timeframe", 3)
        ),
        minimum_selected_cross_timeframe=int(
            raw.get("minimum_selected_cross_timeframe", 4)
        ),
    )


OPERATIONAL_SUFFIXES = (
    "__age_bars",
    "__present",
    "__source_close_ns",
)


def is_operational_mtf_feature(name: str) -> bool:
    return str(name).lower().endswith(OPERATIONAL_SUFFIXES)


def _compact_fused_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.replace([np.inf, -np.inf], np.nan).copy()
    for name in output.columns:
        if pd.api.types.is_bool_dtype(output[name]):
            output[name] = output[name].astype(np.float32)
        elif pd.api.types.is_numeric_dtype(output[name]):
            output[name] = pd.to_numeric(
                output[name], errors="coerce"
            ).astype(np.float32, copy=False)
    return output.copy()



def timeframe_seconds(timeframe: str) -> int:
    token = str(timeframe)
    if token not in _TIMEFRAME_SECONDS:
        raise ValueError(f"unsupported timeframe: {token}")
    return int(_TIMEFRAME_SECONDS[token])


def horizon_bars_for_hours(timeframe: str, hours: float) -> int:
    return max(
        1,
        int(round(float(hours) * 3600.0 / float(timeframe_seconds(timeframe)))),
    )


def mtf_contract_hash(policy: MTFFusionPolicy) -> str:
    payload = {
        "version": MTF_FUSION_VERSION,
        "feature_source": MTF_FEATURE_SOURCE,
        "policy": asdict(policy),
        "alignment": "CLOSED_CANDLE_BACKWARD_ASOF",
        "execution_authority_changed": False,
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def fetch_multitimeframe_frames(
    bridge,
    markets: list[str] | tuple[str, ...],
    *,
    policy: MTFFusionPolicy,
    concurrency: int = 4,
) -> dict[str, dict[str, pd.DataFrame]]:
    output: dict[str, dict[str, pd.DataFrame]] = {}
    for timeframe in (
        policy.base_timeframe,
        *policy.required_context_timeframes,
        *policy.optional_context_timeframes,
    ):
        raw = bridge.ohlcv_many(
            list(markets),
            timeframe,
            persist=False,
            concurrency=int(concurrency),
        )
        output[str(timeframe)] = {
            str(market).upper(): frame
            for market, frame in raw.items()
            if frame is not None and not frame.empty
        }
    return output


def _decision_index(index: pd.Index, timeframe: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index) + pd.to_timedelta(
        timeframe_seconds(timeframe), unit="s"
    )


def align_closed_feature_frame(
    target_index: pd.Index,
    *,
    target_timeframe: str,
    source: pd.DataFrame,
    source_timeframe: str,
    prefix: str,
) -> pd.DataFrame:
    target_index = pd.DatetimeIndex(target_index)
    if source is None or source.empty:
        return pd.DataFrame(index=target_index)

    left = pd.DataFrame(
        {
            "__target_index": target_index,
            "__decision_at": _decision_index(target_index, target_timeframe),
        }
    ).sort_values("__decision_at")

    right = source.copy()
    right["__available_at"] = pd.DatetimeIndex(right.index) + pd.to_timedelta(
        timeframe_seconds(source_timeframe), unit="s"
    )
    right = right.reset_index(drop=True).sort_values("__available_at")

    merged = pd.merge_asof(
        left,
        right,
        left_on="__decision_at",
        right_on="__available_at",
        direction="backward",
        allow_exact_matches=True,
    )

    original_index = pd.DatetimeIndex(merged.pop("__target_index"))
    decision = pd.to_datetime(merged.pop("__decision_at"), utc=True, errors="coerce")
    available = pd.to_datetime(merged.pop("__available_at"), utc=True, errors="coerce")
    age = (decision - available).dt.total_seconds() / float(
        timeframe_seconds(source_timeframe)
    )

    merged.index = original_index
    merged.columns = [f"{prefix}{name}" for name in merged.columns]
    merged[f"{prefix}age_bars"] = age.to_numpy(float)
    merged[f"{prefix}present"] = available.notna().astype(float).to_numpy()
    return _compact_fused_frame(merged)


def _curated_context_columns(frame: pd.DataFrame, *, maximum: int) -> list[str]:
    preferred = (
        "rolling_return_7",
        "rolling_return_20",
        "distance_ema_20",
        "distance_ema_50",
        "ema_50_slope",
        "roc_12",
        "swing_trend",
        "fractal_trend_score",
        "fractal_range_position",
        "range_expansion_score",
        "trend_efficiency_20",
        "normalized_atr_14",
        "atr_percentile_100",
        "downside_volatility_20",
        "upside_volatility_20",
        "crypto_idx_rsi_14",
        "crypto_idx_adx_14",
        "crypto_idx_ppo",
        "crypto_idx_mfi_14",
        "crypto_idx_bollinger_position",
        "crypto_idx_supertrend_direction",
        "crypto_idx_vortex_spread_14",
        "crypto_vwap_distance_20",
        "crypto_vwap_slope_5",
        "crypto_vwap_reclaim",
        "crypto_vwap_reclaim_density_12",
        "crypto_strategy_entry_consensus",
        "crypto_strategy_entry_consensus_12",
        "crypto_strategy_exit_consensus",
        "crypto_tactical_entry_consensus",
        "crypto_tactical_exit_consensus",
        "pattern_price_action_confluence",
        "pattern_candle_balance",
        "pattern_return_8",
        "pattern_return_16",
        "pattern_drawup_16",
        "pattern_drawdown_16",
        "pattern_range_position_16",
    )
    selected = [name for name in preferred if name in frame.columns]
    if len(selected) >= int(maximum):
        return selected[: int(maximum)]

    for prefix in (
        "crypto_tactical_family_",
        "crypto_strategy_family_",
        "pattern_strategy_",
        "pattern_price_volume_corr_",
        "pattern_prior_breakdown_distance_",
        "pattern_prior_breakout_distance_",
        "htf_",
    ):
        for name in sorted(
            str(column) for column in frame.columns if str(column).startswith(prefix)
        ):
            if name not in selected:
                selected.append(name)
            if len(selected) >= int(maximum):
                return selected
    return selected[: int(maximum)]


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series | None:
    if name not in frame.columns:
        return None
    return pd.to_numeric(frame[name], errors="coerce")


def _trend_score(frame: pd.DataFrame) -> pd.Series:
    rows: list[pd.Series] = []
    for name, transform in (
        ("crypto_idx_supertrend_direction", lambda s: s.clip(-1.0, 1.0)),
        ("swing_trend", lambda s: s.clip(-1.0, 1.0)),
        ("crypto_idx_ppo", lambda s: np.tanh(20.0 * s)),
        ("crypto_idx_rsi_14", lambda s: s.clip(-1.0, 1.0)),
        ("distance_ema_50", lambda s: np.tanh(12.0 * s)),
    ):
        series = _numeric(frame, name)
        if series is not None:
            rows.append(transform(series))
    if not rows:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.concat(rows, axis=1).mean(axis=1).clip(-1.0, 1.0)


def _canonical_key(timeframe: str) -> str:
    return "1W" if str(timeframe).lower() == "1w" else str(timeframe)


def _native_features(
    bridge,
    *,
    market: str,
    timeframe: str,
    raw: pd.DataFrame,
    all_raw: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    higher = {
        _canonical_key(candidate): frame
        for candidate, frame in all_raw.items()
        if (
            candidate != timeframe
            and timeframe_seconds(candidate) > timeframe_seconds(timeframe)
            and frame is not None
            and not frame.empty
        )
    }
    return canonical_model_frame(
        bridge,
        raw,
        market=market,
        timeframe=timeframe,
        benchmark=None,
        higher_timeframes=higher,
    )


def _cross_timeframe_features(
    base: pd.DataFrame,
    aligned: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    out = pd.DataFrame(index=base.index)
    trend_rows: list[pd.Series] = [_trend_score(base).rename("15m")]
    rsi_rows: list[pd.Series] = []
    vwap_rows: list[pd.Series] = []
    strategy_rows: list[pd.Series] = []
    volatility_rows: list[pd.Series] = []

    base_rsi = _numeric(base, "crypto_idx_rsi_14")
    if base_rsi is not None:
        rsi_rows.append(base_rsi.rename("15m"))
    base_vwap = _numeric(base, "crypto_vwap_distance_20")
    if base_vwap is not None:
        vwap_rows.append(base_vwap.rename("15m"))
    for candidate in (
        "crypto_tactical_entry_consensus",
        "crypto_strategy_entry_consensus",
        "crypto_strategy_entry_consensus_12",
    ):
        series = _numeric(base, candidate)
        if series is not None:
            strategy_rows.append(series.rename("15m"))
            break
    for candidate in (
        "normalized_atr_14",
        "downside_volatility_20",
        "upside_volatility_20",
    ):
        series = _numeric(base, candidate)
        if series is not None:
            volatility_rows.append(series.rename("15m"))
            break

    for timeframe, frame in aligned.items():
        prefix = f"mtf_{timeframe}__"
        for collection, names in (
            (rsi_rows, (prefix + "crypto_idx_rsi_14",)),
            (vwap_rows, (prefix + "crypto_vwap_distance_20",)),
            (
                strategy_rows,
                (
                    prefix + "crypto_tactical_entry_consensus",
                    prefix + "crypto_strategy_entry_consensus",
                    prefix + "crypto_strategy_entry_consensus_12",
                ),
            ),
            (
                volatility_rows,
                (
                    prefix + "normalized_atr_14",
                    prefix + "downside_volatility_20",
                    prefix + "upside_volatility_20",
                ),
            ),
        ):
            for name in names:
                if name in frame.columns:
                    collection.append(
                        pd.to_numeric(frame[name], errors="coerce").rename(timeframe)
                    )
                    break
        trend_name = prefix + "trend_score"
        if trend_name in frame.columns:
            trend_rows.append(
                pd.to_numeric(frame[trend_name], errors="coerce").rename(timeframe)
            )

    trend = pd.concat(trend_rows, axis=1)
    out["mtf_cross__trend_mean"] = trend.mean(axis=1)
    out["mtf_cross__trend_dispersion"] = trend.std(axis=1, ddof=0)
    out["mtf_cross__bullish_fraction"] = trend.gt(0.0).mean(axis=1)
    out["mtf_cross__bearish_fraction"] = trend.lt(0.0).mean(axis=1)
    for higher in ("1h", "4h", "1d"):
        if "15m" in trend.columns and higher in trend.columns:
            out[f"mtf_cross__trend_spread_15m_{higher}"] = (
                trend["15m"] - trend[higher]
            )

    if rsi_rows:
        table = pd.concat(rsi_rows, axis=1)
        out["mtf_cross__rsi_mean"] = table.mean(axis=1)
        out["mtf_cross__rsi_dispersion"] = table.std(axis=1, ddof=0)
    if vwap_rows:
        table = pd.concat(vwap_rows, axis=1)
        out["mtf_cross__vwap_mean_distance"] = table.mean(axis=1)
        out["mtf_cross__vwap_above_fraction"] = table.gt(0.0).mean(axis=1)
    if strategy_rows:
        table = pd.concat(strategy_rows, axis=1)
        out["mtf_cross__entry_consensus"] = table.mean(axis=1)
        out["mtf_cross__entry_conflict"] = table.std(axis=1, ddof=0)
    if volatility_rows:
        table = pd.concat(volatility_rows, axis=1)
        out["mtf_cross__volatility_mean"] = table.mean(axis=1)
        out["mtf_cross__volatility_dispersion"] = table.std(axis=1, ddof=0)
    return out.replace([np.inf, -np.inf], np.nan)


def required_context_health(
    fused: pd.DataFrame,
    *,
    policy: MTFFusionPolicy,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    ready = bool(not fused.empty)
    for timeframe in policy.required_context_timeframes:
        present_name = f"mtf_{timeframe}__present"
        age_name = f"mtf_{timeframe}__age_bars"
        present = None
        age = None
        if not fused.empty and present_name in fused.columns:
            present = float(pd.to_numeric(fused[present_name], errors="coerce").iloc[-1])
        if not fused.empty and age_name in fused.columns:
            age = float(pd.to_numeric(fused[age_name], errors="coerce").iloc[-1])
        tf_ready = bool(
            present is not None
            and np.isfinite(present)
            and present >= 1.0
            and age is not None
            and np.isfinite(age)
            and age <= policy.maximum_required_context_age_bars
        )
        ready = ready and tf_ready
        rows[timeframe] = {"present": present, "age_bars": age, "ready": tf_ready}
    return {
        "ready": bool(ready),
        "required": rows,
        "maximum_required_context_age_bars": policy.maximum_required_context_age_bars,
    }


def build_market_mtf_feature_frame(
    bridge,
    *,
    market: str,
    frames: Mapping[str, pd.DataFrame],
    policy: MTFFusionPolicy,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    market = str(market).upper()
    base_raw = frames.get(policy.base_timeframe)
    if base_raw is None or base_raw.empty:
        raise ValueError(f"{market}: missing base timeframe {policy.base_timeframe}")
    missing = [
        tf
        for tf in policy.required_context_timeframes
        if frames.get(tf) is None or frames.get(tf).empty
    ]
    if missing:
        raise ValueError(f"{market}: missing required context {','.join(missing)}")

    native: dict[str, pd.DataFrame] = {}
    for timeframe in (
        policy.base_timeframe,
        *policy.required_context_timeframes,
        *policy.optional_context_timeframes,
    ):
        raw = frames.get(timeframe)
        if raw is None or raw.empty:
            continue
        native[timeframe] = _native_features(
            bridge,
            market=market,
            timeframe=timeframe,
            raw=raw,
            all_raw=frames,
        )

    base = native[policy.base_timeframe].copy()
    aligned: dict[str, pd.DataFrame] = {}
    for timeframe in (
        *policy.required_context_timeframes,
        *policy.optional_context_timeframes,
    ):
        source = native.get(timeframe)
        if source is None or source.empty:
            continue
        names = _curated_context_columns(
            source,
            maximum=policy.maximum_context_features_per_timeframe,
        )
        selected = source.reindex(columns=names).copy()
        selected["trend_score"] = _trend_score(source)
        aligned[timeframe] = align_closed_feature_frame(
            base.index,
            target_timeframe=policy.base_timeframe,
            source=selected,
            source_timeframe=timeframe,
            prefix=f"mtf_{timeframe}__",
        )

    cross = _cross_timeframe_features(base, aligned)
    fused = pd.concat([base, *aligned.values(), cross], axis=1)
    fused = fused.loc[:, ~fused.columns.duplicated(keep="last")]
    fused = _compact_fused_frame(fused)
    health = required_context_health(fused, policy=policy)
    audit = {
        "market": market,
        "version": MTF_FUSION_VERSION,
        "contract_hash": mtf_contract_hash(policy),
        "base_timeframe": policy.base_timeframe,
        "base_rows": int(len(base_raw)),
        "required_context_timeframes": list(policy.required_context_timeframes),
        "optional_context_timeframes": list(policy.optional_context_timeframes),
        "native_feature_counts": {
            timeframe: int(len(frame.columns)) for timeframe, frame in native.items()
        },
        "fused_feature_count": int(len(fused.columns)),
        "fused_memory_bytes": int(fused.memory_usage(deep=True).sum()),
        "dataframe_block_count": int(len(getattr(fused, "_mgr").blocks)),
        "runtime_health": health,
        "lookahead_safe": True,
        "closed_candles_only": True,
        "execution_authority_changed": False,
        "capital_caps_changed": False,
        "orders_submitted": 0,
    }
    fused.attrs.update(
        {
            "multitimeframe_fusion": True,
            "multitimeframe_version": MTF_FUSION_VERSION,
            "multitimeframe_contract_hash": mtf_contract_hash(policy),
            "feature_source": MTF_FEATURE_SOURCE,
            "lookahead_safe": True,
            "closed_candles_only": True,
        }
    )
    return fused, audit


def build_multitimeframe_feature_tables(
    bridge,
    frames_by_timeframe: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    policy: MTFFusionPolicy,
    training: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]:
    base_map = dict(frames_by_timeframe.get(policy.base_timeframe, {}))
    base_frames: dict[str, pd.DataFrame] = {}
    feature_tables: dict[str, pd.DataFrame] = {}
    market_audit: dict[str, Any] = {}

    for market, base in sorted(base_map.items()):
        market = str(market).upper()
        if training and len(base) < policy.minimum_base_rows_per_market:
            market_audit[market] = {
                "training_eligible": False,
                "reason": "INSUFFICIENT_BASE_HISTORY",
                "base_rows": int(len(base)),
            }
            continue
        frames = {
            timeframe: dict(frames_by_timeframe.get(timeframe, {})).get(market)
            for timeframe in (
                policy.base_timeframe,
                *policy.required_context_timeframes,
                *policy.optional_context_timeframes,
            )
        }
        missing = [
            tf
            for tf in policy.required_context_timeframes
            if frames.get(tf) is None or frames.get(tf).empty
        ]
        if missing:
            market_audit[market] = {
                "training_eligible": False,
                "reason": "MISSING_REQUIRED_CONTEXT",
                "missing_required": missing,
                "base_rows": int(len(base)),
            }
            continue
        try:
            fused, audit = build_market_mtf_feature_frame(
                bridge,
                market=market,
                frames=frames,
                policy=policy,
            )
        except Exception as exc:
            market_audit[market] = {
                "training_eligible": False,
                "reason": "FEATURE_BUILD_FAILED",
                "error": f"{type(exc).__name__}:{str(exc)[:300]}",
                "base_rows": int(len(base)),
            }
            continue
        base_frames[market] = base
        feature_tables[market] = fused
        market_audit[market] = {**audit, "training_eligible": True}

    eligible = sorted(feature_tables)
    if training and len(eligible) < policy.minimum_training_markets:
        raise ValueError(
            f"MTF training markets {len(eligible)} < {policy.minimum_training_markets}"
        )
    audit = {
        "version": MTF_FUSION_VERSION,
        "feature_source": MTF_FEATURE_SOURCE,
        "contract_hash": mtf_contract_hash(policy),
        "policy": asdict(policy),
        "training_eligible_markets": eligible,
        "training_eligible_market_count": len(eligible),
        "excluded_markets": sorted(set(market_audit) - set(eligible)),
        "market_audit": market_audit,
        "lookahead_safe": True,
        "closed_candles_only": True,
        "execution_authority_changed": False,
        "capital_caps_changed": False,
        "orders_submitted": 0,
    }
    return base_frames, feature_tables, audit


def selected_mtf_group_counts(features: list[str] | tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in features:
        token = str(name).lower()
        group = None
        for timeframe in ("1h", "2h", "4h", "1d", "1w"):
            if token.startswith(f"mtf_{timeframe}__") or token.startswith(
                f"htf_{timeframe}_"
            ):
                group = f"mtf_{timeframe}"
                break
        if token.startswith("mtf_cross__"):
            group = "mtf_cross"
        if group is not None:
            counts[group] = counts.get(group, 0) + 1
    return counts


def validate_selected_mtf_features(
    features: list[str] | tuple[str, ...],
    *,
    policy: MTFFusionPolicy,
) -> dict[str, int]:
    counts = selected_mtf_group_counts(features)
    failures: list[str] = []
    for timeframe in policy.required_context_timeframes:
        count = int(counts.get(f"mtf_{timeframe}", 0))
        if count < policy.minimum_selected_per_required_timeframe:
            failures.append(f"{timeframe}:{count}")
    cross = int(counts.get("mtf_cross", 0))
    if cross < policy.minimum_selected_cross_timeframe:
        failures.append(f"cross:{cross}")
    if failures:
        raise ValueError("MTF selected feature coverage failed: " + ",".join(failures))
    return counts


def build_runtime_mtf_feature_frame(
    bridge,
    *,
    market: str,
    policy: MTFFusionPolicy,
    concurrency: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    market = str(market).upper()
    frames_by_timeframe = fetch_multitimeframe_frames(
        bridge,
        [market],
        policy=policy,
        concurrency=concurrency,
    )
    frames = {
        timeframe: dict(frames_by_timeframe.get(timeframe, {})).get(market)
        for timeframe in (
            policy.base_timeframe,
            *policy.required_context_timeframes,
            *policy.optional_context_timeframes,
        )
    }
    fused, audit = build_market_mtf_feature_frame(
        bridge,
        market=market,
        frames=frames,
        policy=policy,
    )
    health = required_context_health(fused, policy=policy)
    audit["runtime_health"] = health
    if not health["ready"]:
        raise RuntimeError(
            "MTF_REQUIRED_CONTEXT_NOT_READY:"
            + json.dumps(health, sort_keys=True, default=str)
        )
    return fused, audit


__all__ = [
    "MTF_FEATURE_SOURCE",
    "MTF_FUSION_VERSION",
    "MTFFusionPolicy",
    "align_closed_feature_frame",
    "build_market_mtf_feature_frame",
    "build_multitimeframe_feature_tables",
    "build_runtime_mtf_feature_frame",
    "fetch_multitimeframe_frames",
    "horizon_bars_for_hours",
    "is_operational_mtf_feature",
    "mtf_contract_hash",
    "required_context_health",
    "resolve_mtf_policy",
    "selected_mtf_group_counts",
    "timeframe_seconds",
    "validate_selected_mtf_features",
]
