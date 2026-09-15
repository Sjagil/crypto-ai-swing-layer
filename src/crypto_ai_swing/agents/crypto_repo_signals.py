from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


STRATEGY_PREFIX = "crypto_strategy_"

PROSPECTIVE_CONTEXT_FEATURES = (
    "macro_btc_dominance",
    "macro_eth_dominance",
    "macro_fear_greed_scaled",
    "macro_vix_scaled",
    "macro_nasdaq_100_return_5d",
    "macro_sp500_return_5d",
    "macro_altcoin_breadth",
    "macro_stablecoin_liquidity_score",
    "macro_usdt_market_cap_change_1h",
    "macro_usdt_market_cap_change_24h",
    "macro_usdc_market_cap_change_1h",
    "macro_usdc_market_cap_change_24h",
    "macro_aggregate_funding_bps",
    "macro_perpetual_premium_bps",
    "macro_open_interest_log1p",
    "macro_total_crypto_market_cap_log1p",
    "context_vwap_distance",
    "context_anchored_vwap_distance",
    "context_vwap_reclaim",
    "context_anchored_vwap_reclaim",
    "strategy_family_trend",
    "strategy_family_breakout",
    "strategy_family_vwap",
    "strategy_family_reversion",
    "strategy_family_relative_strength",
    "strategy_family_liquidity_recovery",
    "strategy_family_volume",
    "strategy_family_momentum",
    "context_pattern_trend",
    "context_pattern_breakout",
    "context_pattern_pullback",
    "context_pattern_reversal",
    "context_pattern_range",
    "context_pattern_vwap",
    "context_pattern_strategy_confidence",
    "context_pattern_confluence",
)


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        selected = float(value)
    except (TypeError, ValueError):
        return default
    return selected if np.isfinite(selected) else default


def _deep_get(context: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        if path in context:
            value = context.get(path)
            if value is not None:
                return value
        current: Any = context
        ok = True
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                ok = False
                break
            current = current.get(part)
        if ok and current is not None:
            return current
    return None


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    den = pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
    num = pd.to_numeric(numerator, errors="coerce")
    return num / den


def _as_float_bool(series: pd.Series, index: pd.Index) -> pd.Series:
    return (
        series.reindex(index)
        .fillna(False)
        .astype(bool)
        .astype(float)
    )


def _rolling_signal_density(series: pd.Series, window: int = 12) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return numeric.rolling(window, min_periods=1).mean()


def _bars_since_signal(series: pd.Series) -> pd.Series:
    values = series.fillna(False).astype(bool).to_numpy()
    result = np.full(len(values), np.nan, dtype=float)
    age = None
    for idx, active in enumerate(values):
        if active:
            age = 0
        elif age is not None:
            age += 1
        if age is not None:
            result[idx] = float(age)
    return pd.Series(result, index=series.index)


def _recency_score(series: pd.Series) -> pd.Series:
    age = _bars_since_signal(series)
    return 1.0 / (1.0 + age)


def _technical_index_features(features: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=features.index)

    close = pd.to_numeric(features.get("close"), errors="coerce")
    atr = pd.to_numeric(features.get("atr_14"), errors="coerce")
    vwap = pd.to_numeric(features.get("vwap_20"), errors="coerce")
    anchored = pd.to_numeric(features.get("anchored_vwap"), errors="coerce")

    if "vwap_20" in features:
        out["crypto_vwap_distance_20"] = _safe_ratio(close, vwap) - 1.0
        out["crypto_vwap_slope_5"] = vwap.pct_change(5, fill_method=None)
    if "anchored_vwap" in features:
        out["crypto_anchored_vwap_distance"] = _safe_ratio(close, anchored) - 1.0
        out["crypto_anchored_vwap_slope_5"] = anchored.pct_change(
            5,
            fill_method=None,
        )
    if "vwap_20" in features and "anchored_vwap" in features:
        out["crypto_vwap_anchored_spread"] = _safe_ratio(vwap, anchored) - 1.0
    if "vwap_reclaim" in features:
        out["crypto_vwap_reclaim"] = _as_float_bool(
            features["vwap_reclaim"],
            features.index,
        )
        out["crypto_vwap_reclaim_density_12"] = _rolling_signal_density(
            out["crypto_vwap_reclaim"]
        )
        out["crypto_vwap_reclaim_recency"] = _recency_score(
            features["vwap_reclaim"]
        )
    if "anchored_vwap_reclaim" in features:
        out["crypto_anchored_vwap_reclaim"] = _as_float_bool(
            features["anchored_vwap_reclaim"],
            features.index,
        )
        out["crypto_anchored_vwap_reclaim_density_12"] = _rolling_signal_density(
            out["crypto_anchored_vwap_reclaim"]
        )
        out["crypto_anchored_vwap_reclaim_recency"] = _recency_score(
            features["anchored_vwap_reclaim"]
        )

    if "rsi_14" in features:
        out["crypto_idx_rsi_14"] = (
            pd.to_numeric(features["rsi_14"], errors="coerce") - 50.0
        ) / 50.0
    if "mfi_14" in features:
        out["crypto_idx_mfi_14"] = (
            pd.to_numeric(features["mfi_14"], errors="coerce") - 50.0
        ) / 50.0
    if "williams_r_14" in features:
        out["crypto_idx_williams_r_14"] = (
            pd.to_numeric(features["williams_r_14"], errors="coerce") + 50.0
        ) / 50.0
    if "cci_20" in features:
        out["crypto_idx_cci_20"] = np.tanh(
            pd.to_numeric(features["cci_20"], errors="coerce") / 100.0
        )
    if "adx_14" in features:
        out["crypto_idx_adx_14"] = (
            pd.to_numeric(features["adx_14"], errors="coerce") / 100.0
        )
    if "aroon_up_25" in features and "aroon_down_25" in features:
        out["crypto_idx_aroon_spread_25"] = (
            pd.to_numeric(features["aroon_up_25"], errors="coerce")
            - pd.to_numeric(features["aroon_down_25"], errors="coerce")
        ) / 100.0
    if "vortex_plus_14" in features and "vortex_minus_14" in features:
        out["crypto_idx_vortex_spread_14"] = (
            pd.to_numeric(features["vortex_plus_14"], errors="coerce")
            - pd.to_numeric(features["vortex_minus_14"], errors="coerce")
        )
    if "macd" in features and "atr_14" in features:
        out["crypto_idx_macd_atr"] = _safe_ratio(
            pd.to_numeric(features["macd"], errors="coerce"),
            atr,
        )
    if "ppo" in features:
        out["crypto_idx_ppo"] = (
            pd.to_numeric(features["ppo"], errors="coerce") / 100.0
        )
    if (
        "bollinger_lower" in features
        and "bollinger_upper" in features
    ):
        lower = pd.to_numeric(features["bollinger_lower"], errors="coerce")
        upper = pd.to_numeric(features["bollinger_upper"], errors="coerce")
        out["crypto_idx_bollinger_position"] = (
            (close - lower) / (upper - lower).replace(0.0, np.nan)
        )
    if "choppiness_14" in features:
        out["crypto_idx_choppiness_14"] = (
            pd.to_numeric(features["choppiness_14"], errors="coerce") / 100.0
        )
    if "trend_efficiency_20" in features:
        out["crypto_idx_trend_efficiency_20"] = pd.to_numeric(
            features["trend_efficiency_20"],
            errors="coerce",
        )
    if "supertrend_direction" in features:
        out["crypto_idx_supertrend_direction"] = pd.to_numeric(
            features["supertrend_direction"],
            errors="coerce",
        )

    return out.replace([np.inf, -np.inf], np.nan)


def _strategy_features(
    bridge: Any,
    raw_ohlcv: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = pd.DataFrame(index=features.index)
    report: dict[str, Any] = {
        "authority": "RESEARCH_FEATURES_ONLY",
        "orders_generated": 0,
        "orders_submitted": 0,
        "registered": 0,
        "integrated": [],
        "skipped_intelligence": [],
        "skipped_higher_timeframe": [],
        "failed": {},
        "tactical_catalogue": {},
    }

    strategy_frame = features.copy()
    for column in ("open", "high", "low", "close", "volume"):
        if column not in strategy_frame and column in raw_ohlcv:
            strategy_frame[column] = pd.to_numeric(
                raw_ohlcv[column],
                errors="coerce",
            ).reindex(strategy_frame.index)

    try:
        module = bridge.import_module("research.strategies")
        registry = dict(module.strategy_registry())
    except Exception as exc:
        report["failed"]["registry"] = (
            f"{type(exc).__name__}:{str(exc)[:300]}"
        )
        return out, report

    report["registered"] = len(registry)
    family_entries: dict[str, list[pd.Series]] = defaultdict(list)
    family_exits: dict[str, list[pd.Series]] = defaultdict(list)
    all_entries: list[pd.Series] = []
    all_exits: list[pd.Series] = []

    for strategy_id, strategy in sorted(registry.items()):
        sid = str(strategy_id).lower()
        family = str(getattr(strategy, "family", "unknown")).lower()
        if bool(getattr(strategy, "uses_intelligence", False)):
            report["skipped_intelligence"].append(sid)
            continue
        required_htf = tuple(
            getattr(strategy, "required_higher_timeframes", ()) or ()
        )
        if required_htf:
            report["skipped_higher_timeframe"].append(
                {
                    "strategy_id": sid,
                    "required_higher_timeframes": list(required_htf),
                }
            )
            continue
        try:
            generated = strategy.generate(strategy_frame)
        except Exception as exc:
            report["failed"][sid] = (
                f"{type(exc).__name__}:{str(exc)[:300]}"
            )
            continue

        entry = _as_float_bool(generated.entry, strategy_frame.index)
        exit_signal = _as_float_bool(generated.exit, strategy_frame.index)
        reduce_signal = _as_float_bool(generated.reduce, strategy_frame.index)

        base = f"{STRATEGY_PREFIX}{sid}"
        out[f"{base}_entry"] = entry
        out[f"{base}_exit"] = exit_signal
        out[f"{base}_reduce"] = reduce_signal
        out[f"{base}_entry_density_12"] = _rolling_signal_density(entry)
        out[f"{base}_exit_density_12"] = _rolling_signal_density(exit_signal)
        out[f"{base}_entry_recency"] = _recency_score(
            generated.entry.reindex(strategy_frame.index)
        )

        family_entries[family].append(entry)
        family_exits[family].append(exit_signal)
        all_entries.append(entry)
        all_exits.append(exit_signal)
        report["integrated"].append(sid)

    def stack_mean(rows: list[pd.Series]) -> pd.Series:
        if not rows:
            return pd.Series(np.nan, index=features.index)
        return pd.concat(rows, axis=1).mean(axis=1)

    for family in sorted(family_entries):
        slug = family.replace("-", "_").replace(" ", "_")
        out[f"crypto_strategy_family_{slug}_entry_share"] = stack_mean(
            family_entries[family]
        )
        out[f"crypto_strategy_family_{slug}_exit_share"] = stack_mean(
            family_exits[family]
        )

    if all_entries:
        entry_matrix = pd.concat(all_entries, axis=1)
        exit_matrix = pd.concat(all_exits, axis=1)
        out["crypto_strategy_entry_consensus"] = entry_matrix.mean(axis=1)
        out["crypto_strategy_exit_consensus"] = exit_matrix.mean(axis=1)
        out["crypto_strategy_entry_count"] = entry_matrix.sum(axis=1)
        out["crypto_strategy_exit_count"] = exit_matrix.sum(axis=1)
        out["crypto_strategy_entry_consensus_12"] = (
            out["crypto_strategy_entry_consensus"]
            .rolling(12, min_periods=1)
            .mean()
        )

    try:
        tactical = bridge.import_module("research.tactical_multitimeframe")
        specs = tuple(tactical.tactical_strategy_specs())
        report["tactical_catalogue"] = {
            "registered": len(specs),
            "families": sorted(
                {
                    str(getattr(spec, "family", ""))
                    for spec in specs
                    if str(getattr(spec, "family", ""))
                }
            ),
            "timeframes": sorted(
                {
                    str(getattr(spec, "timeframe", ""))
                    for spec in specs
                    if str(getattr(spec, "timeframe", ""))
                }
            ),
            "authority": "CATALOGUE_ONLY_NO_EXECUTION_AUTHORITY",
        }
    except Exception as exc:
        report["tactical_catalogue"] = {
            "status": "UNAVAILABLE",
            "error": f"{type(exc).__name__}:{str(exc)[:300]}",
        }

    return out.replace([np.inf, -np.inf], np.nan), report


def augment_canonical_model_features(
    bridge: Any,
    raw_ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    *,
    market: str,
    timeframe: str,
) -> pd.DataFrame:
    """Add causal VWAP/index/strategy features from the pinned Sjagil/crypto repo.

    This is a research feature bridge only. It does not call any execution
    interface, create orders, alter authority, or change capital limits.
    """

    technical = _technical_index_features(features)
    strategies, report = _strategy_features(
        bridge,
        raw_ohlcv,
        features,
    )

    parts = [features, technical, strategies]
    result = pd.concat(parts, axis=1)
    result = result.loc[:, ~result.columns.duplicated(keep="last")]
    result.attrs.update(features.attrs)
    result.attrs.update(
        {
            "crypto_repo_signal_bridge": True,
            "crypto_repo_signal_bridge_version": "round47d_v1",
            "crypto_repo_signal_bridge_market": str(market).upper(),
            "crypto_repo_signal_bridge_timeframe": str(timeframe),
            "strategy_feature_report": report,
            "orders_generated": 0,
            "orders_submitted": 0,
            "execution_authority_changed": False,
            "capital_caps_changed": False,
        }
    )
    return result


def prospective_context_feature_vector(
    context: Mapping[str, Any],
) -> dict[str, float | None]:
    """Causal PIT macro/index/VWAP/strategy descriptors for forward research."""

    btc_dom = _finite(
        _deep_get(
            context,
            "macro.features.btc_dominance",
            "btc_dominance",
        ),
        None,
    )
    eth_dom = _finite(
        _deep_get(
            context,
            "macro.features.eth_dominance",
            "eth_dominance",
        ),
        None,
    )
    fear = _finite(
        _deep_get(
            context,
            "macro.features.fear_greed",
            "fear_greed",
        ),
        None,
    )
    vix = _finite(
        _deep_get(
            context,
            "macro.features.vix",
            "vix",
        ),
        None,
    )
    ndx_ret = _finite(
        _deep_get(
            context,
            "macro.features.nasdaq_100_return_5d",
            "nasdaq_100_return_5d",
        ),
        None,
    )
    spx_ret = _finite(
        _deep_get(
            context,
            "macro.features.sp500_return_5d",
            "sp500_return_5d",
        ),
        None,
    )
    breadth = _finite(
        _deep_get(
            context,
            "macro.features.altcoin_breadth",
            "market_opportunity_intensity.breadth.1h",
            "altcoin_breadth",
        ),
        None,
    )
    stable_score = _finite(
        _deep_get(
            context,
            "macro.features.stablecoin_liquidity_score",
            "stablecoin_liquidity_score",
        ),
        None,
    )
    usdt_1h = _finite(
        _deep_get(
            context,
            "macro.features.usdt_market_cap_change_1h",
            "usdt_market_cap_change_1h",
        ),
        None,
    )
    usdt_24h = _finite(
        _deep_get(
            context,
            "macro.features.usdt_market_cap_change_24h",
            "usdt_market_cap_change_24h",
        ),
        None,
    )
    usdc_1h = _finite(
        _deep_get(
            context,
            "macro.features.usdc_market_cap_change_1h",
            "usdc_market_cap_change_1h",
        ),
        None,
    )
    usdc_24h = _finite(
        _deep_get(
            context,
            "macro.features.usdc_market_cap_change_24h",
            "usdc_market_cap_change_24h",
        ),
        None,
    )
    funding = _finite(
        _deep_get(
            context,
            "macro.features.aggregate_funding_proxy",
            "aggregate_funding_proxy",
        ),
        None,
    )
    premium = _finite(
        _deep_get(
            context,
            "macro.features.perpetual_premium",
            "perpetual_premium",
        ),
        None,
    )
    oi = _finite(
        _deep_get(
            context,
            "macro.features.open_interest_proxy",
            "open_interest_proxy",
        ),
        None,
    )
    market_cap = _finite(
        _deep_get(
            context,
            "macro.features.total_crypto_market_cap",
            "total_crypto_market_cap",
        ),
        None,
    )

    vwap_distance = _finite(
        _deep_get(
            context,
            "universe_screen.technical.vwap_distance",
            "universe_screen.technical.distance_vwap_20",
            "technical.vwap_distance",
            "technical.distance_vwap_20",
        ),
        None,
    )
    anchored_distance = _finite(
        _deep_get(
            context,
            "universe_screen.technical.anchored_vwap_distance",
            "universe_screen.technical.distance_anchored_vwap",
            "technical.anchored_vwap_distance",
            "technical.distance_anchored_vwap",
        ),
        None,
    )
    vwap_reclaim = _deep_get(
        context,
        "universe_screen.technical.vwap_reclaim",
        "technical.vwap_reclaim",
    )
    anchored_reclaim = _deep_get(
        context,
        "universe_screen.technical.anchored_vwap_reclaim",
        "technical.anchored_vwap_reclaim",
    )

    family = str(
        _deep_get(
            context,
            "mtf_challenger.strategy_family",
            "strategy_family",
        )
        or ""
    ).upper()

    def family_has(*tokens: str) -> float:
        return float(any(token in family for token in tokens))

    technical_score = _finite(
        _deep_get(
            context,
            "universe_screen.execution_adjusted_score",
            "technical.technical_score",
        ),
        None,
    )
    technical_unit = (
        float(1.0 / (1.0 + np.exp(-np.clip(technical_score, -20.0, 20.0))))
        if technical_score is not None
        else None
    )
    diagnostics = dict(_deep_get(context, "mtf_challenger.diagnostics") or {})
    mtf_alignment = _finite(
        _deep_get(context, "mtf_challenger.alignment_score"),
        None,
    )
    mtf_execution = _finite(
        _deep_get(context, "mtf_challenger.execution_score"),
        None,
    )
    mtf_trend = _finite(diagnostics.get("trend"), None)
    mtf_setup = _finite(diagnostics.get("setup"), None)
    mtf_trigger = _finite(diagnostics.get("trigger"), None)
    breakout_state = str(
        _deep_get(
            context,
            "universe_screen.technical.breakout_state",
            "technical.breakout_state",
        )
        or ""
    ).upper()

    def mean01(*values: float | None) -> float | None:
        available = [
            float(np.clip(value, 0.0, 1.0))
            for value in values
            if value is not None and np.isfinite(value)
        ]
        return float(np.mean(available)) if available else None

    family_trend = family_has("TREND", "CONTINUATION")
    family_breakout = family_has("BREAKOUT", "DONCHIAN", "EXPANSION")
    family_pullback = family_has("PULLBACK", "RETEST")
    family_reversal = family_has(
        "REVERSION",
        "RECOVERY",
        "LIQUIDITY",
        "FAILED_BREAK",
        "REVERSAL",
    )
    family_range = family_has("RANGE", "REVERSION")
    family_vwap = family_has("VWAP")

    state_breakout = float(
        any(
            token in breakout_state
            for token in ("BREAKOUT", "DONCHIAN", "TREND_CONTINUATION")
        )
    )
    state_pullback = float(
        any(token in breakout_state for token in ("PULLBACK", "RETEST"))
    )
    state_reversal = float(
        any(token in breakout_state for token in ("REVERS", "FAILED", "RECOVERY"))
    )
    state_range = float("RANGE" in breakout_state)
    vwap_state = mean01(
        family_vwap,
        float(bool(vwap_reclaim)) if vwap_reclaim is not None else None,
        float(bool(anchored_reclaim)) if anchored_reclaim is not None else None,
    )

    context_pattern_trend = mean01(
        family_trend,
        mtf_trend,
        mtf_alignment,
        technical_unit,
    )
    context_pattern_breakout = mean01(
        family_breakout,
        state_breakout,
        mtf_setup,
        mtf_trigger,
    )
    context_pattern_pullback = mean01(
        family_pullback,
        state_pullback,
        mtf_setup,
    )
    context_pattern_reversal = mean01(
        family_reversal,
        state_reversal,
        1.0 - mtf_trend if mtf_trend is not None else None,
    )
    context_pattern_range = mean01(
        family_range,
        state_range,
        1.0 - mtf_trend if mtf_trend is not None else None,
    )
    context_pattern_strategy_confidence = mean01(
        mtf_setup,
        mtf_trigger,
        mtf_execution,
        mtf_alignment,
    )
    context_pattern_confluence = mean01(
        context_pattern_trend,
        context_pattern_breakout,
        context_pattern_pullback,
        context_pattern_reversal,
        context_pattern_range,
        vwap_state,
        context_pattern_strategy_confidence,
        technical_unit,
    )

    return {
        "macro_btc_dominance": btc_dom,
        "macro_eth_dominance": eth_dom,
        "macro_fear_greed_scaled": (
            fear / 100.0 if fear is not None else None
        ),
        "macro_vix_scaled": (
            vix / 100.0 if vix is not None else None
        ),
        "macro_nasdaq_100_return_5d": ndx_ret,
        "macro_sp500_return_5d": spx_ret,
        "macro_altcoin_breadth": breadth,
        "macro_stablecoin_liquidity_score": stable_score,
        "macro_usdt_market_cap_change_1h": usdt_1h,
        "macro_usdt_market_cap_change_24h": usdt_24h,
        "macro_usdc_market_cap_change_1h": usdc_1h,
        "macro_usdc_market_cap_change_24h": usdc_24h,
        "macro_aggregate_funding_bps": (
            funding * 10_000.0 if funding is not None else None
        ),
        "macro_perpetual_premium_bps": (
            premium * 10_000.0 if premium is not None else None
        ),
        "macro_open_interest_log1p": (
            float(np.log1p(max(0.0, oi))) if oi is not None else None
        ),
        "macro_total_crypto_market_cap_log1p": (
            float(np.log1p(max(0.0, market_cap)))
            if market_cap is not None
            else None
        ),
        "context_vwap_distance": vwap_distance,
        "context_anchored_vwap_distance": anchored_distance,
        "context_vwap_reclaim": (
            float(bool(vwap_reclaim))
            if vwap_reclaim is not None
            else None
        ),
        "context_anchored_vwap_reclaim": (
            float(bool(anchored_reclaim))
            if anchored_reclaim is not None
            else None
        ),
        "strategy_family_trend": family_has("TREND"),
        "strategy_family_breakout": family_has("BREAKOUT", "DONCHIAN"),
        "strategy_family_vwap": family_has("VWAP"),
        "strategy_family_reversion": family_has("REVERSION", "RANGE"),
        "strategy_family_relative_strength": family_has(
            "RELATIVE_STRENGTH",
            "RS_",
            "ROTATION",
        ),
        "strategy_family_liquidity_recovery": family_has(
            "LIQUIDITY",
            "RECOVERY",
            "FAILED_BREAK",
        ),
        "strategy_family_volume": family_has("VOLUME"),
        "strategy_family_momentum": family_has("MOMENTUM"),
        "context_pattern_trend": context_pattern_trend,
        "context_pattern_breakout": context_pattern_breakout,
        "context_pattern_pullback": context_pattern_pullback,
        "context_pattern_reversal": context_pattern_reversal,
        "context_pattern_range": context_pattern_range,
        "context_pattern_vwap": vwap_state,
        "context_pattern_strategy_confidence": context_pattern_strategy_confidence,
        "context_pattern_confluence": context_pattern_confluence,
    }


__all__ = [
    "PROSPECTIVE_CONTEXT_FEATURES",
    "STRATEGY_PREFIX",
    "augment_canonical_model_features",
    "prospective_context_feature_vector",
]
