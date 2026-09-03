from __future__ import annotations

import inspect
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge

NATIVE_FOUNDATION_INTERFACES: dict[str, tuple[str, ...]] = {
    "core.live_universe": ("candle_health",),
    "data.feature_store": (
        "FeatureStorePolicy",
        "build_feature_tensors",
        "STRICT_PORTFOLIO_MARKETS",
    ),
    "ml.lifecycle": ("audit_point_in_time_features",),
    "ml.registry": (
        "ImmutableDatasetRegistry",
        "ModelRegistry",
        "evaluate_model_promotion",
    ),
    "research.portfolio_selection": ("RotationPortfolioPolicy",),
    "research.regime_router": (),
    "research.residual_momentum": (
        "residual_momentum_parameter_set",
        "backtest_residual_momentum",
    ),
    "research.residual_reversal": (
        "residual_reversal_parameter_set",
        "backtest_residual_reversal",
    ),
    "research.multi_horizon_trend": (),
    "research.absolute_momentum": (),
    "research.multi_alpha_ensemble": (),
    "research.multi_alpha_ensemble_v2": (),
    "research.optimization": (),
    "research.statistical_evidence": (),
    "research.stochastic_validation": (),
    "research.trading_math": (
        "calculate_position_size",
        "calculate_position_size_from_stop_fraction",
    ),
    "risk.correlation_analyzer": ("CorrelationAnalyzer",),
    "risk.risk_manager": ("RiskManager", "OperationalDegradation"),
    "core.live_asset_preflight": ("live_account_health",),
    "core.execution_authority": (),
}

NATIVE_ALPHA_CAMPAIGNS: dict[str, tuple[str, str]] = {
    "residual-momentum": (
        "research.residual_momentum_campaign",
        "run_residual_momentum_campaign",
    ),
    "residual-reversal": (
        "research.residual_reversal_campaign",
        "run_residual_reversal_campaign",
    ),
    "peer-residual-reversal": (
        "research.peer_residual_reversal_campaign",
        "run_peer_residual_reversal_campaign",
    ),
    "multi-horizon-trend": (
        "research.multi_horizon_trend_campaign",
        "run_multi_horizon_trend_campaign",
    ),
    "multi-alpha-v2": (
        "research.multi_alpha_ensemble_v2_campaign",
        "run_multi_alpha_ensemble_v2_campaign",
    ),
}


@dataclass(frozen=True)
class NativeFoundationStatus:
    ready: bool
    modules: tuple[dict[str, Any], ...]
    imported_modules: int
    required_modules: int


class NativeFoundationBridge:
    """Deep read/research adapter into the canonical Sjagil/crypto stack."""

    def __init__(self, crypto_repo_root: Path):
        self.crypto = CryptoLibraryBridge(crypto_repo_root)

    def status(self) -> NativeFoundationStatus:
        rows: list[dict[str, Any]] = []
        for module_name, required in NATIVE_FOUNDATION_INTERFACES.items():
            row: dict[str, Any] = {
                "module": module_name,
                "required_interfaces": list(required),
                "available_interfaces": [],
                "missing_interfaces": [],
                "ready": False,
            }
            try:
                module = self.crypto.import_module(module_name)
                available = [name for name in required if hasattr(module, name)]
                missing = [name for name in required if not hasattr(module, name)]
                row.update(
                    {
                        "available_interfaces": available,
                        "missing_interfaces": missing,
                        "ready": not missing,
                        "file": str(getattr(module, "__file__", "") or ""),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
            rows.append(row)

        return NativeFoundationStatus(
            ready=all(bool(row["ready"]) for row in rows),
            modules=tuple(rows),
            imported_modules=sum(bool(row["ready"]) for row in rows),
            required_modules=len(rows),
        )

    def candle_health(
        self,
        *,
        markets: Iterable[str],
        timeframes: Iterable[str],
    ) -> dict[str, Any]:
        module = self.crypto.import_module("core.live_universe")
        fn = getattr(module, "candle_health", None)
        if not callable(fn):
            raise RuntimeError("core.live_universe.candle_health unavailable")  # noqa: TRY004

        normalized_timeframes = tuple(
            "1W" if str(tf).lower() == "1w" else str(tf)
            for tf in timeframes
        )
        payload = fn(
            self.crypto.settings(),
            markets=tuple(str(x).upper() for x in markets),
            timeframes=normalized_timeframes,
            write_artifact=False,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("native candle_health returned non-dict")  # noqa: TRY004

        return {
            **payload,
            "source": "Sjagil/crypto:core.live_universe.candle_health",
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    def pit_feature_store_certification(self) -> dict[str, Any]:
        module = self.crypto.import_module("data.feature_store")
        strict_markets = tuple(module.STRICT_PORTFOLIO_MARKETS)
        policy_cls = module.FeatureStorePolicy
        build = module.build_feature_tensors

        frames = {
            market: self.crypto.ohlcv(market, "1d", persist=False)
            for market in strict_markets
        }
        bundle = build(frames, policy=policy_cls())
        manifest = dict(getattr(bundle, "manifest", {}) or {})

        return {
            "schema_version": "crypto_ai_swing_pit_foundation_cert_v1",
            "status": "PASS",
            "source": "Sjagil/crypto:data.feature_store",
            "markets": list(strict_markets),
            "dataset_id": manifest.get("dataset_id"),
            "causality": manifest.get("causality", {}),
            "frequency": manifest.get("frequency"),
            "feature_rows": int(bundle.feature_mask.sum()),
            "target_rows": int(bundle.target_mask.sum()),
            "feature_names": list(manifest.get("feature_names") or []),
            "target_names": list(manifest.get("target_names") or []),
            "manifest": manifest,
            "live_decision_influence": False,
            "automatic_live_promotion": False,
            "orders_submitted": 0,
        }

    def alpha_catalog(self) -> dict[str, Any]:
        campaigns: list[dict[str, Any]] = []
        for name, spec in NATIVE_ALPHA_CAMPAIGNS.items():
            module_name, function_name = spec
            row: dict[str, Any] = {
                "name": name,
                "module": module_name,
                "function": function_name,
                "ready": False,
            }
            try:
                module = self.crypto.import_module(module_name)
                fn = getattr(module, function_name, None)
                row["ready"] = callable(fn)
                row["file"] = str(getattr(module, "__file__", "") or "")
                if callable(fn):
                    row["signature"] = str(inspect.signature(fn))
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
            campaigns.append(row)

        return {
            "schema_version": "crypto_ai_swing_native_alpha_catalog_v1",
            "campaigns": campaigns,
            "ready": all(bool(row["ready"]) for row in campaigns),
            "authority": "RESEARCH_ONLY",
            "automatic_live_promotion": False,
            "orders_submitted": 0,
        }

    @staticmethod
    def _first_value(
        payload: dict[str, Any],
        keys: tuple[str, ...],
    ) -> Any:
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        for value in payload.values():
            if isinstance(value, dict):
                found = NativeFoundationBridge._first_value(value, keys)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _list_count(
        payload: dict[str, Any],
        keys: tuple[str, ...],
    ) -> int:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
        for value in payload.values():
            if isinstance(value, dict):
                count = NativeFoundationBridge._list_count(value, keys)
                if count:
                    return count
        return 0

    @staticmethod
    def _forward_evidence_digest(
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        raw = payload.get("forward_summaries")
        if not isinstance(raw, dict) or not raw:
            return None

        rows = {
            str(name): dict(value)
            for name, value in raw.items()
            if isinstance(value, dict)
        }
        if not rows:
            return None

        def numeric(value: Any) -> float | None:
            try:
                result = float(value)
            except (TypeError, ValueError):
                return None
            return result

        def integer(value: Any) -> int:
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        statuses = sorted(
            {
                str(row.get("status"))
                for row in rows.values()
                if row.get("status") is not None
            }
        )
        closed = [
            integer(row.get("closed_daily_observations"))
            for row in rows.values()
        ]
        required_closed = [
            integer(row.get("required_closed_daily_observations"))
            for row in rows.values()
        ]
        rebalances = [
            integer(row.get("forward_rebalances"))
            for row in rows.values()
        ]
        required_rebalances = [
            integer(row.get("required_forward_rebalances"))
            for row in rows.values()
        ]
        high_volatility = []
        for row in rows.values():
            regime = dict(row.get("regime_coverage") or {})
            counts = dict(regime.get("counts") or {})
            volatility = dict(counts.get("volatility") or {})
            high_volatility.append(integer(volatility.get("HIGH")))

        returns = [
            (name, numeric(row.get("forward_net_return")))
            for name, row in rows.items()
        ]
        returns = [
            (name, value)
            for name, value in returns
            if value is not None
        ]
        best = max(returns, key=lambda item: item[1]) if returns else None
        worst = min(returns, key=lambda item: item[1]) if returns else None
        formal_flags = [
            bool(row.get("formal_performance_gates_evaluated"))
            for row in rows.values()
        ]

        return {
            "candidate_count": len(rows),
            "statuses": statuses,
            "maximum_closed_daily_observations": max(closed, default=0),
            "required_closed_daily_observations": max(
                required_closed,
                default=0,
            ),
            "maximum_forward_rebalances": max(rebalances, default=0),
            "required_forward_rebalances": max(
                required_rebalances,
                default=0,
            ),
            "maximum_high_volatility_observations": max(
                high_volatility,
                default=0,
            ),
            "formal_performance_gates_evaluated_for_any": any(formal_flags),
            "formal_performance_gates_evaluated_for_all": all(formal_flags),
            "best_diagnostic_forward": (
                {
                    "candidate": best[0],
                    "net_return": best[1],
                }
                if best
                else None
            ),
            "worst_diagnostic_forward": (
                {
                    "candidate": worst[0],
                    "net_return": worst[1],
                }
                if worst
                else None
            ),
            "diagnostic_forward_returns_authorize_promotion": False,
        }

    @staticmethod
    def _compact_campaign(payload: dict[str, Any]) -> dict[str, Any]:
        selected = NativeFoundationBridge._first_value(
            payload,
            (
                "selected_candidate",
                "selected_strategy",
                "primary_strategy",
                "primary_strategy_id",
                "primary_policy_name",
                "primary_strategy_name",
                "champion",
            ),
        )
        candidate_count = NativeFoundationBridge._first_value(
            payload,
            (
                "generated_trial_count",
                "candidate_count",
                "registered_unique_trials",
                "trial_count",
                "total_trials",
                "strategy_count",
                "formal_risk_budget_paths",
            ),
        )
        if candidate_count is None:
            candidate_count = NativeFoundationBridge._list_count(
                payload,
                ("candidates", "trials", "strategies", "results"),
            )

        paper_count = NativeFoundationBridge._first_value(
            payload,
            ("paper_candidates", "paper_candidate_count"),
        )
        try:
            paper_count_int = int(paper_count or 0)
        except (TypeError, ValueError):
            paper_count_int = 0

        paper_permitted = bool(
            NativeFoundationBridge._first_value(
                payload,
                ("paper_candidate_permitted",),
            )
            or paper_count_int > 0
        )

        return {
            "campaign": payload.get("campaign"),
            "status": payload.get("status"),
            "promotion_state": (
                payload.get("promotion_state")
                or payload.get("status")
            ),
            "strategy_family": NativeFoundationBridge._first_value(
                payload,
                ("strategy_family",),
            ),
            "selected_candidate": selected,
            "candidate_count": int(candidate_count or 0),
            "registered_unique_trials": NativeFoundationBridge._first_value(
                payload,
                ("registered_unique_trials",),
            ),
            "total_known_trials": NativeFoundationBridge._first_value(
                payload,
                ("total_known_trials",),
            ),
            "pbo": NativeFoundationBridge._first_value(
                payload,
                ("pbo", "inherited_component_pbo"),
            ),
            "economic_pass": NativeFoundationBridge._first_value(
                payload,
                ("economic_pass",),
            ),
            "statistical_pass": NativeFoundationBridge._first_value(
                payload,
                ("statistical_pass",),
            ),
            "research_pass": NativeFoundationBridge._first_value(
                payload,
                ("research_pass",),
            ),
            "primary_positive_research_lead": (
                NativeFoundationBridge._first_value(
                    payload,
                    ("primary_positive_research_lead",),
                )
            ),
            "paper_candidate_count": paper_count_int,
            "paper_candidate_permitted": paper_permitted,
            "forward_evidence": NativeFoundationBridge._forward_evidence_digest(
                payload
            ),
            "live_ready": bool(
                NativeFoundationBridge._first_value(
                    payload,
                    ("live_ready",),
                )
                or False
            ),
            "automatic_live_promotion": False,
            "orders_generated": int(payload.get("orders_generated") or 0),
            "orders_submitted": int(payload.get("orders_submitted") or 0),
        }

    def run_alpha_campaign(self, name: str) -> dict[str, Any]:
        key = str(name).strip().lower()
        if key not in NATIVE_ALPHA_CAMPAIGNS:
            options = ", ".join(sorted(NATIVE_ALPHA_CAMPAIGNS))
            raise ValueError(f"unknown campaign; choose one of {options}")

        module_name, function_name = NATIVE_ALPHA_CAMPAIGNS[key]
        module = self.crypto.import_module(module_name)
        fn = getattr(module, function_name, None)
        if not callable(fn):
            raise RuntimeError(  # noqa: TRY004
                f"native campaign unavailable: {module_name}.{function_name}"
            )

        raw = fn(self.crypto.settings())
        if not isinstance(raw, dict):
            raise RuntimeError("native campaign returned non-dict")  # noqa: TRY004

        return {
            "schema_version": "crypto_ai_swing_native_alpha_run_v1",
            "requested_campaign": key,
            "source": (
                f"Sjagil/crypto:{module_name}.{function_name}"
            ),
            "summary": self._compact_campaign(raw),
            "raw": raw,
            "authority": "RESEARCH_ONLY",
            "live_decision_influence": False,
            "automatic_live_promotion": False,
            "orders_submitted": 0,
        }

    def write_status(self, path: Path) -> dict[str, Any]:
        status = self.status()
        payload = {
            "ready": status.ready,
            "imported_modules": status.imported_modules,
            "required_modules": status.required_modules,
            "modules": list(status.modules),
            "alpha_catalog": self.alpha_catalog(),
            "authority": "FOUNDATION_READ_RESEARCH_ONLY",
            "orders_submitted": 0,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        return payload
