from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
from pathlib import Path
from typing import Any, Iterable

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
            except Exception as exc:
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
            raise RuntimeError("core.live_universe.candle_health unavailable")

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
            raise RuntimeError("native candle_health returned non-dict")

        return {
            **payload,
            "source": "Sjagil/crypto:core.live_universe.candle_health",
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    def pit_feature_store_certification(self) -> dict[str, Any]:
        module = self.crypto.import_module("data.feature_store")
        strict_markets = tuple(getattr(module, "STRICT_PORTFOLIO_MARKETS"))
        policy_cls = getattr(module, "FeatureStorePolicy")
        build = getattr(module, "build_feature_tensors")

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
            "feature_rows": int(getattr(bundle, "feature_mask").sum()),
            "target_rows": int(getattr(bundle, "target_mask").sum()),
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
            except Exception as exc:
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
    def _compact_campaign(payload: dict[str, Any]) -> dict[str, Any]:
        selected = NativeFoundationBridge._first_value(
            payload,
            (
                "selected_candidate",
                "selected_strategy",
                "primary_strategy",
                "champion",
            ),
        )
        candidate_count = NativeFoundationBridge._first_value(
            payload,
            (
                "candidate_count",
                "trial_count",
                "total_trials",
                "strategy_count",
            ),
        )
        if candidate_count is None:
            candidate_count = NativeFoundationBridge._list_count(
                payload,
                ("candidates", "trials", "strategies", "results"),
            )
        return {
            "campaign": payload.get("campaign"),
            "status": payload.get("status"),
            "promotion_state": (
                payload.get("promotion_state")
                or payload.get("status")
            ),
            "strategy_family": payload.get("strategy_family"),
            "selected_candidate": selected,
            "candidate_count": int(candidate_count or 0),
            "economic_pass": NativeFoundationBridge._first_value(
                payload,
                ("economic_pass",),
            ),
            "statistical_pass": NativeFoundationBridge._first_value(
                payload,
                ("statistical_pass",),
            ),
            "paper_candidate_permitted": bool(
                NativeFoundationBridge._first_value(
                    payload,
                    ("paper_candidate_permitted",),
                )
                or False
            ),
            "live_ready": bool(
                NativeFoundationBridge._first_value(
                    payload,
                    ("live_ready",),
                )
                or False
            ),
            "automatic_live_promotion": False,
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
            raise RuntimeError(
                f"native campaign unavailable: {module_name}.{function_name}"
            )

        raw = fn(self.crypto.settings())
        if not isinstance(raw, dict):
            raise RuntimeError("native campaign returned non-dict")

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
