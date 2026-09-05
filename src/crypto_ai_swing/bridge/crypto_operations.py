from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge

REUSED_NATIVE_INTERFACES = {
    "utils.common": ("atomic_write_json", "append_jsonl", "utc_iso", "utc_now"),
    "notifications.telegram": ("TelegramNotifier",),
    "risk.correlation_analyzer": ("CorrelationAnalyzer",),
    "risk.risk_manager": ("OperationalDegradation", "RiskManager"),
    "research.trading_math": (
        "calculate_position_size",
        "calculate_position_size_from_stop_fraction",
    ),
    "core.live_asset_preflight": ("live_account_health",),
    "core.economics": ("CanonicalCostModel",),
}


class NativeOperationsBridge:
    """Reuse Sjagil/crypto operational primitives without reimplementing them."""

    def __init__(self, crypto_repo_root: Path, *, project_root: Path) -> None:
        self.crypto = CryptoLibraryBridge(crypto_repo_root)
        self.project_root = Path(project_root).expanduser().resolve()

    def status(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for module_name, interfaces in REUSED_NATIVE_INTERFACES.items():
            row: dict[str, Any] = {
                "module": module_name,
                "required": list(interfaces),
                "available": [],
                "missing": [],
                "ready": False,
            }
            try:
                module = self.crypto.import_module(module_name)
                row["available"] = [name for name in interfaces if hasattr(module, name)]
                row["missing"] = [name for name in interfaces if not hasattr(module, name)]
                row["ready"] = not row["missing"]
                row["file"] = str(getattr(module, "__file__", "") or "")
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            rows.append(row)
        return {
            "schema_version": "crypto_ai_swing_native_reuse_v1",
            "crypto_repo_root": str(self.crypto.root),
            "ready": all(bool(row["ready"]) for row in rows),
            "modules": rows,
            "policy": {
                "duplicate_telegram_transport": False,
                "duplicate_atomic_persistence": False,
                "duplicate_correlation_engine": False,
                "duplicate_position_sizing_math": False,
                "duplicate_live_account_truth": False,
                "duplicate_transaction_cost_model": False,
            },
        }

    def atomic_write_json(self, path: Path, payload: Any) -> None:
        module = self.crypto.import_module("utils.common")
        module.atomic_write_json(Path(path), payload)

    def notify_system_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        allowed_markets: list[str] | tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Use the canonical crypto Telegram notifier. Failures never affect trading."""
        try:
            native_settings = self.crypto.settings()
            telegram = getattr(native_settings, "telegram", None)
            if telegram is None:
                return {
                    "delivery_status": "UNAVAILABLE",
                    "reason_code": "NATIVE_TELEGRAM_SETTINGS_MISSING",
                    "orders_generated": 0,
                    "orders_submitted": 0,
                }
            module = self.crypto.import_module("notifications.telegram")
            notifier = module.TelegramNotifier(
                telegram,
                output_directory=native_settings.paths.output_dir / "notifications",
                allowed_markets=allowed_markets,
            )
            result = notifier.notify_system_event(event_type, dict(payload))
            return {
                **dict(result or {}),
                "backend": "Sjagil/crypto:notifications.telegram.TelegramNotifier",
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        except Exception as exc:
            return {
                "delivery_status": "ERROR",
                "reason_code": f"NATIVE_TELEGRAM_{type(exc).__name__.upper()}",
                "error": str(exc)[:300],
                "orders_generated": 0,
                "orders_submitted": 0,
            }

    def degradation(
        self,
        *,
        warning: tuple[str, ...] = (),
        reduce_risk: tuple[str, ...] = (),
        block_new_entries: tuple[str, ...] = (),
        kill_switch: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        module = self.crypto.import_module("risk.risk_manager")
        state_dir = self.project_root / "output/crypto_ai_swing/supervisor"
        monitor = module.OperationalDegradation(
            state_path=state_dir / "native_degradation.json",
            audit_path=state_dir / "native_degradation.jsonl",
            persistence=2,
        )
        return monitor.evaluate(
            warning=warning,
            reduce_risk=reduce_risk,
            block_new_entries=block_new_entries,
            kill_switch=kill_switch,
        )

    def canonical_cost_inputs(self) -> dict[str, Any]:
        """Read the versioned cost baseline from canonical Sjagil/crypto."""
        module = self.crypto.import_module("core.economics")
        model = module.CanonicalCostModel.from_settings(self.crypto.settings())
        return {
            "cost_model_version": str(model.cost_model_version),
            "maker_fee_bps": float(model.maker_fee_fraction) * 10_000.0,
            "taker_fee_bps": float(model.taker_fee_fraction) * 10_000.0,
            "spread_bps": float(model.spread_bps),
            "slippage_bps": float(model.slippage_bps),
            "calibration_status": str(model.calibration_status),
        }

    def native_interface(self, module_name: str, name: str):
        module = self.crypto.import_module(module_name)
        value = getattr(module, name, None)
        if value is None:
            raise RuntimeError(f"NATIVE_INTERFACE_MISSING:{module_name}.{name}")
        return value
