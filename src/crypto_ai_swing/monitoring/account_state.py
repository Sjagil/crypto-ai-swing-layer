from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from crypto_ai_swing.execution.crypto_authority import CryptoAuthorityAdapter


def _find(value: Any, keys: tuple[str, ...], depth: int = 0):
    if depth > 7:
        return None
    if isinstance(value, dict):
        for key in keys:
            if key in value:
                return value[key]
        for item in value.values():
            found = _find(item, keys, depth + 1)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value[:50]:
            found = _find(item, keys, depth + 1)
            if found is not None:
                return found
    return None


def collect_account_state(
    settings,
    markets: list[str] | tuple[str, ...],
    *,
    adapter_factory: Callable[[Path], Any] = CryptoAuthorityAdapter,
) -> dict[str, Any]:
    adapter = adapter_factory(settings.crypto_repo_root)
    errors = []
    try:
        authority = dict(adapter.authority_status() or {})
    except Exception as exc:
        authority = {}
        errors.append(f"AUTHORITY:{type(exc).__name__}:{str(exc)[:180]}")
    try:
        portfolio = dict(adapter.portfolio() or {})
    except Exception as exc:
        portfolio = {}
        errors.append(f"PORTFOLIO:{type(exc).__name__}:{str(exc)[:180]}")
    try:
        account = dict(adapter.account_snapshot(list(markets)) or {})
        private_ok = bool(account) and str(
            _find(account, ("status",)) or ""
        ).upper() not in {"BLOCKED", "FAILED", "ERROR"}
    except Exception as exc:
        account = {}
        private_ok = False
        errors.append(f"ACCOUNT:{type(exc).__name__}:{str(exc)[:220]}")

    positions = dict(portfolio.get("positions") or {})
    return {
        "schema_version": "crypto_ai_swing_account_observability_v1",
        "status": "READY" if private_ok else "BLOCKED",
        "private_account_read_ok": private_ok,
        "authority_active": bool(authority.get("active", False)),
        "authority_state": (
            authority.get("state_status")
            or portfolio.get("status")
            or _find(account, ("state_status", "reconciliation_status"))
        ),
        "eur_available": _find(
            account, ("eur_available", "available_eur", "cash_eur")
        ),
        "estimated_equity_eur": _find(
            account,
            ("estimated_total_equity_eur", "equity_eur", "total_equity_eur"),
        ),
        "wallet_crypto_exposure_eur": _find(
            account,
            ("total_wallet_asset_exposure_eur", "crypto_exposure_eur", "exposure_eur"),
        ),
        "entry_allowed": _find(account, ("entry_allowed",)),
        "entry_blockers": _find(
            account, ("entry_blockers", "blockers", "failures")
        ),
        "managed_position_count": len(positions),
        "positions": positions,
        "authority": authority,
        "portfolio": portfolio,
        "account": account,
        "errors": errors,
        "read_only": True,
        "orders_submitted": 0,
        "secrets_serialized": False,
    }
