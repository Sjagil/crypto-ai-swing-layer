from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crypto_ai_swing.contracts import TradeIntent

@dataclass(frozen=True)
class NativeAuthorityResult:
    accepted: bool
    payload: dict[str, Any]

class CryptoAuthorityAdapter:
    """Narrow adapter into Sjagil/crypto native swing canary authority."""

    def __init__(self, crypto_repo_root: Path) -> None:
        self.root = Path(crypto_repo_root).expanduser().resolve()

    def _module(self):
        import importlib, sys
        root = str(self.root)
        if root not in sys.path:
            sys.path.insert(0, root)
        return importlib.import_module("core.swing_layer_live")

    def preflight(self, intent: TradeIntent) -> dict[str, Any]:
        return self._module().swing_layer_live_preflight(intent.to_dict())

    def submit_buy(self, intent: TradeIntent) -> NativeAuthorityResult:
        payload = self._module().submit_swing_layer_buy(intent.to_dict(), execute=True)
        return NativeAuthorityResult(bool(payload.get("accepted")), dict(payload))

    def submit_exit(self, *, market: str, reason: str, quantity: str | None = None) -> NativeAuthorityResult:
        payload = self._module().submit_swing_layer_exit(
            market=market, reason=reason, requested_quantity=quantity, execute=True
        )
        return NativeAuthorityResult(bool(payload.get("accepted")), dict(payload))

    def reconcile(self, markets: list[str]) -> dict[str, Any]:
        return self._module().reconcile_swing_layer_live(tuple(markets))

    def portfolio(self) -> dict[str, Any]:
        return self._module().swing_layer_portfolio()

    def account_snapshot(self, markets: list[str]) -> dict[str, Any]:
        return self._module().swing_layer_account_snapshot(tuple(markets))

    def authority_status(self) -> dict[str, Any]:
        return self._module().swing_layer_authority_status()

    def approve(self, *, markets: list[str], approval: str) -> dict[str, Any]:
        return self._module().approve_swing_layer_canary(markets=tuple(markets), approval=approval)

    def deactivate(self) -> dict[str, Any]:
        return self._module().deactivate_swing_layer_canary()
