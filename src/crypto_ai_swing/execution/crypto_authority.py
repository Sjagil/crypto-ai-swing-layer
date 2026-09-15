from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crypto_ai_swing.contracts import TradeIntent


@dataclass(frozen=True)
class NativeAuthorityResult:
    accepted: bool
    payload: dict[str, Any]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_env_file(
    path: Path,
) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    for raw in path.read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw.strip()
        if (
            not line
            or line.startswith("#")
            or "=" not in line
        ):
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()

    return values


class _RemoteExecutionClient:
    def __init__(
        self,
        base_url: str,
        secret: str,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.secret = secret.encode("utf-8")
        self.timeout = timeout

        if not (
            self.base_url.startswith("http://")
            or self.base_url.startswith("https://")
        ):
            raise ValueError(
                "CRYPTO_EXECUTOR_URL must be http(s)"
            )

        if len(self.secret) < 32:
            raise ValueError(
                "CRYPTO_EXECUTOR_HMAC_SECRET "
                "missing or too short"
            )

    def _request(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(
            payload or {},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")

        timestamp = str(int(time.time()))
        nonce = uuid.uuid4().hex
        body_hash = hashlib.sha256(
            body
        ).hexdigest()

        message = (
            f"{timestamp}\n"
            f"{nonce}\n"
            f"POST\n"
            f"{path}\n"
            f"{body_hash}"
        ).encode("utf-8")

        signature = hmac.new(
            self.secret,
            message,
            hashlib.sha256,
        ).hexdigest()

        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Exec-Timestamp": timestamp,
                "X-Exec-Nonce": nonce,
                "X-Exec-Signature": signature,
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                raw = response.read()

        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                detail = json.loads(
                    raw.decode("utf-8")
                )
            except Exception:
                detail = {
                    "error": raw.decode(
                        "utf-8",
                        errors="replace",
                    )[:500]
                }

            raise RuntimeError(
                (
                    f"REMOTE_EXECUTOR_HTTP_{exc.code}: "
                    f"{json.dumps(detail, default=str)[:1000]}"
                )
            ) from exc

        except urllib.error.URLError as exc:
            raise RuntimeError(
                "REMOTE_EXECUTOR_UNREACHABLE: "
                f"{exc.reason}"
            ) from exc

        value = json.loads(
            raw.decode("utf-8")
        )

        if not isinstance(value, dict):
            raise RuntimeError(
                "REMOTE_EXECUTOR_INVALID_RESPONSE"
            )

        return value

    def health(self) -> dict[str, Any]:
        return self._request(
            "/v1/health"
        )

    def authority(self) -> dict[str, Any]:
        return self._request(
            "/v1/authority"
        )

    def preflight(
        self,
        intent: dict[str, Any],
    ) -> dict[str, Any]:
        return self._request(
            "/v1/preflight",
            {"intent": intent},
        )

    def buy(
        self,
        intent: dict[str, Any],
    ) -> dict[str, Any]:
        return self._request(
            "/v1/buy",
            {"intent": intent},
        )

    def exit(
        self,
        *,
        market: str,
        reason: str,
        quantity: str | None,
    ) -> dict[str, Any]:
        return self._request(
            "/v1/exit",
            {
                "market": market,
                "reason": reason,
                "quantity": quantity,
            },
        )

    def reconcile(
        self,
        markets: list[str],
    ) -> dict[str, Any]:
        return self._request(
            "/v1/reconcile",
            {"markets": markets},
        )

    def portfolio(
        self,
    ) -> dict[str, Any]:
        return self._request(
            "/v1/portfolio"
        )

    def account(
        self,
        markets: list[str],
    ) -> dict[str, Any]:
        return self._request(
            "/v1/account",
            {"markets": markets},
        )

    def deactivate(
        self,
    ) -> dict[str, Any]:
        return self._request(
            "/v1/deactivate"
        )


class CryptoAuthorityAdapter:
    """Adapter into the canonical execution host.

    If remote executor settings are present, all
    private/live execution calls are routed to the
    home Pi. Otherwise the original local canonical
    path remains available.
    """

    def __init__(
        self,
        crypto_repo_root: Path,
    ) -> None:
        self.root = (
            Path(crypto_repo_root)
            .expanduser()
            .resolve()
        )

        values = {
            **_load_env_file(
                _project_root() / ".env"
            ),
            **os.environ,
        }

        url = str(
            values.get(
                "CRYPTO_EXECUTOR_URL"
            )
            or ""
        ).strip()

        secret = str(
            values.get(
                "CRYPTO_EXECUTOR_HMAC_SECRET"
            )
            or ""
        ).strip()

        self.remote = (
            _RemoteExecutionClient(
                url,
                secret,
            )
            if url and secret
            else None
        )

    def _module(self):
        import importlib
        import sys

        root = str(self.root)

        if root not in sys.path:
            sys.path.insert(
                0,
                root,
            )

        return importlib.import_module(
            "core.swing_layer_live"
        )

    def backend(self) -> str:
        return (
            "REMOTE_HOME_EXECUTOR"
            if self.remote
            else "LOCAL_CANONICAL"
        )

    def remote_health(
        self,
    ) -> dict[str, Any]:
        if not self.remote:
            return {
                "status": "LOCAL_MODE",
                "backend": self.backend(),
                "orders_generated": 0,
                "orders_submitted": 0,
            }

        return self.remote.health()

    def preflight(
        self,
        intent: TradeIntent,
    ) -> dict[str, Any]:
        if self.remote:
            return self.remote.preflight(
                intent.to_dict()
            )

        return (
            self._module()
            .swing_layer_live_preflight(
                intent.to_dict()
            )
        )

    def submit_buy(
        self,
        intent: TradeIntent,
    ) -> NativeAuthorityResult:
        if self.remote:
            payload = self.remote.buy(
                intent.to_dict()
            )
        else:
            payload = (
                self._module()
                .submit_swing_layer_buy(
                    intent.to_dict(),
                    execute=True,
                )
            )

        return NativeAuthorityResult(
            bool(
                payload.get("accepted")
            ),
            dict(payload),
        )

    def submit_exit(
        self,
        *,
        market: str,
        reason: str,
        quantity: str | None = None,
    ) -> NativeAuthorityResult:
        if self.remote:
            payload = self.remote.exit(
                market=market,
                reason=reason,
                quantity=quantity,
            )
        else:
            payload = (
                self._module()
                .submit_swing_layer_exit(
                    market=market,
                    reason=reason,
                    requested_quantity=quantity,
                    execute=True,
                )
            )

        return NativeAuthorityResult(
            bool(
                payload.get("accepted")
            ),
            dict(payload),
        )

    def reconcile(
        self,
        markets: list[str],
    ) -> dict[str, Any]:
        if self.remote:
            return self.remote.reconcile(
                markets
            )

        return (
            self._module()
            .reconcile_swing_layer_live(
                tuple(markets)
            )
        )

    def portfolio(
        self,
    ) -> dict[str, Any]:
        if self.remote:
            return self.remote.portfolio()

        return (
            self._module()
            .swing_layer_portfolio()
        )

    def account_snapshot(
        self,
        markets: list[str],
    ) -> dict[str, Any]:
        if self.remote:
            return self.remote.account(
                markets
            )

        return (
            self._module()
            .swing_layer_account_snapshot(
                tuple(markets)
            )
        )

    def authority_status(
        self,
    ) -> dict[str, Any]:
        if self.remote:
            return self.remote.authority()

        return (
            self._module()
            .swing_layer_authority_status()
        )

    def gate_status(
        self,
    ) -> dict[str, Any]:
        try:
            payload = dict(
                self.authority_status()
                or {}
            )
        except Exception as exc:
            return {
                "ready": False,
                "blockers": [
                    (
                        "CANONICAL_AUTHORITY_"
                        "UNAVAILABLE:"
                        f"{type(exc).__name__}:"
                        f"{str(exc)[:300]}"
                    )
                ],
                "backend": self.backend(),
                "canonical": {},
                "orders_generated": 0,
                "orders_submitted": 0,
            }

        blockers: list[str] = []

        if payload.get("active") is not True:
            blockers.append(
                "NATIVE_CANARY_AUTHORITY_"
                "NOT_ACTIVE"
            )

        if str(
            payload.get(
                "state_status"
            )
            or "UNKNOWN"
        ).upper() != "READY":
            blockers.append(
                "NATIVE_CANARY_STATE_NOT_READY"
            )

        if (
            payload.get(
                "execution_environment_ready"
            )
            is not True
        ):
            blockers.append(
                "NATIVE_EXECUTION_"
                "ENVIRONMENT_NOT_READY"
            )

        if payload.get("spot_only") is not True:
            blockers.append(
                "CANONICAL_SPOT_ONLY_"
                "POLICY_NOT_CONFIRMED"
            )

        if any(
            payload.get(key) is True
            for key in (
                "margin",
                "leverage",
                "shorting",
                "withdrawals",
            )
        ):
            blockers.append(
                "CANONICAL_FORBIDDEN_"
                "CAPABILITY_ENABLED"
            )

        return {
            "ready": not blockers,
            "blockers": list(
                dict.fromkeys(blockers)
            ),
            "backend": self.backend(),
            "canonical": payload,
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    def approve(
        self,
        *,
        markets: list[str],
        approval: str,
    ) -> dict[str, Any]:
        if self.remote:
            raise PermissionError(
                (
                    "Remote live approval is "
                    "intentionally disabled. "
                    "Approve authority locally "
                    "on the home execution host."
                )
            )

        return (
            self._module()
            .approve_swing_layer_canary(
                markets=tuple(markets),
                approval=approval,
            )
        )

    def deactivate(
        self,
    ) -> dict[str, Any]:
        if self.remote:
            return self.remote.deactivate()

        return (
            self._module()
            .deactivate_swing_layer_canary()
        )
