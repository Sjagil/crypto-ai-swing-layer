from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Any

import httpx
import pandas as pd


class BitvavoError(RuntimeError):
    pass


@dataclass(frozen=True)
class LiveGateStatus:
    ready: bool
    blockers: tuple[str, ...]


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "enabled"}


def live_gate_status(config: dict | None = None) -> LiveGateStatus:
    cfg = config or {}
    blockers: list[str] = []
    execution = cfg.get("execution", {})
    if not bool(execution.get("live_enabled", False)):
        blockers.append("LIVE_DISABLED_IN_CONFIG")
    if not _truthy(os.getenv("LIVE_TRADING_ALLOWED")):
        blockers.append("LIVE_TRADING_ALLOWED_FALSE")
    if not _truthy(os.getenv("WM_REAL_ORDER_EXECUTION_ENABLED")):
        blockers.append("WM_REAL_ORDER_EXECUTION_DISABLED")
    if not _truthy(os.getenv("WM_I_UNDERSTAND_LIVE_RISK")):
        blockers.append("LIVE_RISK_ACK_MISSING")
    approval = str(os.getenv("WM_LIVE_APPROVAL") or "").strip().upper()
    swing_ack = str(os.getenv("CRYPTO_SWING_LIVE_ACK") or "").strip().upper()
    if approval != "APPROVED_LIVE_ACTIVE" and swing_ack != "I_UNDERSTAND":
        blockers.append("LIVE_APPROVAL_MISSING")
    if not _truthy(os.getenv("WITHDRAWALS_DISABLED")):
        blockers.append("WITHDRAWALS_NOT_DECLARED_DISABLED")
    scope = str(os.getenv("BITVAVO_TRADE_KEY_SCOPE") or "").lower()
    if "withdraw" in scope:
        blockers.append("TRADE_KEY_SCOPE_CONTAINS_WITHDRAW")
    if scope and "trade" not in scope:
        blockers.append("TRADE_SCOPE_NOT_CONFIRMED")
    if not (os.getenv("BITVAVO_TRADE_API_KEY") or os.getenv("BITVAVO_API_KEY")):
        blockers.append("BITVAVO_API_KEY_MISSING")
    if not (os.getenv("BITVAVO_TRADE_API_SECRET") or os.getenv("BITVAVO_API_SECRET")):
        blockers.append("BITVAVO_API_SECRET_MISSING")
    if not str(os.getenv("WM_OPERATOR_ID") or "").strip().isdigit():
        blockers.append("WM_OPERATOR_ID_MISSING_OR_INVALID")
    return LiveGateStatus(not blockers, tuple(blockers))


class BitvavoREST:
    """Small spot-only Bitvavo client.

    Intentionally exposes market data, balances and order placement only. It has no
    withdrawal method. Private signatures follow Bitvavo REST v2 documentation.
    """

    base_url = "https://api.bitvavo.com/v2"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        access_window_ms: int | None = None,
        timeout_seconds: float = 15.0,
        client: httpx.Client | None = None,
    ):
        self.api_key = api_key or os.getenv("BITVAVO_TRADE_API_KEY") or os.getenv("BITVAVO_API_KEY") or ""
        self.api_secret = api_secret or os.getenv("BITVAVO_TRADE_API_SECRET") or os.getenv("BITVAVO_API_SECRET") or ""
        self.access_window_ms = int(access_window_ms or os.getenv("BITVAVO_ACCESS_WINDOW_MS") or 10000)
        self.operator_id = int(os.getenv("WM_OPERATOR_ID") or 0)
        self.client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "BitvavoREST":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def create_signature(self, timestamp_ms: int, method: str, endpoint: str, body: str = "") -> str:
        path = endpoint if endpoint.startswith("/v2") else "/v2" + endpoint
        payload = f"{timestamp_ms}{method.upper()}{path}{body}"
        return hmac.new(
            self.api_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    @staticmethod
    def _raise_for_payload(response: httpx.Response) -> Any:
        try:
            payload = response.json()
        except Exception as exc:
            raise BitvavoError(f"Bitvavo returned non-JSON HTTP {response.status_code}") from exc
        if response.status_code >= 400:
            raise BitvavoError(f"Bitvavo HTTP {response.status_code}: {payload}")
        if isinstance(payload, dict) and payload.get("errorCode") is not None:
            raise BitvavoError(f"Bitvavo error: {payload}")
        return payload

    def public_get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        response = self.client.get(self.base_url + endpoint, params=params)
        return self._raise_for_payload(response)

    def private_request(self, method: str, endpoint: str, body_obj: dict[str, Any] | None = None) -> Any:
        if not self.api_key or not self.api_secret:
            raise BitvavoError("Private Bitvavo credentials are missing")
        method = method.upper()
        timestamp = int(time.time() * 1000)
        body = "" if method == "GET" else json.dumps(body_obj or {}, separators=(",", ":"))
        signature = self.create_signature(timestamp, method, endpoint, body)
        headers = {
            "Bitvavo-Access-Key": self.api_key,
            "Bitvavo-Access-Signature": signature,
            "Bitvavo-Access-Timestamp": str(timestamp),
            "Bitvavo-Access-Window": str(self.access_window_ms),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        response = self.client.request(
            method,
            self.base_url + endpoint,
            headers=headers,
            content=None if method == "GET" else body,
        )
        return self._raise_for_payload(response)

    def balances(self) -> list[dict[str, Any]]:
        payload = self.private_request("GET", "/balance")
        return payload if isinstance(payload, list) else []

    def market_info(self, market: str) -> dict[str, Any]:
        payload = self.public_get("/markets", {"market": market})
        if isinstance(payload, list):
            return payload[0] if payload else {}
        return payload if isinstance(payload, dict) else {}

    def ticker_book(self, market: str) -> dict[str, Any]:
        payload = self.public_get("/ticker/book", {"market": market})
        if isinstance(payload, list):
            return payload[0] if payload else {}
        return payload if isinstance(payload, dict) else {}

    def ticker_24h(self, market: str) -> dict[str, Any]:
        payload = self.public_get("/ticker/24h", {"market": market})
        if isinstance(payload, list):
            return payload[0] if payload else {}
        return payload if isinstance(payload, dict) else {}

    def candles(self, market: str, interval: str = "1h", limit: int = 600) -> pd.DataFrame:
        rows = self.public_get(f"/{market}/candles", {"interval": interval, "limit": int(limit)})
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        records = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 6:
                continue
            records.append(
                {
                    "timestamp": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
            )
        if not records:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(records).drop_duplicates("timestamp").set_index("timestamp").sort_index()
        return df

    @staticmethod
    def _quantize(value: Decimal, decimals: int) -> str:
        quantum = Decimal("1").scaleb(-max(0, int(decimals)))
        return format(value.quantize(quantum, rounding=ROUND_DOWN), "f")

    def place_market_buy(self, market: str, notional_eur: Decimal, client_order_id: str | None = None) -> dict[str, Any]:
        if self.operator_id <= 0:
            raise BitvavoError("WM_OPERATOR_ID is required for live order placement")
        info = self.market_info(market)
        decimals = int(info.get("notionalDecimals", 2) or 2)
        amount_quote = self._quantize(notional_eur, decimals)
        body = {
            "market": market,
            "side": "buy",
            "orderType": "market",
            "operatorId": self.operator_id,
            "clientOrderId": client_order_id or str(uuid.uuid4()),
            "amountQuote": amount_quote,
            "responseRequired": True,
        }
        payload = self.private_request("POST", "/order", body)
        return payload if isinstance(payload, dict) else {"raw": payload}

    def place_market_sell(self, market: str, amount: Decimal, client_order_id: str | None = None) -> dict[str, Any]:
        if self.operator_id <= 0:
            raise BitvavoError("WM_OPERATOR_ID is required for live order placement")
        info = self.market_info(market)
        decimals = int(info.get("quantityDecimals", 8) or 8)
        qty = self._quantize(amount, decimals)
        body = {
            "market": market,
            "side": "sell",
            "orderType": "market",
            "operatorId": self.operator_id,
            "clientOrderId": client_order_id or str(uuid.uuid4()),
            "amount": qty,
            "responseRequired": True,
        }
        payload = self.private_request("POST", "/order", body)
        return payload if isinstance(payload, dict) else {"raw": payload}
