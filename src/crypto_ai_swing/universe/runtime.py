from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Any, Callable

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.data.features import build_features


STABLECOIN_BASES = {
    "USDT", "USDC", "DAI", "EURC", "TUSD", "FDUSD", "PYUSD", "USDE",
    "RLUSD", "USDS", "GUSD", "LUSD", "FRAX", "EURS",
}
FIAT_BASES = {"USD", "EUR", "GBP", "CHF", "JPY", "AUD", "CAD"}
LEVERAGED_SUFFIXES = ("3L", "3S", "5L", "5S", "BULL", "BEAR", "UP", "DOWN")


@dataclass(frozen=True)
class UniverseSnapshot:
    schema_version: str
    generated_at: str
    expires_at: str
    exchange: str
    quote: str
    requested_size: int
    selected_size: int
    markets: tuple[str, ...]
    strict_liquidity_count: int
    preferred_liquidity_count: int
    fallback_liquidity_count: int
    liquidity_degraded: bool
    current_liquidity_snapshot_not_point_in_time: bool
    candidates: tuple[dict[str, Any], ...]
    rejection_counts: dict[str, int]
    policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["markets"] = list(self.markets)
        value["candidates"] = list(self.candidates)
        return value


class UniverseManager:
    """Select a sticky, Shariah-filtered EUR spot universe from public Bitvavo data.

    Runtime selection is deliberately stricter than a broad research universe. A
    current liquidity snapshot is not a historical point-in-time universe and is
    never represented as one.
    """

    def __init__(
        self,
        settings,
        *,
        exchange_factory: Callable[[], Any] | None = None,
        crypto_settings: Any | None = None,
    ) -> None:
        self.settings = settings
        cfg = dict(getattr(settings, "universe", {}) or {})
        runtime = dict(cfg.get("runtime_selection", {}) or {})
        anti_hype = dict(cfg.get("anti_hype", {}) or {})
        self.cfg = runtime
        self.size = int(runtime.get("size", 25))
        self.quote = str(runtime.get("quote", "EUR")).upper()
        self.refresh_seconds = int(runtime.get("refresh_seconds", 21600))
        self.minimum_quote_volume = float(
            runtime.get("minimum_24h_quote_volume_eur", 250_000)
        )
        self.preferred_spread_bps = float(
            runtime.get("preferred_maximum_spread_bps", 15.0)
        )
        self.maximum_spread_bps = float(runtime.get("maximum_spread_bps", 35.0))
        if self.preferred_spread_bps <= 0 or self.maximum_spread_bps <= 0:
            raise ValueError("universe spread limits must be positive")
        if self.preferred_spread_bps > self.maximum_spread_bps:
            raise ValueError("preferred spread cannot exceed hard maximum spread")
        self.maximum_positive_return_24h = float(
            anti_hype.get("maximum_24h_return", 0.35)
        )
        self.core_markets = tuple(
            str(x).upper()
            for x in runtime.get("core_markets", ["BTC-EUR", "ETH-EUR", "SOL-EUR"])
        )
        self.state_path = settings.project_root / runtime.get(
            "state_path", "output/crypto_ai_swing/universe/latest.json"
        )
        self._exchange_factory = exchange_factory
        self._crypto_settings = crypto_settings

    def _native_settings(self):
        if self._crypto_settings is not None:
            return self._crypto_settings
        return CryptoLibraryBridge(self.settings.crypto_repo_root).settings()

    def _exchange(self):
        if self._exchange_factory is not None:
            return self._exchange_factory()
        import ccxt

        return ccxt.bitvavo({"enableRateLimit": True, "timeout": 15000})

    @staticmethod
    def _normalize_market(symbol: str) -> str:
        return str(symbol).upper().replace("/", "-").replace("_", "-")

    @staticmethod
    def _ticker_quote_volume(ticker: dict[str, Any]) -> float:
        for key in ("quoteVolume", "quote_volume", "quoteVolume24h"):
            raw = ticker.get(key)
            if raw is not None:
                try:
                    value = float(raw)
                    if math.isfinite(value) and value >= 0:
                        return value
                except Exception:
                    pass
        try:
            base = float(ticker.get("baseVolume") or 0.0)
            last = float(ticker.get("last") or ticker.get("close") or 0.0)
            value = base * last
            return value if math.isfinite(value) and value >= 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _spread_bps(ticker: dict[str, Any]) -> float | None:
        try:
            bid = float(ticker.get("bid") or 0.0)
            ask = float(ticker.get("ask") or 0.0)
        except Exception:
            return None
        if bid <= 0 or ask <= 0 or ask < bid:
            return None
        mid = (bid + ask) / 2.0
        return (ask - bid) / mid * 10_000.0 if mid > 0 else None

    @staticmethod
    def _return_24h_fraction(ticker: dict[str, Any]) -> float | None:
        try:
            open_price = float(ticker.get("open") or 0.0)
            last = float(ticker.get("last") or ticker.get("close") or 0.0)
            if open_price > 0 and last > 0:
                value = last / open_price - 1.0
                if math.isfinite(value):
                    return value
        except Exception:
            pass
        for key in ("percentage", "changePercentage", "percentage24h"):
            raw = ticker.get(key)
            if raw is None:
                continue
            try:
                value = float(raw) / 100.0
                if math.isfinite(value):
                    return value
            except Exception:
                continue
        return None

    def _allowed_by_shariah(self, market: str) -> bool:
        native = self._native_settings()
        eligibility = native.shariah.eligibility(market)
        status = getattr(getattr(eligibility, "status", None), "value", None)
        return str(status or "").upper() == "ALLOWED"

    def _load_snapshot(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {}
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            return dict(value) if isinstance(value, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _parse_time(raw: Any) -> datetime | None:
        if not raw:
            return None
        try:
            value = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        except Exception:
            return None

    def current(self, *, force_refresh: bool = False) -> dict[str, Any]:
        cached = self._load_snapshot()
        expiry = self._parse_time(cached.get("expires_at"))
        if (
            not force_refresh
            and cached.get("schema_version") == "crypto_ai_swing_runtime_universe_v3"
            and cached.get("selected_size") == self.size
            and isinstance(cached.get("markets"), list)
            and len(cached.get("markets") or []) == self.size
            and expiry is not None
            and datetime.now(timezone.utc) < expiry
        ):
            return cached
        return self.refresh().to_dict()

    def refresh(self) -> UniverseSnapshot:
        exchange = self._exchange()
        rejection_counts: dict[str, int] = {}

        def reject(code: str) -> None:
            rejection_counts[code] = rejection_counts.get(code, 0) + 1

        try:
            markets_raw = exchange.load_markets()
            tickers = exchange.fetch_tickers()
        finally:
            close = getattr(exchange, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        candidates: list[dict[str, Any]] = []
        for symbol, info in markets_raw.items():
            if not isinstance(info, dict):
                reject("INVALID_MARKET_METADATA")
                continue
            normalized = self._normalize_market(info.get("symbol") or symbol)
            parts = normalized.split("-")
            if len(parts) != 2:
                reject("NON_STANDARD_SYMBOL")
                continue
            base, quote = parts
            if quote != self.quote:
                reject("NON_EUR_QUOTE")
                continue
            if info.get("active") is False:
                reject("INACTIVE")
                continue
            if info.get("spot") is False:
                reject("NON_SPOT")
                continue
            if base in STABLECOIN_BASES:
                reject("STABLECOIN_BASE")
                continue
            if base in FIAT_BASES:
                reject("FIAT_BASE")
                continue
            if any(base.endswith(suffix) for suffix in LEVERAGED_SUFFIXES):
                reject("LEVERAGED_TOKEN_PATTERN")
                continue
            try:
                if not self._allowed_by_shariah(normalized):
                    reject("SHARIAH_NOT_ALLOWED")
                    continue
            except Exception:
                reject("SHARIAH_UNKNOWN")
                continue

            ticker = tickers.get(info.get("symbol") or symbol) or tickers.get(symbol) or {}
            if not isinstance(ticker, dict):
                ticker = {}
            quote_volume = self._ticker_quote_volume(ticker)
            spread = self._spread_bps(ticker)
            change_24h = self._return_24h_fraction(ticker)
            if spread is None:
                reject("SPREAD_UNAVAILABLE")
                continue
            if spread > self.maximum_spread_bps:
                reject("SPREAD_TOO_WIDE")
                continue
            if (
                change_24h is not None
                and self.maximum_positive_return_24h > 0
                and change_24h > self.maximum_positive_return_24h
            ):
                reject("ANTI_HYPE_24H_RETURN")
                continue

            strict = quote_volume >= self.minimum_quote_volume
            preferred = strict and spread <= self.preferred_spread_bps
            quality_tier = (
                "PREFERRED"
                if preferred
                else "SPREAD_FALLBACK"
                if strict
                else "LOW_VOLUME_FALLBACK"
            )
            ranking = math.log1p(max(0.0, quote_volume)) - 0.05 * spread
            if normalized in self.core_markets:
                ranking += 1000.0
            candidates.append(
                {
                    "market": normalized,
                    "base": base,
                    "quote": quote,
                    "quote_volume_24h_eur": quote_volume,
                    "spread_bps": spread,
                    "return_24h": change_24h,
                    "strict_liquidity": strict,
                    "preferred_liquidity": preferred,
                    "quality_tier": quality_tier,
                    "ranking_score": ranking,
                }
            )

        candidates.sort(
            key=lambda row: (
                bool(row["preferred_liquidity"]),
                bool(row["strict_liquidity"]),
                float(row["ranking_score"]),
            ),
            reverse=True,
        )
        preferred = [row for row in candidates if row["preferred_liquidity"]]
        strict = [row for row in candidates if row["strict_liquidity"]]
        selected = list(preferred[: self.size])
        used = {row["market"] for row in selected}
        if len(selected) < self.size:
            for row in strict:
                if row["market"] in used:
                    continue
                selected.append(row)
                used.add(row["market"])
                if len(selected) >= self.size:
                    break
        if len(selected) < self.size:
            for row in candidates:
                if row["market"] in used:
                    continue
                selected.append(row)
                used.add(row["market"])
                if len(selected) >= self.size:
                    break
        if len(selected) < self.size:
            raise RuntimeError(
                f"Only {len(selected)} eligible {self.quote} spot markets available; "
                f"cannot construct requested universe of {self.size}"
            )

        fallback_count = sum(row["quality_tier"] != "PREFERRED" for row in selected)
        liquidity_degraded = fallback_count > 0
        now = datetime.now(timezone.utc)
        snapshot = UniverseSnapshot(
            schema_version="crypto_ai_swing_runtime_universe_v3",
            generated_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=self.refresh_seconds)).isoformat(),
            exchange="bitvavo",
            quote=self.quote,
            requested_size=self.size,
            selected_size=len(selected),
            markets=tuple(row["market"] for row in selected),
            strict_liquidity_count=len(strict),
            preferred_liquidity_count=len(preferred),
            fallback_liquidity_count=fallback_count,
            liquidity_degraded=liquidity_degraded,
            current_liquidity_snapshot_not_point_in_time=True,
            candidates=tuple(selected),
            rejection_counts=rejection_counts,
            policy={
                "minimum_24h_quote_volume_eur": self.minimum_quote_volume,
                "preferred_maximum_spread_bps": self.preferred_spread_bps,
                "hard_maximum_spread_bps": self.maximum_spread_bps,
                "maximum_positive_return_24h": self.maximum_positive_return_24h,
                "shariah_filter": "NATIVE_CRYPTO_REPO_FAIL_CLOSED",
            },
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(snapshot.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        return snapshot


def screen_frame(frame) -> dict[str, float]:
    """Cheap closed-bar screen used before expensive MTF/orderflow enrichment."""
    if frame is None or frame.empty:
        return {"opportunity_score": -999.0}
    feat = build_features(frame).dropna()
    if feat.empty:
        return {"opportunity_score": -999.0}
    row = feat.iloc[-1]
    trend = math.tanh(float(row.get("trend_20_50", 0.0)) * 80.0)
    momentum = math.tanh(float(row.get("ret_24", 0.0)) * 15.0)
    breakout = math.tanh(float(row.get("breakout_20", 0.0)) * 40.0)
    volume = math.tanh(float(row.get("volume_z_48", 0.0)) / 3.0)
    close_location = float(row.get("close_location", 0.5) or 0.5)
    close_component = max(-1.0, min(1.0, 2.0 * close_location - 1.0))
    score = (
        0.35 * trend
        + 0.30 * momentum
        + 0.20 * breakout
        + 0.10 * volume
        + 0.05 * close_component
    )
    return {
        "opportunity_score": float(score),
        "trend_component": float(trend),
        "momentum_component": float(momentum),
        "breakout_component": float(breakout),
        "volume_component": float(volume),
        "ret_24": float(row.get("ret_24", 0.0) or 0.0),
        "rsi_14": float(row.get("rsi_14", 50.0) or 50.0),
    }
