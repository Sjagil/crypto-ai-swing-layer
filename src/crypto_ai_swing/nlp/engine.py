from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import re
from typing import Any, Iterable


@dataclass(frozen=True)
class NLPDocument:
    text: str
    usable_at: datetime
    title: str = ""
    source: str = "unknown"
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NLPAssessment:
    score: float
    confidence: float
    severe_negative: bool
    event_tags: tuple[str, ...]
    assets: tuple[str, ...]
    model: str


@dataclass(frozen=True)
class NLPAggregation:
    assessment: NLPAssessment
    diagnostics: dict[str, Any]


POSITIVE = {
    "approval": 1.4,
    "approved": 1.5,
    "adoption": 1.2,
    "partnership": 0.8,
    "upgrade": 0.7,
    "inflow": 1.0,
    "record inflow": 1.4,
    "launch": 0.5,
    "surge": 0.4,
    "beats": 0.6,
    "bullish": 0.7,
    "recovery": 0.5,
}
NEGATIVE = {
    "hack": -2.0,
    "hacked": -2.0,
    "exploit": -2.0,
    "breach": -1.8,
    "delist": -1.6,
    "delisting": -1.6,
    "lawsuit": -1.1,
    "investigation": -1.0,
    "outage": -1.2,
    "bankruptcy": -2.0,
    "insolvency": -2.0,
    "liquidation": -1.0,
    "outflow": -0.8,
    "rejected": -1.3,
    "ban": -1.5,
    "banned": -1.5,
    "bearish": -0.7,
    "fraud": -1.8,
}

EVENT_PATTERNS = {
    "SECURITY_INCIDENT": ("hack", "exploit", "breach", "drain", "stolen"),
    "REGULATION": ("regulation", "regulator", "sec ", "cftc", "mica", "lawsuit", "ban"),
    "ETF_INSTITUTIONAL": ("etf", "institutional", "blackrock", "fidelity", "inflow", "outflow"),
    "LISTING": ("listing", "listed", "delist", "delisting"),
    "NETWORK_OUTAGE": ("outage", "halted", "downtime", "network failure"),
    "ADOPTION": ("adoption", "partnership", "integrates", "integration", "launch"),
    "MACRO": ("federal reserve", "fed ", "inflation", "cpi", "rates", "liquidity", "recession"),
}

DEFAULT_ALIASES = {
    "BTC": ("bitcoin", "btc"),
    "ETH": ("ethereum", "ether", "eth"),
    "SOL": ("solana", "sol"),
    "TAO": ("bittensor", "tao"),
    "LINK": ("chainlink", "link"),
    "XRP": ("ripple", "xrp"),
    "ADA": ("cardano", "ada"),
    "AVAX": ("avalanche", "avax"),
    "DOT": ("polkadot", "dot"),
    "DOGE": ("dogecoin", "doge"),
}

MACRO_CATEGORIES = {
    "macro",
    "macro_calendar",
    "macro_liquidity",
    "interest_rates",
    "inflation",
}
GENERIC_CRYPTO_CATEGORIES = {
    "crypto_market",
    "crypto_news",
    "crypto_market_structure",
    "exchange",
    "regulation",
    "hack_exploit",
    "stablecoin_risk",
    "exchange_risk",
    "token_unlock",
}


class NLPMarketEngine:
    """Finance NLP with market-scoped authority and deterministic fallback.

    NLP can add context or veto a severe direct-asset event. BTC news can inform
    alt markets at a bounded cross-asset weight, but cannot automatically carry a
    severe veto into another asset. Generic security/regulatory headlines are
    context unless an explicit market attribution exists.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.model_name = str(self.config.get("model", "ProsusAI/finbert"))
        self.use_transformer = bool(self.config.get("transformer_enabled", True))
        self._pipe = None
        aliases = dict(DEFAULT_ALIASES)
        for symbol, vals in (self.config.get("asset_aliases", {}) or {}).items():
            aliases[str(symbol).upper()] = tuple(str(x).lower() for x in vals)
        self.aliases = aliases

    def _pipeline(self):
        if self._pipe is not None:
            return self._pipe
        if not self.use_transformer:
            return None
        try:
            from transformers import pipeline

            self._pipe = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                truncation=True,
                top_k=None,
            )
        except Exception:
            self._pipe = False
        return None if self._pipe is False else self._pipe

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _assets(self, text: str) -> tuple[str, ...]:
        lower = f" {text.lower()} "
        found = []
        for symbol, aliases in self.aliases.items():
            for alias in aliases:
                if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lower):
                    found.append(symbol)
                    break
        return tuple(sorted(set(found)))

    @staticmethod
    def _event_tags(text: str) -> tuple[str, ...]:
        lower = text.lower()
        return tuple(
            tag for tag, terms in EVENT_PATTERNS.items() if any(term in lower for term in terms)
        )

    @staticmethod
    def _lexical(text: str) -> tuple[float, float]:
        lower = text.lower()
        raw = 0.0
        hits = 0
        for term, weight in POSITIVE.items():
            if term in lower:
                raw += weight
                hits += 1
        for term, weight in NEGATIVE.items():
            if term in lower:
                raw += weight
                hits += 1
        score = math.tanh(raw / 2.5)
        confidence = min(0.85, 0.35 + 0.10 * hits)
        return score, confidence

    def assess_text(self, text: str) -> NLPAssessment:
        cleaned = self._clean(text)
        if not cleaned:
            return NLPAssessment(0.0, 0.0, False, (), (), "empty")
        tags = self._event_tags(cleaned)
        assets = self._assets(cleaned)
        score, confidence = self._lexical(cleaned)
        model_used = "lexical"

        pipe = self._pipeline()
        if pipe is not None:
            try:
                result = pipe(cleaned[:4000])
                rows = result[0] if result and isinstance(result[0], list) else result
                probs = {
                    str(x.get("label", "")).lower(): float(x.get("score", 0.0))
                    for x in rows
                }
                pos = probs.get("positive", 0.0)
                neg = probs.get("negative", 0.0)
                neutral = probs.get("neutral", 0.0)
                fin_score = max(-1.0, min(1.0, pos - neg))
                fin_conf = max(pos, neg, neutral)
                score = 0.70 * fin_score + 0.30 * score
                confidence = max(confidence, fin_conf)
                model_used = self.model_name
            except Exception:
                pass

        severe_terms = (
            "hack",
            "exploit",
            "breach",
            "bankruptcy",
            "insolvency",
            "delist",
            "fraud",
        )
        severe_negative = score <= -0.45 and any(
            x in cleaned.lower() for x in severe_terms
        )
        return NLPAssessment(
            score=float(max(-1.0, min(1.0, score))),
            confidence=float(max(0.0, min(1.0, confidence))),
            severe_negative=severe_negative,
            event_tags=tags,
            assets=assets,
            model=model_used,
        )

    @staticmethod
    def _metadata_values(doc: NLPDocument, key: str) -> tuple[str, ...]:
        raw = doc.metadata.get(key, ()) if isinstance(doc.metadata, dict) else ()
        if raw is None:
            return ()
        if isinstance(raw, str):
            return (raw,)
        try:
            return tuple(str(value) for value in raw)
        except TypeError:
            return ()

    def aggregate_with_diagnostics(
        self,
        documents: Iterable[NLPDocument],
        market: str,
        now: datetime | None = None,
    ) -> NLPAggregation:
        now = now or datetime.now(timezone.utc)
        base = market.split("-", 1)[0].upper()
        market = market.upper()
        half_life_hours = float(self.config.get("half_life_hours", 12.0))
        maximum_age_hours = float(self.config.get("maximum_age_hours", 72.0))
        direct_weight = float(self.config.get("direct_asset_weight", 1.0))
        btc_cross_weight = float(self.config.get("btc_cross_asset_weight", 0.20))
        macro_weight = float(self.config.get("macro_weight", 0.25))
        generic_weight = float(self.config.get("generic_weight", 0.10))
        severe_direct_only = bool(
            self.config.get("severe_veto_direct_asset_only", True)
        )

        weighted_score = 0.0
        total_weight = 0.0
        tags: set[str] = set()
        models: set[str] = set()
        severe = False
        considered = 0
        used = 0
        scope_counts = {
            "direct_asset": 0,
            "btc_cross_asset": 0,
            "macro": 0,
            "generic_crypto": 0,
            "rejected_other_asset": 0,
            "expired": 0,
            "invalid_timestamp": 0,
        }
        contributors: list[dict[str, Any]] = []

        for doc in documents:
            usable_at = doc.usable_at
            if usable_at.tzinfo is None:
                scope_counts["invalid_timestamp"] += 1
                continue
            age_h = max(
                0.0,
                (now - usable_at.astimezone(timezone.utc)).total_seconds() / 3600.0,
            )
            if age_h > maximum_age_hours:
                scope_counts["expired"] += 1
                continue
            considered += 1
            assessment = self.assess_text(f"{doc.title} {doc.text}")
            metadata_markets = {
                value.upper() for value in self._metadata_values(doc, "markets")
            }
            categories = {
                value.lower() for value in self._metadata_values(doc, "categories")
            }
            detected_assets = set(assessment.assets)

            direct = market in metadata_markets or base in detected_assets
            btc_cross = (
                base != "BTC"
                and not direct
                and ("BTC-EUR" in metadata_markets or "BTC" in detected_assets)
            )
            macro = (
                not direct
                and not btc_cross
                and not metadata_markets
                and not detected_assets
                and (bool(categories & MACRO_CATEGORIES) or "MACRO" in assessment.event_tags)
            )
            generic = (
                not direct
                and not btc_cross
                and not macro
                and not metadata_markets
                and not detected_assets
                and bool(categories & GENERIC_CRYPTO_CATEGORIES)
            )

            if direct:
                scope = "direct_asset"
                scope_weight = direct_weight
            elif btc_cross:
                scope = "btc_cross_asset"
                scope_weight = btc_cross_weight
            elif macro:
                scope = "macro"
                scope_weight = macro_weight
            elif generic:
                scope = "generic_crypto"
                scope_weight = generic_weight
            else:
                scope_counts["rejected_other_asset"] += 1
                continue

            recency = 0.5 ** (age_h / max(half_life_hours, 1e-6))
            relevance = float(doc.metadata.get("relevance_score", 1.0) or 0.0)
            relevance_factor = max(0.20, min(1.0, relevance if relevance > 0 else 1.0))
            impact = float(doc.metadata.get("impact_score", 0.0) or 0.0)
            impact_factor = 1.0 + 0.25 * max(0.0, min(1.0, impact))
            weight = (
                scope_weight
                * recency
                * max(0.10, assessment.confidence)
                * relevance_factor
                * impact_factor
            )
            if weight <= 0:
                continue

            weighted_score += assessment.score * weight
            total_weight += weight
            tags.update(assessment.event_tags)
            models.add(assessment.model)
            scope_counts[scope] += 1
            used += 1
            if assessment.severe_negative and (
                direct or (not severe_direct_only and scope in {"macro", "generic_crypto"})
            ):
                severe = True
            contributors.append(
                {
                    "source": doc.source,
                    "title": doc.title[:180],
                    "scope": scope,
                    "weight": round(float(weight), 8),
                    "score": round(float(assessment.score), 6),
                    "severe_negative": bool(assessment.severe_negative and direct),
                    "assets": list(assessment.assets),
                    "markets": sorted(metadata_markets),
                    "categories": sorted(categories),
                    "age_hours": round(float(age_h), 3),
                }
            )

        diagnostics = {
            "market": market,
            "documents_considered": considered,
            "documents_used": used,
            "scope_counts": scope_counts,
            "total_weight": float(total_weight),
            "severe_veto_policy": (
                "DIRECT_ASSET_ONLY" if severe_direct_only else "CONFIGURED_BROAD"
            ),
            "top_contributors": sorted(
                contributors,
                key=lambda row: abs(float(row["weight"]) * float(row["score"])),
                reverse=True,
            )[:8],
        }
        if total_weight <= 0 or used == 0:
            return NLPAggregation(
                NLPAssessment(0.0, 0.0, False, (), (base,), "none"),
                diagnostics,
            )
        score = weighted_score / total_weight
        confidence = min(
            0.95,
            0.20 + 0.07 * min(used, 8) + min(0.19, total_weight / 8.0),
        )
        assessment = NLPAssessment(
            float(max(-1.0, min(1.0, score))),
            float(confidence),
            severe,
            tuple(sorted(tags)),
            (base,),
            "+".join(sorted(models)),
        )
        return NLPAggregation(assessment, diagnostics)

    def aggregate(
        self,
        documents: Iterable[NLPDocument],
        market: str,
        now: datetime | None = None,
    ) -> NLPAssessment:
        return self.aggregate_with_diagnostics(documents, market, now).assessment
