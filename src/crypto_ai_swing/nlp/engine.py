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


class NLPMarketEngine:
    """Finance NLP with a FinBERT path and deterministic offline fallback.

    NLP is deliberately bounded. It can add context or veto severe negative events,
    but it cannot manufacture a live entry without a price/market signal.
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
                probs = {str(x.get("label", "")).lower(): float(x.get("score", 0.0)) for x in rows}
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

        severe_terms = ("hack", "exploit", "breach", "bankruptcy", "insolvency", "delist", "fraud")
        severe_negative = score <= -0.45 and any(x in cleaned.lower() for x in severe_terms)
        return NLPAssessment(
            score=float(max(-1.0, min(1.0, score))),
            confidence=float(max(0.0, min(1.0, confidence))),
            severe_negative=severe_negative,
            event_tags=tags,
            assets=assets,
            model=model_used,
        )

    def aggregate(
        self,
        documents: Iterable[NLPDocument],
        market: str,
        now: datetime | None = None,
    ) -> NLPAssessment:
        now = now or datetime.now(timezone.utc)
        base = market.split("-", 1)[0].upper()
        half_life_hours = float(self.config.get("half_life_hours", 12.0))
        maximum_age_hours = float(self.config.get("maximum_age_hours", 72.0))
        weighted_score = 0.0
        total_weight = 0.0
        tags: set[str] = set()
        models: set[str] = set()
        severe = False
        used = 0

        for doc in documents:
            usable_at = doc.usable_at
            if usable_at.tzinfo is None:
                continue
            age_h = max(0.0, (now - usable_at.astimezone(timezone.utc)).total_seconds() / 3600.0)
            if age_h > maximum_age_hours:
                continue
            assessment = self.assess_text(f"{doc.title} {doc.text}")
            if assessment.assets and base not in assessment.assets and "BTC" not in assessment.assets:
                continue
            asset_weight = 1.0 if base in assessment.assets else 0.45
            recency = 0.5 ** (age_h / max(half_life_hours, 1e-6))
            weight = asset_weight * recency * max(0.10, assessment.confidence)
            weighted_score += assessment.score * weight
            total_weight += weight
            tags.update(assessment.event_tags)
            models.add(assessment.model)
            severe = severe or (assessment.severe_negative and (base in assessment.assets or not assessment.assets))
            used += 1

        if total_weight <= 0 or used == 0:
            return NLPAssessment(0.0, 0.0, False, (), (base,), "none")
        score = weighted_score / total_weight
        confidence = min(1.0, 0.35 + 0.12 * used + min(0.25, total_weight / 10.0))
        return NLPAssessment(
            float(max(-1.0, min(1.0, score))),
            float(confidence),
            severe,
            tuple(sorted(tags)),
            (base,),
            "+".join(sorted(models)),
        )
