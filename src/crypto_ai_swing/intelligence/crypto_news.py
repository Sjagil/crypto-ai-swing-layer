from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import asyncio
import json
import time

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.nlp.engine import NLPDocument


@dataclass(frozen=True)
class NewsSnapshot:
    status: str
    documents: tuple[NLPDocument, ...]
    source_statuses: tuple[dict[str, Any], ...]
    observed_at: str
    source: str = "Sjagil/crypto:scrapers.rss"


class CryptoNewsCollector:
    """Consume the canonical RSS collector from Sjagil/crypto.

    Publication/observation knowability remains owned by the crypto repository.
    This adapter only converts its IntelligenceRecord contract into NLPDocument.
    """

    def __init__(
        self,
        bridge: CryptoLibraryBridge,
        *,
        output_path: Path | None = None,
        cache_seconds: float = 300.0,
        maximum_documents: int = 500,
    ) -> None:
        self.bridge = bridge
        self.output_path = output_path
        self.cache_seconds = max(0.0, float(cache_seconds))
        self.maximum_documents = max(1, int(maximum_documents))
        self._cached: tuple[float, NewsSnapshot] | None = None

    @staticmethod
    def _run(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError("CryptoNewsCollector sync API used inside active event loop")

    @staticmethod
    def _usable_at(record: Any) -> datetime | None:
        for name in ("usable_at", "published_at", "observed_at"):
            value = getattr(record, name, None)
            if value is None and isinstance(record, dict):
                value = record.get(name)
            if value is None:
                continue
            try:
                ts = value if isinstance(value, datetime) else datetime.fromisoformat(
                    str(value).replace("Z", "+00:00")
                )
                if ts.tzinfo is None:
                    return None
                return ts.astimezone(timezone.utc)
            except Exception:
                continue
        return None

    @classmethod
    def _to_document(cls, record: Any) -> NLPDocument | None:
        usable_at = cls._usable_at(record)
        if usable_at is None:
            return None
        get = (lambda key, default=None: getattr(record, key, default))
        if isinstance(record, dict):
            get = lambda key, default=None: record.get(key, default)
        title = str(get("title", "") or "")
        summary = str(get("summary", "") or "")
        if not title and not summary:
            return None
        markets = tuple(str(x) for x in (get("markets", ()) or ()))
        categories = tuple(str(x) for x in (get("categories", ()) or ()))
        return NLPDocument(
            text=summary or title,
            title=title,
            usable_at=usable_at,
            source=str(get("source", "Sjagil/crypto") or "Sjagil/crypto"),
            url=str(get("url", "") or "") or None,
            metadata={
                "event_id": str(get("event_id", "") or ""),
                "markets": markets,
                "categories": categories,
                "relevance_score": float(get("relevance_score", 0.0) or 0.0),
                "sentiment_score": float(get("sentiment_score", 0.0) or 0.0),
                "impact_score": float(get("impact_score", 0.0) or 0.0),
                "timestamp_quality": str(get("timestamp_quality", "") or ""),
                "historical_coverage": str(get("historical_coverage", "") or ""),
            },
        )

    async def _collect_async(self) -> NewsSnapshot:
        module = self.bridge.import_module("scrapers.rss")
        collect = getattr(module, "collect_registered_feeds", None)
        if not callable(collect):
            raise RuntimeError("scrapers.rss.collect_registered_feeds unavailable")
        collection = await collect()
        records: Iterable[Any] = getattr(collection, "records", ()) or ()
        classifier = None
        try:
            intelligence = self.bridge.import_module("scrapers.intelligence")
            candidate = getattr(intelligence, "classify_text", None)
            classifier = candidate if callable(candidate) else None
        except Exception:
            classifier = None
        docs: list[NLPDocument] = []
        seen: set[tuple[str, str, str]] = set()
        for record in records:
            doc = self._to_document(record)
            if doc is None:
                continue
            if classifier is not None:
                try:
                    metadata = dict(doc.metadata)
                    base_categories = tuple(metadata.get("categories", ()) or ())
                    crypto_native = float(metadata.get("relevance_score", 0.0) or 0.0) >= 0.99
                    (
                        relevance,
                        entities,
                        markets,
                        categories,
                        sentiment,
                        impact,
                    ) = classifier(
                        doc.title,
                        doc.text,
                        base_categories=base_categories,
                        crypto_native=crypto_native,
                    )
                    metadata.update(
                        {
                            "markets": tuple(sorted(set(metadata.get("markets", ())) | set(markets))),
                            "categories": tuple(sorted(set(base_categories) | set(categories))),
                            "entities": tuple(entities),
                            "relevance_score": float(max(float(metadata.get("relevance_score", 0.0) or 0.0), relevance)),
                            "sentiment_score": float(sentiment),
                            "impact_score": float(max(float(metadata.get("impact_score", 0.0) or 0.0), impact)),
                            "classification_source": "Sjagil/crypto:scrapers.intelligence.classify_text",
                        }
                    )
                    doc = NLPDocument(
                        text=doc.text,
                        title=doc.title,
                        usable_at=doc.usable_at,
                        source=doc.source,
                        url=doc.url,
                        metadata=metadata,
                    )
                except Exception:
                    pass
            key = (doc.source, doc.url or "", doc.title or doc.text[:200])
            if key in seen:
                continue
            seen.add(key)
            docs.append(doc)
        docs.sort(key=lambda item: item.usable_at, reverse=True)
        docs = docs[: self.maximum_documents]
        statuses = []
        for row in getattr(collection, "feeds", ()) or ():
            dump = getattr(row, "model_dump", None)
            if callable(dump):
                value = dump(mode="json")
            elif isinstance(row, dict):
                value = dict(row)
            else:
                value = {
                    "feed_id": getattr(row, "feed_id", None),
                    "publisher": getattr(row, "publisher", None),
                    "status": getattr(row, "status", None),
                    "record_count": getattr(row, "record_count", None),
                    "error_code": getattr(row, "error_code", None),
                }
            statuses.append(value)
        observed = getattr(collection, "observed_at", datetime.now(timezone.utc))
        status = str(getattr(collection, "status", "UNKNOWN"))
        return NewsSnapshot(
            status=status,
            documents=tuple(docs),
            source_statuses=tuple(statuses),
            observed_at=observed.isoformat() if hasattr(observed, "isoformat") else str(observed),
        )

    def collect(self, *, force: bool = False, persist: bool = True) -> NewsSnapshot:
        now = time.time()
        if (
            not force
            and self._cached is not None
            and now - self._cached[0] <= self.cache_seconds
        ):
            return self._cached[1]
        snapshot = self._run(self._collect_async())
        self._cached = (now, snapshot)
        if persist and self.output_path is not None:
            self._persist(snapshot)
        return snapshot

    def _persist(self, snapshot: NewsSnapshot) -> None:
        assert self.output_path is not None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as fh:
            for doc in snapshot.documents:
                fh.write(
                    json.dumps(
                        {
                            "title": doc.title,
                            "text": doc.text,
                            "usable_at": doc.usable_at.isoformat(),
                            "source": doc.source,
                            "url": doc.url,
                            "metadata": doc.metadata,
                        },
                        sort_keys=True,
                        default=str,
                    )
                    + "\n"
                )
