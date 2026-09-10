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
    source: str = "Sjagil/crypto:scrapers.intelligence"


class CryptoNewsCollector:
    """Consume the full canonical web + RSS intelligence pipeline.

    Network acquisition, robots handling, publication-time knowability,
    relevance filtering, deduplication and persistence remain owned by
    Sjagil/crypto. This adapter only converts canonical IntelligenceRecord
    objects into NLPDocument objects for the swing layer.
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
                ts = (
                    value
                    if isinstance(value, datetime)
                    else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
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
        get = lambda key, default=None: getattr(record, key, default)
        if isinstance(record, dict):
            get = lambda key, default=None: record.get(key, default)
        title = str(get("title", "") or "")
        summary = str(get("summary", "") or "")
        if not title and not summary:
            return None
        markets = tuple(str(x) for x in (get("markets", ()) or ()))
        categories = tuple(str(x) for x in (get("categories", ()) or ()))
        entities = tuple(str(x) for x in (get("entities", ()) or ()))
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
                "entities": entities,
                "relevance_score": float(get("relevance_score", 0.0) or 0.0),
                "sentiment_score": float(get("sentiment_score", 0.0) or 0.0),
                "impact_score": float(get("impact_score", 0.0) or 0.0),
                "timestamp_quality": str(get("timestamp_quality", "") or ""),
                "historical_coverage": str(get("historical_coverage", "") or ""),
                "classification_source": "Sjagil/crypto:scrapers.intelligence",
            },
        )

    @staticmethod
    def _status_row(row: Any) -> dict[str, Any]:
        dump = getattr(row, "model_dump", None)
        if callable(dump):
            return dict(dump(mode="json"))
        if isinstance(row, dict):
            return dict(row)
        return {
            "source_id": getattr(row, "source_id", None)
            or getattr(row, "feed_id", None),
            "publisher": getattr(row, "publisher", None),
            "status": getattr(row, "status", None),
            "fetched_records": getattr(row, "fetched_records", None)
            or getattr(row, "record_count", None),
            "relevant_records": getattr(row, "relevant_records", None),
            "error_code": getattr(row, "error_code", None),
        }

    async def _canonical_intelligence(self) -> tuple[str, Iterable[Any], Iterable[Any], Any]:
        module = self.bridge.import_module("scrapers.intelligence")
        run = getattr(module, "run_intelligence_pipeline", None)
        if not callable(run):
            raise RuntimeError("scrapers.intelligence.run_intelligence_pipeline unavailable")
        result = await run(self.bridge.settings())
        return (
            str(getattr(result, "status", "UNKNOWN")),
            getattr(result, "records", ()) or (),
            getattr(result, "sources", ()) or (),
            getattr(result, "observed_at", datetime.now(timezone.utc)),
        )

    async def _rss_fallback(self) -> tuple[str, Iterable[Any], Iterable[Any], Any]:
        module = self.bridge.import_module("scrapers.rss")
        collect = getattr(module, "collect_registered_feeds", None)
        if not callable(collect):
            raise RuntimeError("scrapers.rss.collect_registered_feeds unavailable")
        result = await collect()
        return (
            str(getattr(result, "status", "UNKNOWN")),
            getattr(result, "records", ()) or (),
            getattr(result, "feeds", ()) or (),
            getattr(result, "observed_at", datetime.now(timezone.utc)),
        )

    async def _collect_async(self) -> NewsSnapshot:
        try:
            status, records, source_rows, observed = await self._canonical_intelligence()
            source = "Sjagil/crypto:scrapers.intelligence"
        except Exception:
            status, records, source_rows, observed = await self._rss_fallback()
            source = "Sjagil/crypto:scrapers.rss"

        docs: list[NLPDocument] = []
        seen: set[tuple[str, str, str]] = set()
        for record in records:
            doc = self._to_document(record)
            if doc is None:
                continue
            key = (doc.source, doc.url or "", doc.title or doc.text[:200])
            if key in seen:
                continue
            seen.add(key)
            docs.append(doc)
        docs.sort(key=lambda item: item.usable_at, reverse=True)
        docs = docs[: self.maximum_documents]
        statuses = tuple(self._status_row(row) for row in source_rows)
        return NewsSnapshot(
            status=status,
            documents=tuple(docs),
            source_statuses=statuses,
            observed_at=(
                observed.isoformat() if hasattr(observed, "isoformat") else str(observed)
            ),
            source=source,
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
