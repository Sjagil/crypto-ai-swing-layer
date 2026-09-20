from __future__ import annotations

import asyncio
import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.bridge.crypto_operations import NativeOperationsBridge


INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all|any|the)\s+(previous|prior)\s+instructions", re.I),
    re.compile(r"(system|developer)\s+prompt", re.I),
    re.compile(r"execute\s+(this|the following)\s+(command|code)", re.I),
    re.compile(r"reveal\s+(your|the)\s+(secret|api key|password)", re.I),
)


def _run(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[Any] = []
    failures: list[BaseException] = []

    def target() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001
            failures.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join()
    if failures:
        raise failures[0]
    return result[0]


class ExternalResearchPlane:
    """Structured web/RSS evidence plane using canonical Sjagil/crypto.

    Browser content is always data. It is never interpreted as an instruction
    and this module contains no exchange execution adapter.
    """

    SCHEMA = "crypto_ai_swing_external_research_v1"

    def __init__(self, settings) -> None:
        self.settings = settings
        self.crypto = CryptoLibraryBridge(settings.crypto_repo_root)
        self.operations = NativeOperationsBridge(
            settings.crypto_repo_root,
            project_root=settings.project_root,
        )
        self.cfg = dict(
            (getattr(settings, "autonomy", {}) or {}).get(
                "external_research", {}
            )
            or {}
        )
        self.root = (
            Path(settings.project_root)
            / "output/crypto_ai_swing/intelligence/external_research"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.root / "latest.json"
        self.history_path = self.root / "history.jsonl"

    @staticmethod
    def _suspicious(text: str) -> bool:
        return any(pattern.search(text or "") for pattern in INJECTION_PATTERNS)

    def collect(self) -> dict[str, Any]:
        module = self.crypto.import_module("scrapers.intelligence")
        native_settings = self.crypto.settings()
        result = _run(
            module.run_intelligence_pipeline(
                native_settings,
                include_rss=bool(self.cfg.get("include_rss", True)),
            )
        )
        maximum = int(self.cfg.get("maximum_evidence_records", 250))
        records = []
        suspicious = 0
        for record in list(result.records)[:maximum]:
            title = str(record.title or "")
            summary = str(record.summary or "")
            flagged = self._suspicious(f"{title}\n{summary}")
            suspicious += int(flagged)
            records.append(
                {
                    "event_id": str(record.event_id),
                    "source": str(record.source),
                    "url": str(record.url),
                    "title": title,
                    "summary": summary[
                        : int(self.cfg.get("maximum_summary_chars", 2000))
                    ],
                    "published_at": (
                        record.published_at.isoformat()
                        if record.published_at
                        else None
                    ),
                    "observed_at": record.observed_at.isoformat(),
                    "usable_at": record.usable_at.isoformat(),
                    "markets": list(record.markets),
                    "entities": list(record.entities),
                    "categories": list(record.categories),
                    "relevance_score": float(record.relevance_score),
                    "sentiment_score": float(record.sentiment_score),
                    "impact_score": float(record.impact_score),
                    "raw_hash": str(record.raw_hash),
                    "content_trust": "UNTRUSTED_EXTERNAL",
                    "prompt_injection_suspected": flagged,
                    "instruction_authority": False,
                }
            )

        payload = {
            "schema_version": self.SCHEMA,
            "generated_at": datetime.now(UTC).isoformat(),
            "status": str(result.status),
            "record_count": len(result.records),
            "evidence_records": records,
            "source_statuses": [
                item.model_dump(mode="json") for item in result.sources
            ],
            "canonical_audit": dict(result.audit),
            "canonical_output_path": (
                str(result.output_path) if result.output_path else None
            ),
            "prompt_injection_suspected_records": suspicious,
            "security_boundary": {
                "raw_browser_content_is_untrusted": True,
                "external_text_has_instruction_authority": False,
                "direct_exchange_access": False,
                "direct_live_authority": False,
                "browser_worker_may_only_emit_structured_evidence": True,
            },
            "backend": (
                "Sjagil/crypto:scrapers.intelligence."
                "run_intelligence_pipeline"
            ),
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        self.operations.atomic_write_json(self.latest_path, payload)
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(payload, sort_keys=True, default=str) + "\n"
            )
        return payload

    def status(self) -> dict[str, Any]:
        try:
            return dict(
                json.loads(self.latest_path.read_text(encoding="utf-8"))
            )
        except (OSError, TypeError, ValueError):
            return {
                "schema_version": self.SCHEMA,
                "status": "NOT_BUILT",
                "orders_generated": 0,
                "orders_submitted": 0,
            }
