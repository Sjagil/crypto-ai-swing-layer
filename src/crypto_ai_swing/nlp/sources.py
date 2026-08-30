from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from typing import Iterable

import httpx

from .engine import NLPDocument


def _dt(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        out = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return out if out.tzinfo else out.replace(tzinfo=timezone.utc)
    except Exception:
        try:
            out = parsedate_to_datetime(text)
            return out if out.tzinfo else out.replace(tzinfo=timezone.utc)
        except Exception:
            return None


def _record_to_doc(record: dict, source: str) -> NLPDocument | None:
    usable = _dt(
        record.get("usable_at")
        or record.get("published_at")
        or record.get("published")
        or record.get("timestamp")
        or record.get("created_at")
    )
    if usable is None:
        return None
    title = str(record.get("title") or record.get("headline") or "")
    text = str(
        record.get("summary")
        or record.get("description")
        or record.get("text")
        or record.get("content")
        or title
    )
    if not text and not title:
        return None
    return NLPDocument(
        text=text,
        title=title,
        usable_at=usable,
        source=str(record.get("source") or source),
        url=record.get("url") or record.get("link"),
        metadata={"raw_source": source},
    )


def load_json_documents(paths: Iterable[Path], max_documents: int = 1000) -> list[NLPDocument]:
    docs: list[NLPDocument] = []
    for path in paths:
        if len(docs) >= max_documents or not path.exists() or path.stat().st_size > 20_000_000:
            continue
        try:
            if path.suffix.lower() == ".jsonl":
                rows = [json.loads(line) for line in path.read_text(errors="ignore").splitlines() if line.strip()]
            else:
                payload = json.loads(path.read_text(errors="ignore"))
                if isinstance(payload, list):
                    rows = payload
                elif isinstance(payload, dict):
                    rows = payload.get("items") or payload.get("articles") or payload.get("records") or [payload]
                else:
                    rows = []
        except Exception:
            continue
        for row in rows:
            if isinstance(row, dict):
                doc = _record_to_doc(row, str(path))
                if doc:
                    docs.append(doc)
                    if len(docs) >= max_documents:
                        break
    return docs


def discover_crypto_repo_documents(root: Path, max_files: int = 200) -> list[NLPDocument]:
    patterns = (
        "output/**/*rss*.json*",
        "output/**/*news*.json*",
        "output/**/*intelligence*.json*",
        "data/**/*rss*.json*",
        "data/**/*news*.json*",
        "data/**/*intelligence*.json*",
        "artifacts/**/*rss*.json*",
        "artifacts/**/*news*.json*",
    )
    paths: list[Path] = []
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                paths.append(path)
                if len(paths) >= max_files:
                    break
        if len(paths) >= max_files:
            break
    paths = sorted(set(paths), key=lambda p: p.stat().st_mtime, reverse=True)
    return load_json_documents(paths[:max_files])


def fetch_rss_documents(urls: Iterable[str], timeout_seconds: float = 10.0) -> list[NLPDocument]:
    docs: list[NLPDocument] = []
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        for url in urls:
            try:
                response = client.get(str(url), headers={"User-Agent": "crypto-ai-swing-layer/0.2"})
                response.raise_for_status()
                root = ET.fromstring(response.content)
            except Exception:
                continue
            entries = list(root.findall(".//item")) + list(root.findall(".//{http://www.w3.org/2005/Atom}entry"))
            for item in entries[:100]:
                def first_text(names):
                    for name in names:
                        node = item.find(name)
                        if node is not None and node.text:
                            return node.text.strip()
                    return ""
                title = first_text(("title", "{http://www.w3.org/2005/Atom}title"))
                summary = first_text(("description", "summary", "{http://www.w3.org/2005/Atom}summary"))
                published = first_text(("pubDate", "published", "updated", "{http://www.w3.org/2005/Atom}published", "{http://www.w3.org/2005/Atom}updated"))
                usable = _dt(published)
                if usable is None:
                    continue
                docs.append(NLPDocument(text=summary or title, title=title, usable_at=usable, source=str(url)))
    return docs
