from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sqlite3
from typing import Any


class ShadowLedger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def _last_hash(self) -> str:
        row = self.conn.execute(
            "SELECT event_hash FROM events ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else "GENESIS"

    def append(self, event_id: str, kind: str, payload: dict[str, Any]) -> bool:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        prev = self._last_hash()
        digest = hashlib.sha256(f"{event_id}|{kind}|{canonical}|{prev}".encode()).hexdigest()
        try:
            self.conn.execute(
                "INSERT INTO events(event_id, ts, kind, payload, prev_hash, event_hash) VALUES(?,?,?,?,?,?)",
                (
                    event_id,
                    datetime.now(timezone.utc).isoformat(),
                    kind,
                    canonical,
                    prev,
                    digest,
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def verify_chain(self) -> bool:
        prev = "GENESIS"
        rows = self.conn.execute(
            "SELECT event_id, kind, payload, prev_hash, event_hash FROM events ORDER BY rowid"
        ).fetchall()
        for event_id, kind, payload, stored_prev, stored_hash in rows:
            if stored_prev != prev:
                return False
            expected = hashlib.sha256(
                f"{event_id}|{kind}|{payload}|{prev}".encode()
            ).hexdigest()
            if expected != stored_hash:
                return False
            prev = stored_hash
        return True

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def close(self) -> None:
        self.conn.close()
