from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from crypto_ai_swing.production.audit import redact

UNRESOLVED_STATES = {"SUBMITTING", "ACCEPTED", "MANUAL_REVIEW_REQUIRED"}

def _now() -> str:
    return datetime.now(UTC).isoformat()

class ExecutionRecoveryJournal:
    SCHEMA = "crypto_ai_swing_execution_recovery_v1"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS operations (
                    operation_key TEXT PRIMARY KEY,
                    operation_id TEXT NOT NULL,
                    intent_id TEXT,
                    market TEXT NOT NULL,
                    side TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_execution_recovery_state
                    ON operations(state, updated_at);
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def get(self, operation_key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT operation_key,operation_id,intent_id,market,side,state,
                       created_at,updated_at,payload
                FROM operations WHERE operation_key=?
                """,
                (str(operation_key),),
            ).fetchone()
        if not row:
            return None
        try:
            payload = json.loads(row[8] or "{}")
        except Exception:
            payload = {}
        return {
            "operation_key": row[0], "operation_id": row[1],
            "intent_id": row[2], "market": row[3], "side": row[4],
            "state": row[5], "created_at": row[6], "updated_at": row[7],
            "payload": payload,
        }

    def begin(
        self, *, operation_key: str, market: str, side: str,
        intent_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        existing = self.get(operation_key)
        if existing is not None:
            return {**existing, "created": False, "duplicate": True}
        operation_id = uuid.uuid4().hex
        timestamp = _now()
        safe_payload = redact(dict(payload or {}))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO operations(
                    operation_key,operation_id,intent_id,market,side,state,
                    created_at,updated_at,payload
                ) VALUES (?,?,?,?,?,'SUBMITTING',?,?,?)
                """,
                (
                    str(operation_key), operation_id, intent_id,
                    str(market).upper(), str(side).upper(),
                    timestamp, timestamp,
                    json.dumps(safe_payload, sort_keys=True, default=str),
                ),
            )
            conn.commit()
        return {
            "operation_key": str(operation_key), "operation_id": operation_id,
            "intent_id": intent_id, "market": str(market).upper(),
            "side": str(side).upper(), "state": "SUBMITTING",
            "created_at": timestamp, "updated_at": timestamp,
            "payload": safe_payload, "created": True, "duplicate": False,
        }

    def transition(
        self, operation_key: str, state: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = self.get(operation_key)
        if current is None:
            raise KeyError(operation_key)
        safe_payload = {
            **dict(current.get("payload") or {}),
            **redact(dict(payload or {})),
        }
        timestamp = _now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE operations SET state=?,updated_at=?,payload=? WHERE operation_key=?",
                (
                    str(state).upper(), timestamp,
                    json.dumps(safe_payload, sort_keys=True, default=str),
                    str(operation_key),
                ),
            )
            conn.commit()
        result = self.get(operation_key)
        assert result is not None
        return result

    def unresolved(self) -> list[dict[str, Any]]:
        placeholders = ",".join("?" for _ in UNRESOLVED_STATES)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT operation_key FROM operations WHERE state IN ({placeholders}) ORDER BY updated_at",
                tuple(sorted(UNRESOLVED_STATES)),
            ).fetchall()
        result = []
        for key, in rows:
            row = self.get(str(key))
            if row is not None:
                result.append(row)
        return result

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT operation_key FROM operations ORDER BY updated_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        result = []
        for key, in rows:
            row = self.get(str(key))
            if row is not None:
                result.append(row)
        return result

    def status(self) -> dict[str, Any]:
        unresolved = self.unresolved()
        return {
            "schema_version": self.SCHEMA,
            "status": "READY" if not unresolved else "RECOVERY_REQUIRED",
            "unresolved_count": len(unresolved),
            "unresolved": unresolved,
            "automatic_resubmission": False,
        }
