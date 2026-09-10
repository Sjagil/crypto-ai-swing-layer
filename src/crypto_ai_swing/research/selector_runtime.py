from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from crypto_ai_swing.research.entry_selector import (
    SELECTOR_FEATURES,
    ProspectiveSwingEntrySelector,
    selector_feature_vector,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def selector_context_quality(context: Mapping[str, Any]) -> dict[str, Any]:
    """Measure selector input coverage without logging raw market context."""
    features = selector_feature_vector(context)
    available = [
        name
        for name in SELECTOR_FEATURES
        if _finite(features.get(name))
    ]
    missing = [name for name in SELECTOR_FEATURES if name not in available]
    total = max(1, len(SELECTOR_FEATURES))
    coverage = len(available) / total
    required_sections = {
        "timeframe_pipeline": bool(context.get("timeframe_pipeline")),
        "mtf_challenger": bool(context.get("mtf_challenger")),
        "universe_screen": bool(context.get("universe_screen")),
    }
    canonical_source = str(context.get("data_source") or "").strip() == "Sjagil/crypto"
    return {
        "feature_count": len(SELECTOR_FEATURES),
        "available_feature_count": len(available),
        "feature_coverage": float(coverage),
        "available_features": available,
        "missing_features": missing,
        "required_sections": required_sections,
        "canonical_source": canonical_source,
    }


class SelectorDecisionLedger:
    """Idempotent local ledger for advisory selector decisions."""

    SCHEMA = "crypto_ai_swing_selector_runtime_ledger_v1"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS selector_decisions (
                decision_id TEXT PRIMARY KEY,
                recorded_at TEXT NOT NULL,
                decision_at TEXT,
                market TEXT NOT NULL,
                action TEXT NOT NULL,
                horizon_hours INTEGER,
                passes INTEGER NOT NULL,
                feature_coverage REAL NOT NULL,
                context_hash TEXT NOT NULL,
                policy_hash TEXT,
                reason_codes TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_selector_decisions_time "
            "ON selector_decisions(decision_at, market)"
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def record(self, payload: Mapping[str, Any]) -> bool:
        row = dict(payload)
        decision_id = str(row["decision_id"])
        try:
            self.conn.execute(
                """
                INSERT INTO selector_decisions (
                    decision_id, recorded_at, decision_at, market, action,
                    horizon_hours, passes, feature_coverage, context_hash,
                    policy_hash, reason_codes, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    str(row.get("recorded_at") or _now().isoformat()),
                    row.get("decision_at"),
                    str(row.get("market") or "UNKNOWN"),
                    str(row.get("action") or "ABSTAIN"),
                    (
                        int(row["horizon_hours"])
                        if row.get("horizon_hours") is not None
                        else None
                    ),
                    1 if bool(row.get("passes")) else 0,
                    float(row.get("feature_coverage") or 0.0),
                    str(row.get("context_hash") or ""),
                    row.get("policy_hash"),
                    json.dumps(list(row.get("reason_codes") or []), sort_keys=True),
                    json.dumps(row, sort_keys=True, default=str),
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def status(self) -> dict[str, Any]:
        total, selected, abstained = self.conn.execute(
            """
            SELECT
                COUNT(*),
                SUM(CASE WHEN passes=1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN action='ABSTAIN' THEN 1 ELSE 0 END)
            FROM selector_decisions
            """
        ).fetchone()
        latest = self.conn.execute(
            """
            SELECT decision_at, market, action, horizon_hours,
                   passes, feature_coverage, reason_codes
            FROM selector_decisions
            ORDER BY COALESCE(decision_at, recorded_at) DESC, recorded_at DESC
            LIMIT 1
            """
        ).fetchone()
        latest_payload = None
        if latest:
            latest_payload = {
                "decision_at": latest[0],
                "market": latest[1],
                "action": latest[2],
                "horizon_hours": latest[3],
                "passes": bool(latest[4]),
                "feature_coverage": float(latest[5]),
                "reason_codes": json.loads(latest[6] or "[]"),
            }
        return {
            "schema_version": self.SCHEMA,
            "status": "READY",
            "path": str(self.path),
            "decisions": int(total or 0),
            "selected": int(selected or 0),
            "abstained": int(abstained or 0),
            "latest": latest_payload,
            "raw_context_persisted": False,
            "secrets_persisted": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }


class ProspectiveSelectorRuntime:
    """Quality-gated, auditable runtime wrapper around the v0.27 selector.

    This layer is advisory only. It can reduce paper/shadow influence by
    abstaining when the policy is stale or canonical context is incomplete.
    It never grants execution authority and it never changes live decisions.
    """

    SCHEMA = "crypto_ai_swing_selector_runtime_v1"

    def __init__(
        self,
        settings,
        *,
        selector: ProspectiveSwingEntrySelector | None = None,
        mode: str = "paper",
    ) -> None:
        self.settings = settings
        self.mode = str(mode).lower()
        self.selector = selector or ProspectiveSwingEntrySelector(
            settings,
            mode=mode,
        )
        cfg = dict(
            (getattr(settings, "autonomy", {}) or {}).get(
                "entry_selector",
                {},
            )
            or {}
        )
        self.minimum_feature_coverage = float(
            cfg.get(
                "runtime_minimum_feature_coverage",
                cfg.get("minimum_feature_coverage", 0.60),
            )
        )
        self.maximum_policy_age_seconds = float(
            cfg.get("runtime_maximum_policy_age_seconds", 3600.0)
        )
        self.require_canonical_context = bool(
            cfg.get("runtime_require_canonical_context", True)
        )
        self.require_core_sections = bool(
            cfg.get("runtime_require_core_sections", True)
        )
        ledger_path = Path(settings.project_root) / cfg.get(
            "runtime_ledger_path",
            "output/crypto_ai_swing/research/entry_selector/runtime_decisions.sqlite",
        )
        self.ledger = SelectorDecisionLedger(ledger_path)

    def close(self) -> None:
        self.ledger.close()

    def _active_policy(
        self,
        policy: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if policy:
            return dict(policy)
        cached = getattr(self.selector, "_latest", None)
        if isinstance(cached, Mapping) and cached:
            return dict(cached)
        read_latest = getattr(self.selector, "_read_latest", None)
        if callable(read_latest):
            value = read_latest()
            if isinstance(value, Mapping):
                return dict(value)
        return {}

    def evaluate_context(
        self,
        context: Mapping[str, Any],
        *,
        policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        active = self._active_policy(policy)
        quality = selector_context_quality(context)
        generated_at = _parse_time(active.get("generated_at"))
        age_seconds = (
            max(0.0, (_now() - generated_at.astimezone(UTC)).total_seconds())
            if generated_at is not None
            else None
        )
        policy_fresh = (
            age_seconds is not None
            and age_seconds <= self.maximum_policy_age_seconds
        )
        core_sections_ready = all(
            bool(value)
            for value in quality["required_sections"].values()
        )

        runtime_checks = {
            "policy_present": bool(active),
            "policy_qualified": bool(active.get("qualified", False)),
            "policy_fresh": policy_fresh,
            "feature_coverage": (
                float(quality["feature_coverage"])
                >= self.minimum_feature_coverage
            ),
            "core_sections": (
                core_sections_ready if self.require_core_sections else True
            ),
            "canonical_source": (
                bool(quality["canonical_source"])
                if self.require_canonical_context
                else True
            ),
        }
        runtime_ready = all(runtime_checks.values())

        if runtime_ready:
            decision = dict(
                self.selector.evaluate_context(context, policy=active)
            )
        else:
            decision = {
                "schema_version": "crypto_ai_swing_entry_selector_decision_v1",
                "status": (
                    str(active.get("status") or "COLLECTING")
                    if active
                    else "COLLECTING"
                ),
                "action": "ABSTAIN",
                "passes": False,
                "qualified_policy_required": True,
                "authority": "ADVISORY_ONLY",
                "live_decision_influence": False,
                "reason_codes": [
                    name.upper() + "_FAILED"
                    for name, passed in runtime_checks.items()
                    if not passed
                ],
            }

        decision["authority"] = "ADVISORY_ONLY"
        decision["live_decision_influence"] = False
        decision["automatic_live_promotion"] = False
        decision["selector_runtime"] = {
            "schema_version": self.SCHEMA,
            "mode": self.mode,
            "runtime_ready": runtime_ready,
            "checks": runtime_checks,
            "feature_coverage": quality["feature_coverage"],
            "available_feature_count": quality["available_feature_count"],
            "feature_count": quality["feature_count"],
            "missing_features": quality["missing_features"],
            "policy_age_seconds": age_seconds,
            "maximum_policy_age_seconds": self.maximum_policy_age_seconds,
            "minimum_feature_coverage": self.minimum_feature_coverage,
            "canonical_source": quality["canonical_source"],
            "raw_context_persisted": False,
        }

        market = str(context.get("market") or "UNKNOWN").upper()
        decision_at = str(context.get("decision_at") or "") or None
        feature_values = selector_feature_vector(context)
        feature_snapshot = {
            name: (
                float(feature_values.get(name))
                if _finite(feature_values.get(name))
                else None
            )
            for name in SELECTOR_FEATURES
        }
        context_hash = _stable_hash(
            {
                "market": market,
                "decision_at": decision_at,
                "features": feature_snapshot,
            }
        )
        policy_hash = str(active.get("model_contract_hash") or "") or None
        identity = {
            "market": market,
            "decision_at": decision_at,
            "context_hash": context_hash,
            "policy_hash": policy_hash,
            "action": decision.get("action"),
            "horizon_hours": decision.get("horizon_hours"),
        }
        decision_id = _stable_hash(identity)
        ledger_payload = {
            "schema_version": self.SCHEMA,
            "decision_id": decision_id,
            "recorded_at": _now().isoformat(),
            "decision_at": decision_at,
            "market": market,
            "action": str(decision.get("action") or "ABSTAIN"),
            "horizon_hours": decision.get("horizon_hours"),
            "passes": bool(decision.get("passes", False)),
            "feature_coverage": float(quality["feature_coverage"]),
            "context_hash": context_hash,
            "policy_hash": policy_hash,
            "reason_codes": list(decision.get("reason_codes") or []),
            "runtime_checks": runtime_checks,
            "authority": "ADVISORY_ONLY",
            "live_decision_influence": False,
            "raw_context_persisted": False,
            "secrets_persisted": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        inserted = self.ledger.record(ledger_payload)
        decision["selector_runtime"]["decision_id"] = decision_id
        decision["selector_runtime"]["ledger_inserted"] = inserted
        return decision

    def status(self) -> dict[str, Any]:
        return self.ledger.status()


__all__ = [
    "ProspectiveSelectorRuntime",
    "SelectorDecisionLedger",
    "selector_context_quality",
]
