from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar


def _now() -> datetime:
    return datetime.now(UTC)


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class ResearchDriftMonitor:
    """Diagnose evidence, agent and research degradation without changing gates."""

    SCHEMA = "crypto_ai_swing_round43_research_drift_v1"
    _SEVERITY: ClassVar[dict[str, int]] = {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

    def __init__(self, settings) -> None:
        self.settings = settings
        self.project_root = Path(settings.project_root)
        self.cfg = dict(
            (getattr(settings, "autonomy", {}) or {}).get(
                "drift_monitor",
                {},
            )
            or {}
        )
        self.root = (
            self.project_root
            / "output/crypto_ai_swing/monitoring/round43_drift"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.root / "latest.json"
        self.history_path = self.root / "history.jsonl"

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return dict(value) if isinstance(value, dict) else {}
        except (OSError, TypeError, ValueError):
            return {}

    @staticmethod
    def _age_hours(payload: dict[str, Any]) -> float | None:
        raw = (
            payload.get("generated_at")
            or payload.get("checked_at")
            or payload.get("completed_at")
        )
        parsed = _parse(raw)
        if parsed is None:
            return None
        return max(
            0.0,
            (_now() - parsed.astimezone(UTC)).total_seconds() / 3600.0,
        )

    @staticmethod
    def _agent_state(payload: dict[str, Any], key: str) -> str:
        section = dict(payload.get(key) or {})
        return str(
            section.get("status")
            or section.get("state")
            or "UNKNOWN"
        ).upper()

    def evaluate(
        self,
        *,
        evidence: dict[str, Any],
        agent_status: dict[str, Any],
        strategy_status: dict[str, Any],
    ) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []

        def add(
            code: str,
            severity: str,
            detail: str,
            action: str,
        ) -> None:
            issues.append(
                {
                    "code": code,
                    "severity": severity,
                    "detail": detail,
                    "recommended_action": action,
                }
            )

        if evidence.get("status") not in {"READY", "NO_DATABASE"}:
            add(
                "EVIDENCE_DIAGNOSTICS_ERROR",
                "HIGH",
                str(evidence.get("status")),
                "REPAIR_EVIDENCE_PIPELINE",
            )

        observation_age = evidence.get("latest_observation_age_hours")
        stale_hours = float(
            self.cfg.get("maximum_observation_staleness_hours", 2.0)
        )
        if (
            observation_age is not None
            and float(observation_age) > stale_hours
        ):
            add(
                "FORWARD_OBSERVATIONS_STALE",
                "MEDIUM",
                f"latest observation age={float(observation_age):.2f}h",
                "CHECK_DATA_AND_PROACTIVE_RUNTIME",
            )

        primary = int(evidence.get("primary_horizon_hours") or 4)
        pending = int(
            (evidence.get("pending_by_horizon") or {}).get(primary, 0)
        )
        backlog_limit = int(
            self.cfg.get("maximum_primary_maturation_backlog", 20)
        )
        if pending > backlog_limit:
            add(
                "PRIMARY_MATURATION_BACKLOG",
                "MEDIUM",
                f"pending={pending} limit={backlog_limit}",
                "PRIORITIZE_EVIDENCE_MATURATION",
            )

        discovery = int(
            evidence.get("minimum_discovery_observations") or 80
        )
        primary_count = int(
            evidence.get("primary_horizon_outcomes") or 0
        )
        champion = strategy_status.get("champion")
        if primary_count >= discovery and not champion:
            add(
                "NO_CHAMPION_WITH_DISCOVERY_EVIDENCE",
                "MEDIUM",
                f"primary outcomes={primary_count}",
                "FORCE_BOUNDED_STRATEGY_RESEARCH",
            )

        for key in ("supervised", "rl"):
            state = self._agent_state(agent_status, key)
            if state in {"ERROR", "EXPIRED", "FAILED", "NOT_TRAINED"}:
                add(
                    f"{key.upper()}_AGENT_{state}",
                    "HIGH" if state in {"ERROR", "FAILED"} else "MEDIUM",
                    f"{key} status={state}",
                    "FORCE_BOUNDED_AGENT_RETRAIN",
                )

        external = self._read(
            self.project_root
            / "output/crypto_ai_swing/intelligence/external_research/latest.json"
        )
        external_age = self._age_hours(external)
        external_stale = float(
            self.cfg.get("maximum_external_research_staleness_hours", 2.0)
        )
        if external and external_age is not None and external_age > external_stale:
            add(
                "EXTERNAL_RESEARCH_STALE",
                "LOW",
                f"external research age={external_age:.2f}h",
                "REFRESH_EXTERNAL_RESEARCH",
            )

        attribution = self._read(
            self.project_root
            / "output/crypto_ai_swing/research/attribution/latest.json"
        )
        attribution_age = self._age_hours(attribution)
        attribution_stale = float(
            self.cfg.get("maximum_attribution_staleness_hours", 2.0)
        )
        if (
            attribution
            and attribution_age is not None
            and attribution_age > attribution_stale
        ):
            add(
                "ATTRIBUTION_STALE",
                "LOW",
                f"attribution age={attribution_age:.2f}h",
                "REFRESH_ATTRIBUTION",
            )

        maximum = max(
            (self._SEVERITY.get(str(row["severity"]), 0) for row in issues),
            default=0,
        )
        severity = next(
            (
                name
                for name, value in self._SEVERITY.items()
                if value == maximum
            ),
            "INFO",
        )
        payload = {
            "schema_version": self.SCHEMA,
            "generated_at": _now().isoformat(),
            "status": "DEGRADED" if maximum >= 2 else "HEALTHY",
            "severity": severity,
            "issues": issues,
            "issue_codes": [str(row["code"]) for row in issues],
            "threshold_relaxation_allowed": False,
            "risk_widening_allowed": False,
            "automatic_live_authority": False,
            "automatic_live_promotion": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        self.latest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return payload

    def status(self) -> dict[str, Any]:
        return self._read(self.latest_path) or {
            "schema_version": self.SCHEMA,
            "status": "NOT_BUILT",
            "threshold_relaxation_allowed": False,
            "automatic_live_authority": False,
            "automatic_live_promotion": False,
        }
