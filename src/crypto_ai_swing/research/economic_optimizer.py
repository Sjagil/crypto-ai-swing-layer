from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> datetime:
    return datetime.now(UTC)


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class EconomicImprovementController:
    """Convert weak paper economics into bounded research work."""

    SCHEMA = "crypto_ai_swing_economic_improvement_controller_v1"

    def __init__(self, settings) -> None:
        autonomy = dict(getattr(settings, "autonomy", {}) or {})
        self.cfg = dict(autonomy.get("economic_optimizer", {}) or {})
        self.root = (
            Path(settings.project_root)
            / "output/crypto_ai_swing/research/economic_improvement"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.state = self._read()

    def _read(self) -> dict[str, Any]:
        try:
            value = json.loads(
                self.state_path.read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError):
            return {}
        return dict(value) if isinstance(value, dict) else {}

    def _write(self) -> None:
        self.state_path.write_text(
            json.dumps(
                self.state,
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )

    def _due(self, key: str, seconds: int) -> bool:
        previous = _parse_time(self.state.get(key))
        if previous is None:
            return True
        elapsed = (
            _now() - previous.astimezone(UTC)
        ).total_seconds()
        return elapsed >= max(60, int(seconds))

    def plan(self, economics: dict[str, Any]) -> dict[str, Any]:
        status = str(economics.get("status") or "COLLECTING")
        overall = dict(economics.get("overall") or {})
        closed_trades = int(overall.get("closed_trades") or 0)
        minimum = int(self.cfg.get("minimum_closed_trades", 10))
        degraded = (
            status == "DEGRADED"
            and closed_trades >= minimum
        )
        force_train = bool(
            degraded
            and self._due(
                "last_force_train_at",
                int(
                    self.cfg.get(
                        "force_train_cooldown_seconds",
                        14400,
                    )
                ),
            )
        )
        force_research = bool(
            degraded
            and self._due(
                "last_force_research_at",
                int(
                    self.cfg.get(
                        "force_research_cooldown_seconds",
                        21600,
                    )
                ),
            )
        )
        now = _now().isoformat()
        if force_train:
            self.state["last_force_train_at"] = now
        if force_research:
            self.state["last_force_research_at"] = now
        if force_train or force_research:
            self.state["last_action_closed_trades"] = closed_trades
            self._write()

        return {
            "schema_version": self.SCHEMA,
            "status": "READY",
            "economic_status": status,
            "closed_trades": closed_trades,
            "force_train": force_train,
            "force_research": force_research,
            "research_priorities": list(
                economics.get("research_priorities") or []
            ),
            "quarantine_candidates": list(
                economics.get(
                    "research_quarantine_candidates"
                )
                or []
            ),
            "economic_qualification": bool(
                economics.get("economic_qualification", False)
            ),
            "threshold_relaxation_allowed": False,
            "risk_widening_allowed": False,
            "automatic_live_authority": False,
            "automatic_live_promotion": False,
            "orders_submitted": 0,
        }
