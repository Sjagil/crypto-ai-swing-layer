from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ResearchPromotionRegistry:
    """Research-only registry for meta, strategy, geometry and RL challengers."""

    SCHEMA = "crypto_ai_swing_research_promotion_registry_v5"

    def __init__(self, settings) -> None:
        self.settings = settings
        self.root = Path(settings.project_root) / "output/crypto_ai_swing/promotion"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "registry.json"

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    def _artifact(self, relative: str) -> dict[str, Any] | None:
        path = Path(self.settings.project_root) / relative
        value = self._read_json(path)
        if value:
            value["artifact_path"] = str(path)
        return value

    def _edge(self) -> dict[str, Any] | None:
        return self._artifact("output/crypto_ai_swing/agents/meta/latest.json")

    def _strategy(self) -> dict[str, Any] | None:
        return self._artifact("output/crypto_ai_swing/research/strategy_lab/latest.json")

    def _attribution(self) -> dict[str, Any] | None:
        return self._artifact("output/crypto_ai_swing/research/attribution/latest.json")

    def _net_edge(self) -> dict[str, Any] | None:
        return self._artifact("output/crypto_ai_swing/research/net_edge/latest.json")

    def _swing_geometry(self) -> dict[str, Any] | None:
        return self._artifact("output/crypto_ai_swing/research/swing_geometry/latest.json")

    def _entry_selector(self) -> dict[str, Any] | None:
        return self._artifact(
            "output/crypto_ai_swing/research/entry_selector/latest.json"
        )

    def _rl(self) -> dict[str, Any] | None:
        pointer = Path(self.settings.project_root) / "output/crypto_ai_swing/agents/rl/latest.pointer.json"
        p = self._read_json(pointer)
        if not p or not p.get("manifest_path"):
            return None
        path = Path(str(p["manifest_path"]))
        value = self._read_json(path)
        if value:
            value["artifact_path"] = str(path)
        return value

    @staticmethod
    def _entry(kind: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        if not payload:
            return {"kind": kind, "state": "MISSING", "qualified": False}
        status = str(payload.get("status") or "UNQUALIFIED")
        qualified = bool(payload.get("qualified"))
        if kind in {"edge", "meta_manager", "swing_geometry"}:
            qualified = qualified or status.upper() == "QUALIFIED"
        if kind == "strategy_lab":
            qualified = bool(payload.get("champion")) and status.upper() == "RESEARCH_CHAMPION"
        return {
            "kind": kind,
            "state": "RESEARCH_CANDIDATE" if qualified else status,
            "qualified": qualified,
            "artifact_path": payload.get("artifact_path"),
            "observations": payload.get("observations"),
            "selected_seed": payload.get("selected_seed"),
            "metrics": (
                payload.get("test_metrics")
                or payload.get("holdout")
                or payload.get("overall")
                or payload.get("sequential_oos")
            ),
            "reason_codes": payload.get("reason_codes"),
            "live_decision_influence": False,
            "automatic_live_promotion": False,
        }

    def refresh(self) -> dict[str, Any]:
        strategy = self._strategy()
        edge = self._edge()
        rl = self._rl()
        attribution = self._attribution()
        net_edge = self._net_edge()
        swing_geometry = self._swing_geometry()
        entry_selector = self._entry_selector()
        challengers = [
            self._entry("edge", edge),
            self._entry("strategy_lab", strategy),
            self._entry("swing_geometry", swing_geometry),
            self._entry("entry_selector", entry_selector),
            self._entry("rl", rl),
        ]
        champion = None
        if strategy and isinstance(strategy.get("champion"), dict):
            champion = {
                **dict(strategy["champion"]),
                "entry_selector": {
                "status": (entry_selector or {}).get("status", "MISSING"),
                "qualified": bool(
                    (entry_selector or {}).get("qualified", False)
                ),
                "observations": (entry_selector or {}).get("observations"),
                "artifact_path": (entry_selector or {}).get("artifact_path"),
                "live_decision_influence": False,
            },
            "promotion_scope": "RESEARCH_ONLY",
                "live_decision_influence": False,
            }
        payload = {
            "schema_version": self.SCHEMA,
            "status": "READY",
            "generated_at": _now(),
            "champion": champion,
            "challengers": challengers,
            "qualified_research_candidates": sum(
                bool(x.get("qualified")) for x in challengers
            ),
            "attribution": {
                "status": (attribution or {}).get("status", "MISSING"),
                "observations": (attribution or {}).get("observations"),
                "artifact_path": (attribution or {}).get("artifact_path"),
            },
            "net_edge_calibration": {
                "status": (net_edge or {}).get("status", "MISSING"),
                "qualified": bool((net_edge or {}).get("qualified", False)),
                "observations": (net_edge or {}).get("observations"),
                "artifact_path": (net_edge or {}).get("artifact_path"),
                "live_decision_influence": False,
            },
            "swing_geometry": {
                "status": (swing_geometry or {}).get("status", "MISSING"),
                "qualified": bool((swing_geometry or {}).get("qualified", False)),
                "observations": (swing_geometry or {}).get("observations"),
                "artifact_path": (swing_geometry or {}).get("artifact_path"),
                "live_decision_influence": False,
            },
            "promotion_scope": "RESEARCH_ONLY",
            "live_decision_influence": False,
            "automatic_live_promotion": False,
            "manual_live_authority_still_required": True,
        }
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return payload

    def status(self) -> dict[str, Any]:
        return self._read_json(self.path) or {
            "schema_version": self.SCHEMA,
            "status": "NOT_BUILT",
            "promotion_scope": "RESEARCH_ONLY",
            "live_decision_influence": False,
            "automatic_live_promotion": False,
        }
