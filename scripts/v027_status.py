from __future__ import annotations

import json
from pathlib import Path

import crypto_ai_swing
from crypto_ai_swing.settings import Settings


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def main() -> int:
    settings = Settings.load()
    root = Path(settings.project_root) / "output/crypto_ai_swing"
    autonomy = read_json(root / "autonomy/latest.json")
    selector = read_json(root / "research/entry_selector/latest.json")
    edge = read_json(root / "agents/meta/latest.json")
    registry = read_json(root / "promotion/registry.json")

    oos = dict(selector.get("oos_validation") or {})
    normal = dict(oos.get("normal_net") or {})
    stressed = dict(oos.get("stressed_net") or {})
    payload = {
        "version": crypto_ai_swing.__version__,
        "autonomy": {
            "state": autonomy.get("state"),
            "errors": autonomy.get("errors"),
        },
        "entry_selector": {
            "status": selector.get("status", "NOT_BUILT"),
            "qualified": bool(selector.get("qualified", False)),
            "observations": selector.get("observations"),
            "oos_selected": oos.get("oos_selected"),
            "selection_fraction": oos.get("selection_fraction"),
            "abstention_fraction": oos.get("abstention_fraction"),
            "selected_horizon_counts": oos.get("selected_horizon_counts"),
            "mean_normal_net_bps": normal.get("mean_bps"),
            "mean_stressed_net_bps": stressed.get("mean_bps"),
            "bayesian": selector.get("bayesian"),
            "checks": selector.get("checks"),
            "reason_codes": selector.get("reason_codes"),
            "model_contract_hash": selector.get("model_contract_hash"),
            "live_decision_influence": False,
        },
        "agent_manager": {
            "status": edge.get("status", "NOT_BUILT"),
            "qualified": bool(edge.get("qualified", False)),
            "entry_selector": edge.get("entry_selector"),
        },
        "promotion": {
            "status": registry.get("status", "NOT_BUILT"),
            "qualified_research_candidates": registry.get(
                "qualified_research_candidates"
            ),
            "entry_selector": registry.get("entry_selector"),
        },
        "automatic_live_promotion": False,
        "live_decision_influence": False,
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
