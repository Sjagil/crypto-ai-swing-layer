#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from crypto_ai_swing.settings import Settings


def _read(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def main() -> int:
    s = Settings.load()
    root = Path(s.project_root) / "output/crypto_ai_swing"
    attr = _read(root / "research/attribution/latest.json")
    lab = _read(root / "research/strategy_lab/latest.json")
    edge = _read(root / "agents/meta/latest.json")
    reg = _read(root / "promotion/registry.json")
    auto = _read(root / "autonomy/latest.json")
    payload = {
        "version_scope": "v0.24",
        "autonomy": {"state": auto.get("state"), "errors": auto.get("errors")},
        "attribution": {
            "status": attr.get("status"),
            "observations": attr.get("observations"),
            "normal_net": (attr.get("overall") or {}).get("normal_net"),
            "mfe_to_abs_mae_ratio": (attr.get("overall") or {}).get("mfe_to_abs_mae_ratio"),
            "paper": attr.get("paper"),
        },
        "strategy_lab": {
            "status": lab.get("status"),
            "known_trial_count": lab.get("known_trial_count"),
            "frozen_candidate": lab.get("frozen_candidate"),
            "champion": lab.get("champion"),
        },
        "agent_manager": {
            "status": edge.get("status"),
            "qualified": edge.get("qualified"),
            "weights": edge.get("weights"),
            "threshold": edge.get("threshold"),
            "reason_codes": edge.get("reason_codes"),
        },
        "promotion": {
            "champion": reg.get("champion"),
            "qualified_research_candidates": reg.get("qualified_research_candidates"),
        },
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
