from __future__ import annotations
import json
from pathlib import Path
import crypto_ai_swing
from crypto_ai_swing.production.certification import CanaryCertification
from crypto_ai_swing.production.recovery import ExecutionRecoveryJournal
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
    readiness = read_json(root / "production/readiness.json")
    selector = read_json(root / "research/entry_selector/latest.json")
    geometry = read_json(root / "research/swing_geometry/latest.json")
    edge = read_json(root / "research/net_edge/latest.json")
    recovery = ExecutionRecoveryJournal(
        root / "production/execution_recovery.sqlite"
    ).status()
    payload = {
        "version": crypto_ai_swing.__version__,
        "production_readiness": readiness or {"status": "NOT_RUN"},
        "recovery": recovery,
        "canary_certification": CanaryCertification(settings).status(),
        "strategy_evidence": {
            "entry_selector": {
                "status": selector.get("status", "MISSING"),
                "qualified": bool(selector.get("qualified", False)),
                "observations": selector.get("observations"),
            },
            "swing_geometry": {
                "status": geometry.get("status", "MISSING"),
                "qualified": bool(geometry.get("qualified", False)),
                "observations": geometry.get("observations"),
            },
            "net_edge": {
                "status": edge.get("status", "MISSING"),
                "qualified": bool(edge.get("qualified", False)),
                "observations": edge.get("observations"),
            },
        },
        "execution_validation_canary_can_precede_strategy_qualification": True,
        "strategy_scaling_authorized": False,
        "automatic_live_promotion": False,
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
