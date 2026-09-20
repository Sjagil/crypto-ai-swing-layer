from __future__ import annotations

import json
from pathlib import Path

from crypto_ai_swing.agents.runtime import AgentRuntime
from crypto_ai_swing.agents.rl_runtime import RLRuntime
from crypto_ai_swing.execution.crypto_authority import CryptoAuthorityAdapter
from crypto_ai_swing.settings import Settings


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, dict) else {}
    except Exception:
        return {}


def main() -> None:
    root = Path.cwd()
    settings = Settings.load(root)
    supervised = AgentRuntime(settings, mode="live").status()
    rl = RLRuntime(settings, mode="live").status()
    authority = CryptoAuthorityAdapter(settings.crypto_repo_root)
    health = authority.remote_health()

    learning = read_json(
        root / "output/crypto_ai_swing/learning_worker/heartbeat.json"
    )
    hpo = read_json(root / "output/crypto_ai_swing/hpo/best.json")
    promotion = read_json(
        root / "output/crypto_ai_swing/agents/live_promotion_status.json"
    )

    print(json.dumps({
        "execution": {
            "backend": authority.backend(),
            "gateway": health.get("status"),
            "egress": (health.get("egress") or {}).get("status"),
            "authority_active": (
                health.get("canonical_authority") or {}
            ).get("active"),
            "execution_ready": (
                health.get("canonical_authority") or {}
            ).get("execution_environment_ready"),
        },
        "supervised_live": {
            "status": supervised.get("status"),
            "live_decision_influence": supervised.get(
                "live_decision_influence"
            ),
            "live_pointer_selected": supervised.get(
                "live_pointer_selected"
            ),
            "head_qualifications": supervised.get(
                "head_qualifications"
            ),
            "promotion_score": supervised.get("promotion_score"),
        },
        "rl_live": {
            "status": rl.get("status"),
            "qualified": rl.get("qualified"),
            "live_decision_influence": rl.get(
                "live_decision_influence"
            ),
            "live_pointer_selected": rl.get(
                "live_pointer_selected"
            ),
            "selected_seed": rl.get("selected_seed"),
        },
        "learning_worker": {
            "status": learning.get("status"),
            "completed_at": learning.get("completed_at"),
            "markets": learning.get("markets"),
            "errors": learning.get("errors"),
        },
        "hpo": {
            "generated_at": hpo.get("generated_at"),
            "dataset_id": hpo.get("dataset_id"),
            "backprop_mlp_enabled": hpo.get(
                "backprop_mlp_enabled"
            ),
            "heads": hpo.get("heads"),
            "rl": hpo.get("rl"),
        },
        "promotion": promotion,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
