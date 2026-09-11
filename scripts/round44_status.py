

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crypto_ai_swing.settings import Settings
from crypto_ai_swing.research.feature_attribution import Round44FeatureAttribution
from crypto_ai_swing.research.live_readiness import Round44LiveReadiness


def read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    settings = Settings.load()
    root = settings.project_root / "output/crypto_ai_swing/round44"
    latest = read(root / "latest.json")
    forward_cfg = dict(settings.autonomy.get("forward_evidence", {}) or {})
    forward = settings.project_root / str(
        forward_cfg.get("path", "output/crypto_ai_swing/forward/forward.sqlite")
    )
    attribution = Round44FeatureAttribution(settings).evaluate(forward)
    readiness = Round44LiveReadiness(settings).evaluate(forward)
    payload = {
        "schema_version": "round44_status_v2",
        "comprehensive": latest.get("summary") or {},
        "attribution": attribution,
        "live_readiness": readiness,
        "automatic_live_authority": False,
        "automatic_live_promotion": False,
    }
    if args.compact:
        summary = payload["comprehensive"]
        print(
            json.dumps(
                {
                    "stage": readiness.get("stage"),
                    "next_stage": readiness.get("next_stage"),
                    "next_stage_blockers": readiness.get("next_stage_blockers"),
                    "requested_markets": summary.get("requested_markets"),
                    "technical_ready_markets": summary.get(
                        "technical_ready_markets"
                    ),
                    "l1_l2_ready_markets": summary.get("l1_l2_ready_markets"),
                    "mdpro_ready_markets": summary.get("mdpro_ready_markets"),
                    "l3": summary.get("l3"),
                    "attribution_status": attribution.get("status"),
                    "attribution_ready_horizons": attribution.get("ready_horizons"),
                    "attribution_4h_observations": (
                        attribution.get("primary") or {}
                    ).get("observations"),
                    "attribution_stable_shortlist_size": len(
                        attribution.get("stable_shortlist_features") or []
                    ),
                    "automatic_live_authority": False,
                    "automatic_live_promotion": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
