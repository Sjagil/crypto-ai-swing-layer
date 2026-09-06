from __future__ import annotations

import json
from pathlib import Path

import crypto_ai_swing
from crypto_ai_swing.settings import Settings


def read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def main() -> int:
    settings = Settings.load()
    root = Path(settings.project_root) / "output/crypto_ai_swing"
    autonomy = read(root / "autonomy/latest.json")
    attribution = read(root / "research/attribution/latest.json")
    net_edge = read(root / "research/net_edge/latest.json")
    geometry = read(root / "research/swing_geometry/latest.json")
    strategy = read(root / "research/strategy_lab/latest.json")
    edge = read(root / "agents/meta/latest.json")
    promotion = read(root / "promotion/registry.json")
    oos = dict(geometry.get("sequential_oos") or {})
    linear = dict(geometry.get("geometry") or {})
    payload = {
        "version": crypto_ai_swing.__version__,
        "autonomy": {
            "state": autonomy.get("state"),
            "errors": autonomy.get("errors"),
        },
        "attribution_4h": {
            "status": attribution.get("status"),
            "observations": attribution.get("observations"),
            "mean_normal_net_bps": (
                (attribution.get("overall") or {}).get("normal_net") or {}
            ).get("mean_bps"),
        },
        "net_edge_anchor": {
            "status": net_edge.get("status"),
            "qualified": net_edge.get("qualified"),
            "horizon_hours": net_edge.get("horizon_hours"),
            "observations": net_edge.get("observations"),
            "overall": net_edge.get("overall"),
            "oos_validation": net_edge.get("oos_validation"),
        },
        "multi_horizon_swing_geometry": {
            "status": geometry.get("status"),
            "qualified": geometry.get("qualified"),
            "observations": geometry.get("observations"),
            "horizons_hours": geometry.get("horizons_hours"),
            "horizon_economics": geometry.get("horizon_economics"),
            "reference_cost_burdens": geometry.get("reference_cost_burdens"),
            "linear_algebra": {
                "status": linear.get("status"),
                "anchor_horizon_hours": linear.get("anchor_horizon_hours"),
                "observations": linear.get("observations"),
                "feature_names": linear.get("feature_names"),
                "effective_rank": linear.get("effective_rank"),
                "rank_90pct_variance": linear.get("rank_90pct_variance"),
                "condition_number": linear.get("condition_number"),
                "shrinkage": linear.get("shrinkage"),
                "fisher_separation": linear.get("fisher_separation"),
                "fisher_bootstrap_cosine_median": linear.get(
                    "fisher_bootstrap_cosine_median"
                ),
                "ood_threshold_mahalanobis": linear.get(
                    "ood_threshold_mahalanobis"
                ),
                "principal_factors": linear.get("principal_factors"),
                "structurally_qualified": linear.get(
                    "structurally_qualified"
                ),
            },
            "sequential_oos": {
                "status": oos.get("status"),
                "qualified": oos.get("qualified"),
                "oos_selected": oos.get("oos_selected"),
                "selected_horizon_counts": oos.get(
                    "selected_horizon_counts"
                ),
                "normal_net": oos.get("normal_net"),
                "stressed_net": oos.get("stressed_net"),
                "cost_fraction_of_mean_winner": oos.get(
                    "cost_fraction_of_mean_winner"
                ),
                "bayesian": oos.get("bayesian"),
                "checks": oos.get("checks"),
            },
        },
        "strategy_lab": {
            "status": strategy.get("status"),
            "observations": strategy.get("observations"),
            "known_trial_count": strategy.get("known_trial_count"),
            "frozen_candidate": strategy.get("frozen_candidate"),
            "champion": strategy.get("champion"),
        },
        "agent_manager": {
            "status": edge.get("status"),
            "qualified": edge.get("qualified"),
            "horizon_hours": edge.get("horizon_hours"),
            "observations": edge.get("observations"),
            "threshold": edge.get("threshold"),
            "net_edge_calibration": edge.get("net_edge_calibration"),
            "swing_geometry": edge.get("swing_geometry"),
            "reason_codes": edge.get("reason_codes"),
        },
        "promotion": {
            "status": promotion.get("status"),
            "champion": promotion.get("champion"),
            "qualified_research_candidates": promotion.get(
                "qualified_research_candidates"
            ),
            "swing_geometry": promotion.get("swing_geometry"),
        },
        "live_decision_influence": False,
        "automatic_live_promotion": False,
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
