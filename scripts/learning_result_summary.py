#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/crypto_ai_swing"


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def section(name: str):
    print()
    print("=" * 72)
    print(name)
    print("=" * 72)


# ------------------------------------------------------------
# SUPERVISED
# ------------------------------------------------------------

section("SUPERVISED MODEL")

pointer = read_json(
    OUT / "agents/latest.pointer.json"
)

if not pointer:
    print("STATUS=NO_POINTER")
else:
    print("pointer_status =", pointer.get("status"))
    print("updated_at     =", pointer.get("updated_at"))
    print(
        "heads          =",
        pointer.get("head_qualifications"),
    )

    manifest_path = Path(
        str(pointer.get("manifest_path") or "")
    )

    manifest = (
        read_json(manifest_path)
        if manifest_path.is_file()
        else {}
    )

    print("dataset_id     =", manifest.get("dataset_id"))
    print("dataset_rows   =", manifest.get("dataset_rows"))
    print("markets        =", len(manifest.get("markets") or []))
    print("timeframe      =", manifest.get("timeframe"))
    print("horizon_bars   =", manifest.get("horizon_bars"))

    metrics = dict(manifest.get("metrics") or {})

    print(
        "alpha_auc      =",
        metrics.get("alpha_auc"),
    )
    print(
        "positive_oos_net_proxy =",
        metrics.get("positive_oos_net_proxy"),
    )
    print(
        "selected_mean_net =",
        metrics.get(
            "selected_mean_net_after_cost_floor"
        ),
    )
    print(
        "market_balanced_net =",
        metrics.get(
            "selected_market_balanced_mean_net"
        ),
    )
    print(
        "positive_market_fraction =",
        metrics.get(
            "selected_positive_market_fraction"
        ),
    )
    print(
        "stochastic_validation_pass =",
        metrics.get("stochastic_validation_pass"),
    )
    print(
        "point_in_time_universe_qualified =",
        metrics.get(
            "point_in_time_universe_qualified"
        ),
    )

    hpo_summary = dict(metrics.get("hpo") or {})

    print(
        "hpo_contract_status =",
        hpo_summary.get("contract_status"),
    )
    print(
        "hpo_contract_compatible =",
        hpo_summary.get("contract_compatible"),
    )
    print(
        "hpo_contract_mismatches =",
        hpo_summary.get("contract_mismatches"),
    )


# ------------------------------------------------------------
# HPO
# ------------------------------------------------------------

section("HPO")

hpo = read_json(
    OUT / "hpo/best.json"
)

print("generated_at =", hpo.get("generated_at"))
print("dataset_id   =", hpo.get("dataset_id"))

heads = dict(hpo.get("heads") or {})

for name, row in heads.items():
    row = dict(row or {})
    print(
        f"{name:<12} "
        f"family={row.get('family')} "
        f"objective={row.get('objective')} "
        f"trial={row.get('trial_number')}"
    )

rl = dict(hpo.get("rl") or {})

if rl:
    print(
        "RL HPO       "
        f"objective={rl.get('objective')} "
        f"trial={rl.get('trial_number')}"
    )


# ------------------------------------------------------------
# RL
# ------------------------------------------------------------

section("RL")

rl_pointer = read_json(
    OUT / "agents/rl/latest.pointer.json"
)

if not rl_pointer:
    print("STATUS=NO_POINTER")
else:
    print("status       =", rl_pointer.get("status"))
    print("qualified    =", rl_pointer.get("qualified"))
    print(
        "selected_seed =",
        rl_pointer.get("selected_seed"),
    )

    manifest_path = Path(
        str(rl_pointer.get("manifest_path") or "")
    )

    rl_manifest = (
        read_json(manifest_path)
        if manifest_path.is_file()
        else {}
    )

    metrics = dict(
        rl_manifest.get("test_metrics") or {}
    )

    for key in (
        "mean_return",
        "median_return",
        "mean_excess_vs_buy_hold",
        "worst_maximum_drawdown",
    ):
        print(f"{key:<28} =", metrics.get(key))

    stochastic = dict(
        rl_manifest.get("stochastic_validation") or {}
    )

    print(
        "stochastic_pass =",
        stochastic.get("passed"),
    )


# ------------------------------------------------------------
# TCN + GRU
# ------------------------------------------------------------

section("TCN + GRU")

tcn_pointer = read_json(
    OUT / "agents/tcn_gru/latest.pointer.json"
)

if not tcn_pointer:
    print("STATUS=NOT_TRAINED")
else:
    print("status     =", tcn_pointer.get("status"))
    print("qualified  =", tcn_pointer.get("qualified"))
    print(
        "updated_at =",
        tcn_pointer.get("updated_at")
        or tcn_pointer.get("trained_at"),
    )

    manifest_path = Path(
        str(tcn_pointer.get("manifest_path") or "")
    )

    tcn_manifest = (
        read_json(manifest_path)
        if manifest_path.is_file()
        else {}
    )

    print("dataset_id =", tcn_manifest.get("dataset_id"))
    print("rows       =", tcn_manifest.get("row_count"))
    print("markets    =", tcn_manifest.get("markets"))
    print("metrics    =", tcn_manifest.get("metrics"))


# ------------------------------------------------------------
# CMC
# ------------------------------------------------------------

section("CMC PIT RESEARCH")

cmc = read_json(
    OUT / "research/cmc_pit_feature_research.json"
)

print("status       =", cmc.get("status"))
print("generated_at =", cmc.get("generated_at"))
print("markets      =", cmc.get("market_count"))
print("features     =", cmc.get("feature_count"))
print("pairs        =", cmc.get("pair_count"))

print()
print("TOP CMC FEATURES")

for row in (cmc.get("aggregate") or [])[:15]:
    print(
        f"{str(row.get('feature')):<50} "
        f"markets={row.get('market_count')} "
        f"median_ic={row.get('median_ic')} "
        f"median_abs_ic={row.get('median_abs_ic')}"
    )


# ------------------------------------------------------------
# PROMOTION
# ------------------------------------------------------------

section("LIVE MODEL GOVERNOR")

promotion = read_json(
    OUT / "agents/live_promotion_status.json"
)

print(json.dumps(promotion, indent=2, default=str))


# ------------------------------------------------------------
# SUMMARY GATES
# ------------------------------------------------------------

section("READY CHECK")

failures = []

if manifest:
    metrics = dict(manifest.get("metrics") or {})
    heads = dict(
        manifest.get("head_qualifications") or {}
    )

    if not (heads.get("alpha") or heads.get("return")):
        failures.append("NO_DIRECTIONAL_HEAD")

    if metrics.get("positive_oos_net_proxy") is not True:
        failures.append("NO_POSITIVE_OOS_NET")

    if (
        metrics.get("point_in_time_universe_qualified")
        is not True
    ):
        failures.append("PIT_UNIVERSE_NOT_QUALIFIED")

    if (
        metrics.get("stochastic_validation_pass")
        is not True
    ):
        failures.append("STOCHASTIC_VALIDATION_FAILED")

else:
    failures.append("NO_SUPERVISED_MANIFEST")

if failures:
    print("MODEL_INFLUENCE_READY=NO")
    print("FAILURES=" + ",".join(failures))
else:
    print("MODEL_INFLUENCE_READY=YES")

print()
print(
    "NOTE: model influence readiness is NOT exchange/capital authority."
)
