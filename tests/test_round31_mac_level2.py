from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_level2_swing_canary_is_configured_but_not_automatic() -> None:
    config = yaml.safe_load(
        (ROOT / "config" / "execution.yaml").read_text(encoding="utf-8")
    )
    canary = config["swing_canary"]
    assert canary["maximum_order_eur"] == 25
    assert canary["maximum_total_exposure_eur"] == 75
    assert canary["maximum_positions"] == 3
    assert canary["maximum_new_orders_per_day"] == 3
    assert canary["maximum_risk_per_trade_eur"] == 2
    assert canary["autoscale"] is False
    assert canary["automatic_authority"] is False


def test_proactive_execution_validation_uses_level2_order_cap() -> None:
    config = yaml.safe_load(
        (ROOT / "config" / "proactive.yaml").read_text(encoding="utf-8")
    )
    policy = config["active_swing"]["execution_validation_canary"]
    assert policy["maximum_order_eur"] == 25.0
    assert policy["manual_authority_required"] is True
    assert policy["prospective_evidence_required_for_scaling"] is True
    assert policy["autoscale"] is False


def test_optuna_reference_worker_is_pinned_to_major_four() -> None:
    source = (
        ROOT / "scripts" / "workers" / "reference_stack_worker.py"
    ).read_text(encoding="utf-8")
    assert "OPTUNA_MAJOR_VERSION_UNSUPPORTED" in source
    assert "major != 4" in source
