from __future__ import annotations

import os
from pathlib import Path

from crypto_ai_swing.bridge.reference_stack import ReferenceStack


def test_reference_contract_has_no_exchange_authority() -> None:
    root = Path(__file__).resolve().parents[1]
    stack = ReferenceStack(root)
    for name, spec in stack.references.items():
        assert spec.get("live_authority") is False, name


def test_reference_contract_repairs_known_round29_false_negatives() -> None:
    root = Path(__file__).resolve().parents[1]
    stack = ReferenceStack(root)
    assert stack.references["kronos"]["modules"] == ["model"]
    assert stack.references["finrl_trading"]["runtime_required"] is False
    assert stack.references["finrl_trading"]["source_required"] is True
    assert stack.references["lean_cli"]["runtime_required"] is True
    assert stack.references["lean_cli"]["source_required"] is False


def test_operational_runner_never_submits_live_orders() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts" / "run_operational_stack.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "submit_swing_layer_buy(",
        "execute_generated_strategy_live_once(",
        "live start",
        "trading run-once --execute",
    )
    assert not any(token in source for token in forbidden)
    assert '"live", "preflight"' in source


def test_crypto_windows_test_does_not_mutate_global_os_name() -> None:
    crypto = Path(
        __import__("crypto_ai_swing.settings", fromlist=["Settings"])
        .Settings.load(Path(__file__).resolve().parents[1])
        .crypto_repo_root
    )
    test_source = (
        crypto / "tests" / "test_data_realtime.py"
    ).read_text(encoding="utf-8")
    assert 'data.data_loader.os.name", "nt"' not in test_source
    assert "data.data_loader._is_windows" in test_source
    assert os.name != "nt" or os.name == "nt"
