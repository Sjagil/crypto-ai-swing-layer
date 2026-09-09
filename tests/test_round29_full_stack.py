from __future__ import annotations

from pathlib import Path

from crypto_ai_swing.bridge.reference_stack import (
    ReferenceStack,
)


def test_reference_stack_never_grants_execution_authority() -> None:
    root = Path(__file__).resolve().parents[1]
    stack = ReferenceStack(root)
    for name, raw in stack.references.items():
        assert raw.get("live_authority") is False
        assert name


def test_reference_stack_has_all_declared_roles() -> None:
    root = Path(__file__).resolve().parents[1]
    stack = ReferenceStack(root)
    expected = {
        "nautilus_trader",
        "vectorbt",
        "optuna",
        "skfolio",
        "stable_baselines3_contrib",
        "kronos",
        "finrl_trading",
        "lean",
        "lean_cli",
        "moondev",
        "qlib",
        "pybroker",
        "vnpy",
        "vnpy_ib",
    }
    assert expected.issubset(stack.references)


def test_crypto_bridge_requires_round29_canonical_modules() -> None:
    from crypto_ai_swing.bridge.crypto_library import (
        REQUIRED_CRYPTO_MODULES,
    )

    required = {
        "core.swing_layer_live",
        "execution.execution",
        "execution.bitvavo_private_errors",
        "reporting.reference_integration_health",
    }
    assert required.issubset(
        set(REQUIRED_CRYPTO_MODULES)
    )
