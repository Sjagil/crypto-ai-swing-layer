from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from crypto_ai_swing.bridge.crypto_operations import REUSED_NATIVE_INTERFACES
from crypto_ai_swing.bridge.ownership import (
    MANDATORY_CRYPTO_DOMAINS,
    audit_source_ownership,
    load_ownership_contract,
)
from crypto_ai_swing.contracts import Side
from crypto_ai_swing.strategies.swing import build_signal


def _row() -> pd.Series:
    return pd.Series({"trend_8_20": 1.0, "trend_20_50": 1.0, "trend_50_200": 1.0,
                      "rsi_14": 55.0, "ret_8": 0.02, "breakout_20": 1.0,
                      "volume_z_48": 1.0, "atr_pct": 0.02})


def test_round39_ownership_contract_is_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = load_ownership_contract(root)
    assert contract.fail_closed is True
    assert contract.canonical_repository == "Sjagil/crypto"
    assert MANDATORY_CRYPTO_DOMAINS.issubset(contract.crypto_owned)
    assert not (contract.crypto_owned & contract.swing_owned)


def test_round39_native_contract_exposes_canonical_math_risk_execution() -> None:
    assert {"expectancy", "kelly_fraction", "fractional_kelly", "calculate_position_size"}.issubset(
        set(REUSED_NATIVE_INTERFACES["research.trading_math"]))
    assert {"PositionExposure", "PortfolioSnapshot", "KillSwitch", "RiskManager"}.issubset(
        set(REUSED_NATIVE_INTERFACES["risk.risk_manager"]))
    assert {"BitvavoSpotClient", "ExecutionMarketRules", "plan_bounded_entry_order"}.issubset(
        set(REUSED_NATIVE_INTERFACES["execution.execution"]))


def test_round39_source_ownership_audit_has_no_new_violations() -> None:
    root = Path(__file__).resolve().parents[1]
    report = audit_source_ownership(root)
    assert report["ready"] is True
    assert report["violations"] == []
    assert report["round40_required"] is True


def test_round39_strategy_uses_configurable_ensemble() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/crypto_ai_swing/strategies/swing.py").read_text(encoding="utf-8")
    assert "ensemble_score(votes" in source
    assert "weighted = dscore * 0.60" not in source


def test_round39_minimum_model_probability_vetoes_entry() -> None:
    signal = build_signal("BTC-EUR", datetime.now(UTC), _row(), ml_probability=0.54,
                          minimum_model_probability=0.55, minimum_entry_score=0.0)
    assert signal.side is Side.HOLD
