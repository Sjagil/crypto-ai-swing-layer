from pathlib import Path

from crypto_ai_swing.bridge.crypto_operations import REUSED_NATIVE_INTERFACES

ROOT = Path(__file__).resolve().parents[1]


def test_round41_pipeline_has_no_local_portfolio_or_cost_authority():
    text = (
        ROOT / "src/crypto_ai_swing/orchestration/pipeline.py"
    ).read_text(encoding="utf-8")
    assert "crypto_ai_swing.portfolio.allocator" not in text
    assert "crypto_ai_swing.portfolio.risk" not in text
    assert "crypto_ai_swing.execution.costs" not in text
    assert "CanonicalPortfolioRiskBridge" in text
    assert "DecisionPacket" in text


def test_round41_native_contract_contains_risk_kelly_and_cost_authority():
    risk = set(REUSED_NATIVE_INTERFACES["risk.risk_manager"])
    math = set(REUSED_NATIVE_INTERFACES["research.trading_math"])
    assert {"PortfolioSnapshot", "PositionExposure", "RiskManager"} <= risk
    assert {"kelly_fraction", "fractional_kelly"} <= math


def test_round41_local_duplicate_authority_files_removed():
    assert not (
        ROOT / "src/crypto_ai_swing/portfolio/allocator.py"
    ).exists()
    assert not (
        ROOT / "src/crypto_ai_swing/portfolio/risk.py"
    ).exists()
    assert not (
        ROOT / "src/crypto_ai_swing/execution/costs.py"
    ).exists()
