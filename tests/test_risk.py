from pathlib import Path

from crypto_ai_swing.bridge.crypto_operations import REUSED_NATIVE_INTERFACES

ROOT = Path(__file__).resolve().parents[1]


def test_risk_authority_is_canonical_crypto_repo():
    assert not (
        ROOT / "src/crypto_ai_swing/portfolio/risk.py"
    ).exists()
    risk = set(REUSED_NATIVE_INTERFACES["risk.risk_manager"])
    assert {"RiskManager", "PortfolioSnapshot", "PositionExposure"} <= risk
