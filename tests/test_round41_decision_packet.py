from datetime import UTC, datetime

from crypto_ai_swing.contracts import Authority, Side, Signal
from crypto_ai_swing.decision_packet import DecisionPacket


def test_round41_decision_packet_uses_only_qualified_probability():
    signal = Signal(
        market="BTC-EUR",
        timestamp=datetime.now(UTC),
        side=Side.BUY,
        score=0.75,
        confidence=0.70,
        expected_edge_bps=120.0,
        stop_pct=0.02,
        take_profit_pct=0.06,
        trailing_stop_pct=0.02,
        strategy="TEST",
        edge_source="CALIBRATED_TEST",
        features={"price": 50000.0},
    )
    packet = DecisionPacket.from_signal(
        signal,
        context={
            "ml_probability": 0.61,
            "agents": {
                "alpha_probability": 0.61,
                "diagnostics": {"head_influence": {"alpha": True}},
            },
            "prospective_context_status": "READY",
            "data_source": "Sjagil/crypto",
        },
        authority=Authority.SHADOW,
    )
    assert packet.calibrated_win_probability == 0.61
    assert packet.probability_source == "agents.calibrated_alpha"
    assert packet.prospective_qualified is True
    assert len(packet.canonical_hash()) == 64


def test_round41_unqualified_alpha_cannot_drive_kelly():
    signal = Signal(
        market="BTC-EUR",
        timestamp=datetime.now(UTC),
        side=Side.BUY,
        score=0.75,
        confidence=0.70,
        expected_edge_bps=120.0,
        stop_pct=0.02,
        take_profit_pct=0.06,
        trailing_stop_pct=0.02,
        strategy="TEST",
        edge_source="TEST",
        features={"price": 50000.0},
    )
    packet = DecisionPacket.from_signal(
        signal,
        context={
            "agents": {
                "alpha_probability": 0.99,
                "diagnostics": {"head_influence": {"alpha": False}},
            },
            "prospective_context_status": "READY",
        },
        authority=Authority.LIVE,
    )
    assert packet.calibrated_win_probability is None
