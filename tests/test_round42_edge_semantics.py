from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_round42_net_edge_is_after_canonical_costs():
    text = (
        ROOT / "src/crypto_ai_swing/bridge/canonical_portfolio.py"
    ).read_text(encoding="utf-8")
    assert (
        "net_edge_bps = float(packet.expected_edge_bps) "
        "- round_trip_cost_bps"
    ) in text
    assert "if net_edge_bps <= minimum_net_edge_bps" in text
