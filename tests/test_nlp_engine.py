from crypto_ai_swing.nlp.engine import NLPMarketEngine


def test_nlp_detects_security_risk_and_asset():
    engine = NLPMarketEngine({"transformer_enabled": False})
    a = engine.assess_text("Solana suffers major exploit and hack with stolen funds")
    assert a.score < 0
    assert a.severe_negative
    assert "SOL" in a.assets
    assert "SECURITY_INCIDENT" in a.event_tags


def test_nlp_positive_finance_context():
    engine = NLPMarketEngine({"transformer_enabled": False})
    a = engine.assess_text("Bitcoin ETF approved with record institutional inflow")
    assert a.score > 0
    assert "BTC" in a.assets
