from crypto_ai_swing.integrations.whole_context import _descriptive,_selector_context

def test_round33_advisory_score_not_execution_score():
    r=_descriptive(.4,.5,.2,-.1); assert r["status"]=="READY"; assert r["contract"]=="DESCRIPTIVE_ADVISORY_SCORE_NOT_EXECUTION_SCORE"

def test_round33_selector_context_has_no_execution_authority():
    c={"mtf":{"swing_score":.4,"position_score":.3,"alignment_score_01":.75},"timeframes":{"15m":{"technical_score":.2,"rsi_14":52},"1h":{"technical_score":.5,"rsi_14":58},"4h":{"technical_score":.6,"breakout_state":"TREND_CONTINUATION"},"1d":{"technical_score":.3}}}; r=_selector_context("BTC-EUR",c,.1,{"sentiment_score":.2,"confidence":.7}); assert r["authority"]=="ADVISORY_ONLY"; assert r["live_decision_influence"] is False; assert r["mtf_challenger"]["execution_score"] is None
