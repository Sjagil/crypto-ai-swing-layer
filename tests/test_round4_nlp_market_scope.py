from datetime import datetime, timezone

from crypto_ai_swing.nlp.engine import NLPDocument, NLPMarketEngine


NOW = datetime(2026, 8, 30, 18, 0, tzinfo=timezone.utc)


def doc(title, *, markets=(), categories=("crypto_news",), impact=0.5):
    return NLPDocument(
        text=title,
        title=title,
        usable_at=NOW,
        source="test",
        metadata={
            "markets": markets,
            "categories": categories,
            "relevance_score": 1.0,
            "impact_score": impact,
        },
    )


def engine():
    return NLPMarketEngine(
        {
            "transformer_enabled": False,
            "severe_veto_direct_asset_only": True,
            "direct_asset_weight": 1.0,
            "btc_cross_asset_weight": 0.2,
            "macro_weight": 0.25,
            "generic_weight": 0.1,
        }
    )


def test_btc_hack_does_not_severe_veto_eth():
    result = engine().aggregate_with_diagnostics(
        [doc("Bitcoin hacked after major exploit", markets=("BTC-EUR",))],
        "ETH-EUR",
        now=NOW,
    )
    assert result.assessment.severe_negative is False
    assert result.diagnostics["scope_counts"]["btc_cross_asset"] == 1


def test_direct_eth_hack_can_veto_eth():
    result = engine().aggregate_with_diagnostics(
        [doc("Ethereum hacked after exploit", markets=("ETH-EUR",))],
        "ETH-EUR",
        now=NOW,
    )
    assert result.assessment.severe_negative is True
    assert result.diagnostics["scope_counts"]["direct_asset"] == 1


def test_generic_security_story_is_context_not_global_veto():
    result = engine().aggregate_with_diagnostics(
        [doc("Crypto card hack and breach", categories=("crypto_news", "hack_exploit"))],
        "SOL-EUR",
        now=NOW,
    )
    assert result.assessment.severe_negative is False
    assert result.diagnostics["scope_counts"]["generic_crypto"] == 1
