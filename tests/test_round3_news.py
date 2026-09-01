from datetime import datetime, timezone
from types import SimpleNamespace

from crypto_ai_swing.intelligence.crypto_news import CryptoNewsCollector


class FakeBridge:
    def import_module(self, name):
        assert name == "scrapers.rss"

        async def collect_registered_feeds():
            record = SimpleNamespace(
                title="Bitcoin ETF inflows rise",
                summary="Institutional inflow increased.",
                usable_at=datetime(2026, 8, 30, 12, tzinfo=timezone.utc),
                source="Test Feed",
                url="https://example.test/a",
                event_id="evt1",
                markets=("BTC-EUR",),
                categories=("crypto_news",),
                relevance_score=1.0,
                sentiment_score=0.5,
                impact_score=0.4,
                timestamp_quality="SOURCE_REPORTED",
                historical_coverage="HISTORICAL",
            )
            feed = SimpleNamespace(
                model_dump=lambda mode="json": {
                    "feed_id": "TEST",
                    "publisher": "Test Feed",
                    "status": "OK",
                    "record_count": 1,
                }
            )
            return SimpleNamespace(
                status="OK",
                records=(record,),
                feeds=(feed,),
                observed_at=datetime(2026, 8, 30, 12, 1, tzinfo=timezone.utc),
            )

        return SimpleNamespace(collect_registered_feeds=collect_registered_feeds)


def test_crypto_news_collector_converts_native_records(tmp_path):
    collector = CryptoNewsCollector(
        FakeBridge(),
        output_path=tmp_path / "news.jsonl",
        cache_seconds=0,
    )
    snapshot = collector.collect(force=True)
    assert snapshot.status == "OK"
    assert len(snapshot.documents) == 1
    assert snapshot.documents[0].title == "Bitcoin ETF inflows rise"
    assert snapshot.documents[0].metadata["markets"] == ("BTC-EUR",)
    assert (tmp_path / "news.jsonl").exists()
