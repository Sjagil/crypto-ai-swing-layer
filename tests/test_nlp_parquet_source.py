from datetime import datetime, timezone

import pandas as pd
import pytest

from crypto_ai_swing.nlp.sources import discover_crypto_repo_documents


def test_crypto_repo_parquet_intelligence_is_discovered(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "data_store/context/intelligence/articles.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "title": "Bitcoin ETF institutional inflow",
                "summary": "Record inflow into Bitcoin ETF",
                "usable_at": datetime.now(timezone.utc),
                "source": "unit",
            }
        ]
    ).to_parquet(path)
    docs = discover_crypto_repo_documents(tmp_path)
    assert len(docs) == 1
    assert "Bitcoin" in docs[0].title
