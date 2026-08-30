import pandas as pd
from crypto_ai_swing.data.canonical import canonicalize_ohlcv


def test_canonical_ohlcv():
    idx = pd.date_range("2025-01-01", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [10, 11, 12],
            "high": [12, 13, 14],
            "low": [9, 10, 11],
            "close": [11, 12, 13],
            "volume": [100, 200, 150],
        },
        index=idx,
    )
    out = canonicalize_ohlcv(df)
    assert list(out.index) == list(idx)
    assert out.index.name == "timestamp"
