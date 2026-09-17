from pathlib import Path

import pandas as pd

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge


def test_historical_path_prefers_flat_canonical(
    tmp_path,
    monkeypatch,
):
    processed = tmp_path / "normalized"

    flat = processed / "BTC-EUR_1h.parquet"
    provider = (
        processed
        / "bitvavo"
        / "BTC-EUR"
        / "1h.parquet"
    )

    flat.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    provider.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=10,
                freq="1h",
                tz="UTC",
            ),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
    ).to_parquet(flat)

    pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=1,
                freq="1h",
                tz="UTC",
            ),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
    ).to_parquet(provider)

    bridge = object.__new__(
        CryptoLibraryBridge
    )

    class Paths:
        processed_data_dir = processed

    class Settings:
        paths = Paths()

    monkeypatch.setattr(
        bridge,
        "settings",
        lambda: Settings(),
    )

    result = bridge.historical_path(
        "BTC-EUR",
        "1h",
        provider="bitvavo",
    )

    assert result == flat
