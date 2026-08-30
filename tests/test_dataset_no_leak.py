import numpy as np
import pandas as pd
from crypto_ai_swing.data.dataset import supervised_dataset


def test_last_horizon_rows_are_not_training_rows():
    n = 300
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = np.linspace(100, 150, n)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=idx,
    )
    x, y, future = supervised_dataset(df, horizon_bars=8)
    assert x.index.max() <= idx[-9]
    assert len(x) == len(y) == len(future)
