import pandas as pd
from crypto_ai_swing.data.alignment import lagged_asof_join


def test_higher_timeframe_is_lagged():
    base_idx = pd.date_range("2025-01-01", periods=8, freq="1h", tz="UTC")
    high_idx = pd.date_range("2025-01-01", periods=3, freq="4h", tz="UTC")
    base = pd.DataFrame({"x": range(8)}, index=base_idx)
    higher = pd.DataFrame({"regime": [10.0, 20.0, 30.0]}, index=high_idx)
    joined = lagged_asof_join(base, higher, "h4_", lag_bars=1)

    # The first published higher bar cannot appear immediately because it is lagged.
    assert pd.isna(joined.iloc[0]["h4_regime"])
