import pandas as pd

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge


def test_higher_timeframe_frame_is_lagged_to_decision_close(tmp_path):
    bridge = CryptoLibraryBridge(tmp_path)
    idx = pd.to_datetime([
        "2026-08-29T00:00:00Z",
        "2026-08-30T00:00:00Z",
    ])
    frame = pd.DataFrame({"close": [100.0, 110.0]}, index=idx)
    decision = pd.Timestamp("2026-08-30T12:00:00Z")
    causal = bridge.causal_frame(frame, "1d", decision)
    assert list(causal.index) == [pd.Timestamp("2026-08-29T00:00:00Z")]


def test_quote_volume_uses_direct_quote_value_first(tmp_path):
    bridge = CryptoLibraryBridge(tmp_path)
    assert bridge.quote_volume_24h({"volumeQuote": "1234567"}) == 1234567.0


def test_quote_volume_can_be_inferred_from_base_volume_and_price(tmp_path):
    bridge = CryptoLibraryBridge(tmp_path)
    assert bridge.quote_volume_24h({"volume": "10", "price": "250"}) == 2500.0
