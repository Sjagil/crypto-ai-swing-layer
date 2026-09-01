import pytest

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge


def test_quantity_field_drives_cvd_and_notional():
    ticker = {"best_bid": "99", "best_ask": "101"}
    trades = [
        {"side": "buy", "quantity": "2", "price": "100"},
        {"side": "sell", "quantity": "1", "price": "101"},
    ]
    book = {
        "bids": [["99", "5"]],
        "asks": [["101", "5"]],
    }
    result = CryptoLibraryBridge.microstructure_features(ticker, trades, book)
    assert result["buy_volume"] == pytest.approx(2.0)
    assert result["sell_volume"] == pytest.approx(1.0)
    assert result["cvd_ratio"] == pytest.approx(1.0 / 3.0)
    assert result["buy_notional"] == pytest.approx(200.0)
    assert result["sell_notional"] == pytest.approx(101.0)
    assert result["cvd_notional_ratio"] == pytest.approx(99.0 / 301.0)
