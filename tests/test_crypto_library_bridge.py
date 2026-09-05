
from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge


def test_microstructure_features_are_bounded_and_use_book_and_trades(tmp_path):
    bridge = CryptoLibraryBridge(tmp_path)
    features = bridge.microstructure_features(
        {"bid": "99", "ask": "101"},
        [
            {"side": "buy", "amount": "3"},
            {"side": "sell", "amount": "1"},
        ],
        {
            "bids": [["99", "5"], ["98", "2"]],
            "asks": [["101", "2"], ["102", "1"]],
        },
    )
    assert features["spread_bps"] > 0
    assert features["book_imbalance"] > 0
    assert features["cvd_ratio"] == 0.5
    assert features["trade_count"] == 2.0


def test_integration_status_can_validate_selected_fake_crypto_modules(tmp_path, monkeypatch):
    (tmp_path / "fakepkg").mkdir()
    (tmp_path / "fakepkg/__init__.py").write_text("VALUE = 1\n")
    bridge = CryptoLibraryBridge(tmp_path)
    status = bridge.integration_status(("fakepkg",))
    assert status["ready"] is True
    assert status["imported_modules"] == 1
