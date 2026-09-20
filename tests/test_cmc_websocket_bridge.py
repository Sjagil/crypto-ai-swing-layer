from crypto_ai_swing.intelligence.cmc_startup_bridge import (
    _normalize_crypto_ids,
)


def test_cmc_ws_scalar_crypto_id_is_supported():
    assert _normalize_crypto_ids(1) == {1}


def test_cmc_ws_list_crypto_ids_are_supported():
    assert _normalize_crypto_ids(
        [1, 1027, "5426"]
    ) == {
        1,
        1027,
        5426,
    }


def test_cmc_ws_missing_crypto_ids_is_safe():
    assert _normalize_crypto_ids(None) == set()


def test_cmc_ws_invalid_crypto_ids_are_ignored():
    assert _normalize_crypto_ids(
        ["bad", None, 1]
    ) == {1}
