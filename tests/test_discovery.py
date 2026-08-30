from crypto_ai_swing.bridge.discovery import discover


def test_discovery_finds_known_crypto_path(tmp_path):
    path = tmp_path / "src/data/providers/coinmarketcap_client.py"
    path.parent.mkdir(parents=True)
    path.write_text("# test", encoding="utf-8")
    items = {x.capability: x for x in discover(tmp_path)}
    assert items["coinmarketcap_client"].found
