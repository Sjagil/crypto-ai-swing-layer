from __future__ import annotations

from pathlib import Path


def test_lean_bridge_uses_external_canonical_crypto_checkout() -> None:
    from crypto_ai_swing.bridge.runtime import activate_canonical_crypto
    from crypto_ai_swing.settings import Settings

    root = Path(__file__).resolve().parents[1]
    settings = Settings.load(root)
    bridge = activate_canonical_crypto(settings)
    module = bridge.import_module("core.autonomous_live")
    crypto_root = Path(settings.crypto_repo_root).resolve()
    assert crypto_root in Path(module.__file__).resolve().parents
    assert crypto_root != root


def test_swing_repo_does_not_vendor_crypto_runtime() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "core", "execution", "portfolio", "risk", "research", "reporting",
        "scrapers", "notifications", "ml", "ui", "utils", "data_store",
        "crypto-references",
    ):
        assert not (root / name).exists(), name
