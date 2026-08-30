from crypto_ai_swing.execution.shadow import ShadowLedger


def test_shadow_ledger_idempotent(tmp_path):
    ledger = ShadowLedger(tmp_path / "shadow.sqlite")
    try:
        assert ledger.append("abc", "TEST", {"x": 1}) is True
        assert ledger.append("abc", "TEST", {"x": 1}) is False
        assert ledger.count() == 1
        assert ledger.verify_chain() is True
    finally:
        ledger.close()
