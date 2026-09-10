from pathlib import Path

import yaml

from crypto_ai_swing.bridge.ownership import audit_source_ownership

ROOT = Path(__file__).resolve().parents[1]


def test_round40_has_zero_local_exchange_migration_exceptions():
    cfg = yaml.safe_load((ROOT / "config/canonical_ownership.yaml").read_text(encoding="utf-8"))
    assert cfg["legacy_migration_exceptions"] == []


def test_round40_source_ownership_is_finalized():
    report = audit_source_ownership(ROOT)
    assert report["ready"] is True
    assert report["legacy_migration_hits"] == []
    assert report["round40_required"] is False


def test_round40_local_bitvavo_implementation_is_absent():
    assert not (ROOT / "src/crypto_ai_swing/execution/bitvavo.py").exists()
