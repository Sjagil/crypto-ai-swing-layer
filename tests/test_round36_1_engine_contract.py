from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


def test_engine_contract_lock_targets_configured_crypto_checkout() -> None:
    from crypto_ai_swing.settings import Settings

    root = Path(__file__).resolve().parents[1]
    settings = Settings.load(root)
    crypto = Path(settings.crypto_repo_root).resolve()
    lock = json.loads(
        (root / "config/canonical_crypto_engine.json").read_text(
            encoding="utf-8"
        )
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=crypto,
        text=True,
    ).strip()
    assert lock["commit"] == head
    assert lock["repository"] == "Sjagil/crypto"


def test_private_probe_contains_no_order_submission_surface() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/diagnose_private_account.py").read_text(
        encoding="utf-8"
    )
    assert "submit_order(" not in source
    lowered = source.casefold()
    assert "/withdraw" not in lowered
    assert ".withdraw(" not in lowered
    assert "withdraw(" not in lowered
    assert 'signed_path = "/v2/balance"' in source
    assert 'bridge.import_module("aiohttp")' not in source
    assert "import aiohttp" in source


def test_live_engine_contract_fails_on_head_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crypto_ai_swing.bridge import engine_contract

    swing = tmp_path / "swing"
    crypto = tmp_path / "crypto"
    (swing / "config").mkdir(parents=True)
    crypto.mkdir()
    (swing / "config/canonical_crypto_engine.json").write_text(
        json.dumps({"commit": "expected"}),
        encoding="utf-8",
    )

    class Settings:
        crypto_repo_root = crypto

    values = {
        ("rev-parse", "HEAD"): "actual",
        ("branch", "--show-current"): "main",
        ("status", "--porcelain"): "",
    }

    monkeypatch.setattr(
        engine_contract,
        "_git",
        lambda _root, *args: values[tuple(args)],
    )
    with pytest.raises(RuntimeError, match="differs from the pinned"):
        engine_contract.validate_engine_contract(
            swing,
            settings=Settings(),
        )
