from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.settings import Settings


def canonical_crypto_root(settings: Settings | None = None) -> Path:
    selected = settings or Settings.load()
    root = Path(selected.crypto_repo_root).expanduser().resolve()
    if not (root / ".git").is_dir():
        raise RuntimeError(f"canonical crypto checkout unavailable: {root}")
    if not (root / "core").is_dir() or not (root / "execution").is_dir():
        raise RuntimeError(f"canonical crypto checkout incomplete: {root}")
    return root


def activate_canonical_crypto(settings: Settings | None = None) -> CryptoLibraryBridge:
    selected = settings or Settings.load()
    root = canonical_crypto_root(selected)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return CryptoLibraryBridge(root)


def canonical_module(name: str, settings: Settings | None = None) -> Any:
    return activate_canonical_crypto(settings).import_module(name)
