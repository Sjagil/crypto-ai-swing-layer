from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from dotenv import dotenv_values
except Exception:  # pragma: no cover - python-dotenv is a project dependency
    dotenv_values = None


_SWING_PREFIX = "CRYPTO_SWING_"


def hydrate_canonical_environment(root: Path) -> dict[str, Any]:
    """Make the canonical Sjagil/crypto `.env` authoritative for its keys.

    The swing repository is an orchestrator. Provider, exchange, risk and live
    authority settings belong to the canonical crypto repository. Therefore a
    stale shell or swing `.env` value must not silently override the canonical
    value. Only `CRYPTO_SWING_*` variables remain outside canonical ownership.

    The function returns key names only. Secret values are never returned.
    """

    env_path = Path(root).resolve() / ".env"
    if dotenv_values is None or not env_path.is_file():
        return {
            "status": "NO_CANONICAL_ENV",
            "path": str(env_path),
            "imported_keys": [],
            "overridden_keys": [],
            "conflict_keys": [],
            "secret_values_exposed": False,
        }

    raw = dotenv_values(env_path)
    precedence = str(
        os.environ.get("CRYPTO_SWING_CANONICAL_ENV_PRECEDENCE", "canonical")
    ).strip().casefold()
    canonical_wins = precedence not in {"process", "existing", "shell"}
    imported: list[str] = []
    overridden: list[str] = []
    conflicts: list[str] = []

    for key, value in raw.items():
        name = str(key)
        if name.startswith(_SWING_PREFIX):
            continue
        if value is None or not str(value).strip():
            continue
        selected = str(value)
        existing = os.environ.get(name)
        if existing is None or not str(existing).strip():
            os.environ[name] = selected
            imported.append(name)
            continue
        if str(existing) == selected:
            continue
        conflicts.append(name)
        if canonical_wins:
            os.environ[name] = selected
            overridden.append(name)

    return {
        "status": "LOADED",
        "path": str(env_path),
        "precedence": "canonical" if canonical_wins else "process",
        "imported_keys": sorted(imported),
        "overridden_keys": sorted(overridden),
        "conflict_keys": sorted(set(conflicts)),
        "secret_values_exposed": False,
    }


__all__ = ["hydrate_canonical_environment"]
