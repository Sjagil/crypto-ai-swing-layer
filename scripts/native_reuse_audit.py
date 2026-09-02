#!/usr/bin/env python3
from __future__ import annotations

import json

from crypto_ai_swing.bridge.crypto_operations import NativeOperationsBridge
from crypto_ai_swing.settings import Settings


def main() -> int:
    settings = Settings.load()
    bridge = NativeOperationsBridge(
        settings.crypto_repo_root,
        project_root=settings.project_root,
    )
    payload = bridge.status()
    print(json.dumps(payload, indent=2, default=str))
    return 0 if payload.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
