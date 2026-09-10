#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from crypto_ai_swing.intelligence.external_research import (
    ExternalResearchPlane,
)
from crypto_ai_swing.settings import Settings


def main() -> int:
    settings = Settings.load(Path.cwd())
    payload = ExternalResearchPlane(settings).collect()
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    ok = payload.get("record_count", 0) > 0 and payload.get(
        "status"
    ) in {"OK", "PARTIAL"}
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
