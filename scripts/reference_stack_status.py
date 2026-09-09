#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_ai_swing.bridge.reference_stack import (
    ReferenceStack,
)


if __name__ == "__main__":
    print(
        json.dumps(
            ReferenceStack(ROOT).status(),
            indent=2,
            sort_keys=True,
        )
    )
