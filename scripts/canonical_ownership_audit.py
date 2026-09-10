#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from crypto_ai_swing.bridge.ownership import audit_source_ownership

root = Path(__file__).resolve().parents[1]
report = audit_source_ownership(root)
print(json.dumps(report, indent=2, sort_keys=True))
raise SystemExit(0 if report["ready"] else 2)
