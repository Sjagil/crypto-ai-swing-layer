from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib


class ModelRegistry:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, name: str, artifact: Path, metrics: dict, status: str = "SHADOW") -> Path:
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact.exists() else None
        record = {
            "name": name,
            "artifact": str(artifact),
            "sha256": digest,
            "status": status,
            "metrics": metrics,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "live_authority": False,
        }
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return path
