from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from crypto_ai_swing.orchestration.unified_runtime import UnifiedAutonomyRuntime
from crypto_ai_swing.research.promotion import ResearchPromotionRegistry


def test_forward_execution_timeframe_uses_autonomy_config(tmp_path: Path):
    obj = UnifiedAutonomyRuntime.__new__(UnifiedAutonomyRuntime)
    obj.settings = SimpleNamespace(
        autonomy={
            "forward_evidence": {
                "execution_timeframe": "15m",
                "mature_on_cycle": True,
            }
        },
        proactive={"execution_timeframe": "1h"},
    )
    assert obj._execution_timeframe() == "15m"


def test_registry_reads_meta_edge_artifact(tmp_path: Path):
    meta = tmp_path / "output/crypto_ai_swing/agents/meta"
    meta.mkdir(parents=True)
    (meta / "latest.json").write_text(
        json.dumps(
            {
                "status": "COLLECTING",
                "observations": 25,
                "qualified": False,
            }
        ),
        encoding="utf-8",
    )
    settings = SimpleNamespace(project_root=tmp_path)
    registry = ResearchPromotionRegistry(settings)
    payload = registry.refresh()
    assert payload["status"] == "READY"
    edge = payload["challengers"][0]
    assert edge["kind"] == "edge"
    assert edge["state"] == "COLLECTING"
    assert edge["observations"] == 25
    assert edge["qualified"] is False
