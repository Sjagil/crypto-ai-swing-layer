from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from crypto_ai_swing.orchestration.unified_runtime import UnifiedAutonomyRuntime
from crypto_ai_swing.research.promotion import ResearchPromotionRegistry


def test_registry_never_grants_live_authority(tmp_path: Path):
    settings = SimpleNamespace(project_root=tmp_path)
    payload = ResearchPromotionRegistry(settings).refresh()
    assert payload["promotion_scope"] == "RESEARCH_ONLY"
    assert payload["live_decision_influence"] is False
    assert payload["automatic_live_promotion"] is False
    assert payload["manual_live_authority_still_required"] is True


def test_due_is_fail_safe_for_missing_timestamp():
    runtime = object.__new__(UnifiedAutonomyRuntime)
    runtime.state = {}
    assert runtime._due("anything", 60) is True


def test_task_spec_contract_is_nonempty():
    runtime = object.__new__(UnifiedAutonomyRuntime)
    runtime.cfg = {}
    runtime._mature_forward = lambda: {}
    runtime._refresh_edge = lambda: {}
    runtime._registry_refresh = lambda: {}
    runtime._rl_tournament = lambda: {}
    specs = runtime._task_specs()
    assert {spec.name for spec in specs} == {
        "forward_maturation",
        "edge_policy",
        "registry",
        "rl_tournament",
    }
    assert all(spec.every_seconds > 0 for spec in specs)
