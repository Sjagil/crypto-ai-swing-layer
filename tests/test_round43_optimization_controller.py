from __future__ import annotations

from types import SimpleNamespace

from crypto_ai_swing.research.optimization_controller import (
    OptimizationController,
)


def _settings(tmp_path):
    return SimpleNamespace(
        project_root=tmp_path,
        autonomy={
            "entry_selector": {
                "minimum_train_observations": 1,
            },
            "optimization_controller": {
                "enabled": True,
                "minimum_complete_horizon_observations_for_optuna": 2,
                "minimum_complete_horizon_observations_for_exact": 2,
                "minimum_new_primary_outcomes_for_research": 1,
                "minimum_new_complete_horizon_observations_for_exact": 1,
                "minimum_new_complete_horizon_observations_for_optuna": 1,
                "research_cooldown_seconds": 999999,
                "exact_research_cooldown_seconds": 999999,
                "selector_optuna_cooldown_seconds": 999999,
            },
        },
    )


def test_round43_optimization_is_evidence_triggered_and_bounded(tmp_path):
    controller = OptimizationController(_settings(tmp_path))
    evidence = {
        "primary_horizon_outcomes": 2,
        "complete_required_horizon_observations": 2,
        "minimum_discovery_observations": 2,
    }
    drift = {
        "issue_codes": ["NO_CHAMPION_WITH_DISCOVERY_EVIDENCE"],
    }
    agent_status = {
        "supervised": {"status": "READY"},
        "rl": {"status": "READY"},
    }
    strategy_status = {"champion": None}

    first = controller.plan(
        evidence=evidence,
        drift=drift,
        agent_status=agent_status,
        strategy_status=strategy_status,
    )
    assert first["directives"]["force_research"] is True
    assert first["directives"]["force_exact_research"] is True
    assert first["directives"]["run_selector_optuna"] is True
    assert first["threshold_relaxation_allowed"] is False
    assert first["automatic_live_authority"] is False

    controller.record_execution(
        plan=first,
        evidence=evidence,
        actions={},
    )
    second = controller.plan(
        evidence=evidence,
        drift=drift,
        agent_status=agent_status,
        strategy_status=strategy_status,
    )
    assert second["directives"]["force_research"] is False
    assert second["directives"]["force_exact_research"] is False
    assert second["directives"]["run_selector_optuna"] is False
