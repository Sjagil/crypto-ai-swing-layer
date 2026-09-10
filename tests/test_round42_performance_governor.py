from types import SimpleNamespace

from crypto_ai_swing.agents.performance_governor import PerformanceGovernor


class _Ops:
    def atomic_write_json(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")


def test_round42_governor_targets_real_failure_modes(tmp_path):
    settings = SimpleNamespace(
        project_root=tmp_path,
        crypto_repo_root=tmp_path,
        autonomy={"performance_governor": {}},
    )
    governor = PerformanceGovernor(settings)
    governor.operations = _Ops()
    payload = governor.evaluate(
        agent_status={
            "supervised": {
                "status": "SHADOW",
                "head_qualifications": {
                    "alpha": False,
                    "return": False,
                    "regime": True,
                    "risk": True,
                    "execution": True,
                },
                "metrics": {
                    "positive_oos_net_proxy": False,
                    "model_selection_pass": False,
                    "stochastic_validation_pass": False,
                    "point_in_time_universe_qualified": False,
                    "return_test": {"skill_vs_zero": -0.01},
                },
            },
            "rl": {
                "status": "SHADOW",
                "qualified": False,
                "stochastic_validation": {"passed": False},
                "test_metrics": {
                    "mean_excess_vs_buy_hold": -0.2,
                    "worst_maximum_drawdown": 0.4,
                },
            },
        }
    )
    codes = {row["code"] for row in payload["priorities"]}
    assert "IMPROVE_ALPHA_OOS_EDGE" in codes
    assert "IMPROVE_RETURN_HEAD" in codes
    assert "IMPROVE_RL_ROBUSTNESS" in codes
    assert "POINT_IN_TIME_UNIVERSE" in codes
    assert payload["ready_for_more_model_influence"] is False
    assert payload["automatic_live_promotion"] is False
