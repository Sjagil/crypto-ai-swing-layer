import json
from pathlib import Path
from types import SimpleNamespace

from crypto_ai_swing.agents.manager import AgentManager


class _Runtime:
    def __init__(self, status):
        self._status = status

    def status(self):
        return {"status": self._status}


class _Trainer:
    calls = 0

    def train(self, **kwargs):
        type(self).calls += 1
        return SimpleNamespace(
            status="SHADOW",
            artifact_path=Path("/tmp/agent.joblib"),
            dataset_id="dataset-1",
            row_count=9000,
            markets=("BTC-EUR",),
            metrics={"ok": True},
        )


class _RLTrainer:
    calls = 0

    def train(self, **kwargs):
        type(self).calls += 1
        return {"status": "SHADOW", "qualified": False}


class _Edge:
    def refresh_policy(self, path, force=False):
        return {"status": "COLLECTING", "qualified": False}


class _Registry:
    def refresh(self):
        return {"status": "READY"}


class _Universe:
    def current(self):
        return {"markets": ["BTC-EUR"]}


class _Operations:
    def atomic_write_json(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")


def test_round41_agent_manager_trains_missing_agents_but_never_promotes_live(
    tmp_path,
):
    settings = SimpleNamespace(
        project_root=tmp_path,
        crypto_repo_root=tmp_path,
        agents={
            "timeframe": "1h",
            "horizon_bars": 4,
            "minimum_rows": 100,
            "minimum_net_move_bps": 65.0,
            "rl": {"enabled": True, "timeframe": "1h"},
            "manager": {
                "enabled": True,
                "supervised_retrain_seconds": 14400,
                "rl_retrain_seconds": 86400,
            },
        },
    )
    manager = AgentManager(
        settings,
        mode="shadow",
        trainer=_Trainer(),
        runtime=_Runtime("NOT_TRAINED"),
        rl_trainer=_RLTrainer(),
        rl_runtime=_Runtime("NOT_TRAINED"),
        edge_manager=_Edge(),
        registry=_Registry(),
        universe=_Universe(),
    )
    manager.operations = _Operations()
    _Trainer.calls = 0
    _RLTrainer.calls = 0

    payload = manager.cycle(
        markets=["BTC-EUR"],
        forward_database_path=tmp_path / "forward.sqlite",
    )

    assert _Trainer.calls == 1
    assert _RLTrainer.calls == 1
    assert payload["automatic_live_authority"] is False
    assert payload["automatic_model_live_promotion"] is False
    assert payload["orders_submitted"] == 0
