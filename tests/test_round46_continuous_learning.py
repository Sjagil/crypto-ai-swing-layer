from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import joblib

from crypto_ai_swing.agents.hpo import HPOModelFactory
from crypto_ai_swing.agents.live_promotion import LiveModelGovernor


def _settings(tmp_path: Path):
    return SimpleNamespace(
        project_root=tmp_path,
        agents={
            "hpo": {"maximum_state_age_hours": 168},
            "live_promotion": {
                "minimum_supervised_rows": 10,
                "minimum_promotion_score_delta": 0.0,
            },
        },
    )


def test_hpo_factory_builds_backprop_candidate(tmp_path):
    root = tmp_path / "output/crypto_ai_swing/hpo"
    root.mkdir(parents=True)
    (root / "best.json").write_text(
        json.dumps({
            "generated_at": datetime.now(UTC).isoformat(),
            "heads": {
                "alpha": {
                    "family": "mlp",
                    "params": {
                        "hidden_layer_sizes": [64, 32],
                        "alpha": 0.0001,
                        "batch_size": 64,
                        "learning_rate_init": 0.0005,
                        "max_iter": 240,
                        "activation": "relu",
                    },
                }
            },
        }),
        encoding="utf-8",
    )
    factory = HPOModelFactory(_settings(tmp_path))
    model = factory.alpha_candidate()
    assert model is not None
    assert model.named_steps["model"].__class__.__name__ == "MLPClassifier"


def test_governor_promotes_qualified_supervised_candidate(tmp_path):
    settings = _settings(tmp_path)
    root = tmp_path / "output/crypto_ai_swing/agents"
    artifact_dir = root / "artifacts/test"
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "bundle.joblib"
    bundle = {
        "status": "SHADOW",
        "dataset_id": "dataset-1",
        "dataset_rows": 1000,
        "shadow_decision_qualified": True,
        "head_qualifications": {
            "alpha": True,
            "regime": False,
            "return": True,
            "risk": False,
            "execution": True,
        },
        "metrics": {
            "positive_oos_net_proxy": True,
            "point_in_time_universe_qualified": True,
            "selected_conservative_mean_net": 0.001,
            "selected_market_balanced_mean_net": 0.001,
            "selected_positive_market_fraction": 0.60,
            "return_test": {"skill_vs_zero": 0.05},
        },
        "models": {},
    }
    joblib.dump(bundle, artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (root / "latest.pointer.json").write_text(
        json.dumps({
            "artifact_path": str(artifact),
            "artifact_hash": digest,
        }),
        encoding="utf-8",
    )

    result = LiveModelGovernor(settings).promote_supervised()
    assert result["status"] == "PROMOTED"

    live = json.loads(
        (root / "live.pointer.json").read_text(encoding="utf-8")
    )
    promoted = joblib.load(Path(live["artifact_path"]))
    assert promoted["status"] == "CANARY"
    assert promoted["live_decision_influence"] is True
    assert promoted["execution_authority_granted"] is False
