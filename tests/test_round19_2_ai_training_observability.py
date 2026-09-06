from __future__ import annotations

import json

from crypto_ai_swing.monitoring.ai_training import collect_ai_training_state


def test_ai_training_observer_reads_config_and_artifact(tmp_path):
    project = tmp_path / "swing"
    crypto = tmp_path / "crypto"
    (project / "config").mkdir(parents=True)
    (project / "output" / "training").mkdir(parents=True)
    crypto.mkdir()

    (project / "config" / "agents.yaml").write_text(
        """
research_agent_panel:
  authority: ADVISORY_ONLY
  live_decision_influence: false
  agents:
    - alpha
    - regime
    - bayesian_edge
""".strip()
    )
    artifact = {
        "status": "SHADOW",
        "validation_auc": 0.525,
        "live_decision_influence": False,
    }
    (project / "output" / "training" / "alpha_training.json").write_text(
        json.dumps(artifact)
    )

    payload = collect_ai_training_state(project, crypto)

    assert payload["authority"] == "READ_ONLY_OBSERVABILITY"
    assert payload["live_decision_influence"] is False
    assert any(row["component"] == "alpha" for row in payload["agents"])
    assert any(
        row["component"] == "alpha_training"
        for row in payload["training"]
    )
