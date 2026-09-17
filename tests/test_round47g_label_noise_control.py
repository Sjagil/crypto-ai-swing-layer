from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crypto_ai_swing.agents.label_noise import (
    LABEL_NOISE_VERSION,
    alpha_label_confidence,
    alpha_label_noise_summary,
    economic_alpha_sample_weight,
    label_noise_band_bps,
)


ROOT = Path(__file__).resolve().parents[1]


def _frame(net_bps):
    cost_floor = 65.0 / 10_000.0
    net = np.asarray(net_bps, dtype=float) / 10_000.0
    realized = cost_floor + net
    return pd.DataFrame(
        {
            "target_forward_return": realized,
            "target_net_return": net,
            "target_mae": np.full(len(net), 0.004),
            "target_mfe": np.full(len(net), 0.010),
            "target_alpha": (net > 0.0).astype(float),
        }
    )


def test_round47g_band_is_cost_aware_and_bounded():
    band = label_noise_band_bps(cost_floor=65.0 / 10_000.0)
    assert 10.0 <= band <= 30.0
    assert abs(band - 16.25) < 1e-9


def test_round47g_boundary_confidence_is_symmetric_and_monotonic():
    frame = _frame([-40.0, -16.25, -8.0, 0.0, 8.0, 16.25, 40.0])
    c = alpha_label_confidence(frame, cost_floor=65.0 / 10_000.0)
    assert c[3] < c[2] < c[1]
    assert c[3] < c[4] < c[5]
    assert abs(c[0] - c[-1]) < 1e-12
    assert abs(c[1] - c[-2]) < 1e-12
    assert abs(c[2] - c[-3]) < 1e-12
    assert c[0] == 1.0
    assert c[-1] == 1.0


def test_round47g_ambiguous_rows_receive_lower_training_weight():
    frame = _frame([-50.0, -20.0, -1.0, 0.0, 1.0, 20.0, 50.0])
    weights = economic_alpha_sample_weight(
        frame,
        cost_floor=65.0 / 10_000.0,
    )
    assert weights[3] < weights[0]
    assert weights[3] < weights[-1]
    assert weights[2] < weights[1]
    assert weights[4] < weights[-2]
    assert np.isfinite(weights).all()


def test_round47g_summary_preserves_oos_semantics():
    frame = _frame([-25.0, -5.0, 0.0, 5.0, 25.0])
    summary = alpha_label_noise_summary(
        frame,
        cost_floor=65.0 / 10_000.0,
    )
    assert summary["version"] == LABEL_NOISE_VERSION
    assert summary["ambiguous_rows"] >= 3
    assert summary["validation_labels_mutated"] is False
    assert summary["test_labels_mutated"] is False
    assert summary["return_target_mutated"] is False
    assert summary["risk_target_mutated"] is False


def test_round47g_training_and_hpo_share_label_contract():
    training = (ROOT / "src/crypto_ai_swing/agents/training.py").read_text()
    hpo = (ROOT / "src/crypto_ai_swing/agents/hpo.py").read_text()
    for text in (training, hpo):
        assert "LABEL_NOISE_VERSION" in text
        assert "economic_alpha_sample_weight" in text
        assert "alpha_label_noise_summary" in text
    assert '"label_noise_version": LABEL_NOISE_VERSION' in training
    assert '"label_noise_version": LABEL_NOISE_VERSION' in hpo


def test_round47g_hpo_alpha_identity_includes_label_contract():
    text = (ROOT / "src/crypto_ai_swing/agents/hpo.py").read_text()
    assert 'LABEL_NOISE_VERSION if str(head) == "alpha"' in text


def test_round47g_config_is_explicit():
    text = (ROOT / "config/agents.yaml").read_text()
    assert "label_noise:" in text
    assert "boundary_band_fraction_of_cost: 0.25" in text
    assert "minimum_confidence: 0.05" in text
