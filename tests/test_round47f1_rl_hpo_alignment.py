from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_round47f1_rl_hpo_uses_train_return_target():
    text = (
        ROOT
        / "src/crypto_ai_swing/agents/hpo.py"
    ).read_text()

    assert 'block["__round47f1_target"]' in text
    assert 'target=train_target_full' in text
    assert '"feature_selection_target": "TRAIN_NEXT_RETURN_ONLY"' in text


def test_round47f1_rl_hpo_study_identity_contains_feature_contract():
    text = (
        ROOT
        / "src/crypto_ai_swing/agents/hpo.py"
    ).read_text()

    assert '"noise_control_version": NOISE_CONTROL_VERSION' in text
    assert '"feature_columns": list(features)' in text
    assert '"feature_group_counts": feature_group_counts(features)' in text


def test_round47f1_old_unsupervised_rl_hpo_selector_is_gone():
    text = (
        ROOT
        / "src/crypto_ai_swing/agents/hpo.py"
    ).read_text()

    old = (
        "features = select_train_only_features("
        "full, tuple(full.columns), target=None"
    )
    assert old not in text
