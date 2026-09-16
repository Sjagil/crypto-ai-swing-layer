from __future__ import annotations

import inspect
from pathlib import Path

from crypto_ai_swing.agents.feature_denoising import (
    MAX_FEATURE_SELECTION_ROWS,
    select_stable_train_features,
)
from crypto_ai_swing.agents.multitimeframe import (
    MTF_FEATURE_SOURCE,
    MTF_FUSION_VERSION,
    is_operational_mtf_feature,
)
from crypto_ai_swing.agents.rl_multi_market import (
    RL_STATE_VERSION,
    _environment_kwargs,
)


def test_round47h_final_contract_versions():
    assert MTF_FUSION_VERSION == "round47h_final_canonical_mtf_v2"
    assert MTF_FEATURE_SOURCE == "canonical_mtf_feature_pipeline_v2"
    assert RL_STATE_VERSION == "round47h_final_mtf_position_state_v2"


def test_round47h_final_operational_metadata_filtered():
    assert is_operational_mtf_feature("mtf_4h__age_bars")
    assert is_operational_mtf_feature("mtf_1d__present")
    assert not is_operational_mtf_feature("mtf_4h__crypto_idx_rsi_14")


def test_round47h_final_selection_memory_is_bounded_and_train_only():
    assert MAX_FEATURE_SELECTION_ROWS <= 100_000
    source = inspect.getsource(select_stable_train_features).lower()
    assert "_bounded_train_selection_sample" in source
    assert "validation" not in source
    assert "test" not in source


def test_round47h_final_15m_rl_keeps_old_clock_semantics():
    timing = _environment_kwargs("15m")
    assert timing["minimum_hold_bars"] == 24
    assert timing["cooldown_bars"] == 8


def test_round47h_final_rl_runtime_uses_exact_mtf_contract():
    from crypto_ai_swing.agents.rl_runtime import RLRuntime

    source = inspect.getsource(RLRuntime.predict_frame)
    assert "MTF_FEATURE_SOURCE" in source
    assert "build_runtime_mtf_feature_frame" in source
    assert "RL_MTF_CONTRACT_MISMATCH" in source
    assert "RL_MTF_CONTEXT_NOT_READY" in source


def test_round47h_final_config_is_mtf_rl_and_caps_are_not_edited_here():
    root = Path(__file__).resolve().parents[1]
    text = (root / "config/agents.yaml").read_text()
    rl_block = text.split("rl:", 1)[1].split("research_agent_panel:", 1)[0]
    assert "timeframe: 15m" in rl_block
    assert "minimum_rows_per_market: 8000" in rl_block
    assert "rl_maximum_features: 80" in text
