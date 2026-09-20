from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_ai_swing.agents.rl_multi_market import (
    RL_STATE_FEATURES,
    RL_STATE_VERSION,
    _env,
)
from crypto_ai_swing.agents.rl_runtime import (
    _runtime_live_influence_allowed,
    _runtime_state_vector,
)


def _frame(returns):
    index = pd.date_range(
        "2026-01-01",
        periods=len(returns),
        freq="h",
        tz="UTC",
    )
    features = pd.DataFrame(
        {
            "x1": np.linspace(0.0, 1.0, len(index)),
            "x2": np.ones(len(index)),
        },
        index=index,
    )
    series = pd.Series(
        returns,
        index=index,
        dtype=float,
    )
    return features, series


def _state(obs, feature_count):
    return obs[feature_count:]


def test_round47c3_observation_exposes_execution_state():
    x, r = _frame(
        [0.01] * 20
    )
    env = _env(
        x,
        r,
        minimum_hold_bars=4,
        cooldown_bars=2,
    )
    obs, _ = env.reset()

    assert len(obs) == x.shape[1] + len(RL_STATE_FEATURES)
    np.testing.assert_allclose(
        _state(obs, x.shape[1]),
        np.zeros(len(RL_STATE_FEATURES)),
    )

    obs, _, _, _, info = env.step(1)
    state = _state(
        obs,
        x.shape[1],
    )
    assert info["executed_target"] == 1.0
    assert state[0] == 1.0
    assert state[1] > 0.0
    assert state[3] != 0.0


def test_round47c3_invalid_exit_is_observable_and_blocked():
    x, r = _frame(
        [0.01] * 20
    )
    env = _env(
        x,
        r,
        minimum_hold_bars=3,
        cooldown_bars=2,
    )
    env.reset()
    env.step(1)

    obs, _, _, _, info = env.step(0)

    assert info["invalid_action"] is True
    assert info["executed_target"] == 1.0
    state = _state(
        obs,
        x.shape[1],
    )
    assert state[0] == 1.0
    assert state[1] > 0.0


def test_round47c3_runup_and_giveback_are_visible():
    x, r = _frame(
        [0.05, -0.03] + [0.0] * 18
    )
    env = _env(
        x,
        r,
        minimum_hold_bars=1,
        cooldown_bars=1,
    )
    env.reset()

    obs, _, _, _, _ = env.step(1)
    first = _state(
        obs,
        x.shape[1],
    )
    assert first[4] > 0.0

    obs, _, _, _, info = env.step(1)
    second = _state(
        obs,
        x.shape[1],
    )
    assert info["trade_giveback"] > 0.0
    assert second[5] > 0.0


def test_round47c3_cooldown_is_visible_and_blocks_reentry():
    x, r = _frame(
        [0.01] * 20
    )
    env = _env(
        x,
        r,
        minimum_hold_bars=1,
        cooldown_bars=2,
    )
    env.reset()
    env.step(1)

    obs, _, _, _, exit_info = env.step(0)
    assert exit_info["executed_target"] == 0.0
    state = _state(
        obs,
        x.shape[1],
    )
    assert state[2] == 1.0

    _, _, _, _, blocked = env.step(1)
    assert blocked["invalid_action"] is True
    assert blocked["executed_target"] == 0.0


def test_round47c3_runtime_uses_legacy_shape_for_old_artifacts():
    state = _runtime_state_vector(
        {}
    )
    assert state.shape == (2,)


def test_round47c3_runtime_uses_new_state_shape_for_new_artifacts():
    manifest = {
        "observation_state_version": RL_STATE_VERSION,
    }
    state = _runtime_state_vector(
        manifest
    )
    assert state.shape == (
        len(RL_STATE_FEATURES),
    )
    np.testing.assert_allclose(
        state,
        0.0,
    )


def test_round47c3_new_position_manager_cannot_gain_live_influence_yet():
    manifest = {
        "observation_state_version": RL_STATE_VERSION,
        "live_decision_influence": True,
        "position_manager_live_ready": False,
    }
    assert (
        _runtime_live_influence_allowed(
            manifest
        )
        is False
    )


def test_round47c3_legacy_live_contract_is_not_broken():
    manifest = {
        "live_decision_influence": True,
    }
    assert (
        _runtime_live_influence_allowed(
            manifest
        )
        is True
    )
