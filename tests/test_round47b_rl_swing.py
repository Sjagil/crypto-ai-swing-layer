from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_ai_swing.agents.rl_multi_market import _env

def test_minimum_hold_prevents_one_bar_churn():
    idx = pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC")
    x = pd.DataFrame({"a": np.zeros(20), "b": np.ones(20)}, index=idx)
    r = pd.Series(np.zeros(20), index=idx)
    env = _env(x, r, minimum_hold_bars=3, cooldown_bars=2)
    env.reset()
    _, _, _, _, first = env.step(1)
    assert first["turnover"] == 1.0
    _, _, _, _, second = env.step(0)
    assert second["executed_target"] == 1.0
    assert second["turnover"] == 0.0
