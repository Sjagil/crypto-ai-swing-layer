from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class RLEnvConfig:
    fee_bps_per_turnover: float = 30.0
    slippage_bps_per_turnover: float = 5.0
    drawdown_penalty: float = 0.20

def build_long_only_env(feature_frame: pd.DataFrame, returns: pd.Series, *, config: RLEnvConfig | None = None):
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except Exception as exc:
        raise RuntimeError("Install RL extras with: pip install -e '.[ai]'") from exc
    cfg = config or RLEnvConfig()
    x = feature_frame.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    r = returns.reindex(x.index).fillna(0).astype(float)
    if len(x) < 100:
        raise ValueError("RL environment requires >=100 rows")

    class LongOnlySwingEnv(gym.Env):
        metadata = {"render_modes": []}
        def __init__(self):
            super().__init__()
            self.x = x.to_numpy(np.float32)
            self.r = r.to_numpy(np.float32)
            self.action_space = spaces.Discrete(2)  # 0 flat, 1 long
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.x.shape[1]+2,), dtype=np.float32)
            self.i = 0; self.position = 0.0; self.equity = 1.0; self.peak = 1.0
        def _obs(self):
            dd = self.equity / max(self.peak, 1e-9) - 1.0
            return np.concatenate([self.x[self.i], np.asarray([self.position, dd], np.float32)]).astype(np.float32)
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.i=0; self.position=0.0; self.equity=1.0; self.peak=1.0
            return self._obs(), {}
        def step(self, action):
            target = 1.0 if int(action)==1 else 0.0
            turnover = abs(target-self.position)
            cost = turnover*(cfg.fee_bps_per_turnover+cfg.slippage_bps_per_turnover)/10000
            gross = self.position*float(self.r[min(self.i+1, len(self.r)-1)])
            net = gross-cost
            self.equity *= max(1e-6, 1+net)
            self.peak=max(self.peak,self.equity)
            dd=max(0.0,1-self.equity/max(self.peak,1e-9))
            reward=net-cfg.drawdown_penalty*dd
            self.position=target; self.i+=1
            done=self.i>=len(self.x)-2
            return self._obs(), float(reward), done, False, {"equity":self.equity,"turnover":turnover}
    return LongOnlySwingEnv()
