from __future__ import annotations

from datetime import datetime, timezone
import json
import pandas as pd

from crypto_ai_swing.agents.dataset import DEFAULT_FEATURES
from crypto_ai_swing.agents.rl_env import build_long_only_env
from crypto_ai_swing.data.features import build_features

def train_ppo_challenger(settings, *, frame: pd.DataFrame, total_timesteps: int = 50_000) -> dict:
    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise RuntimeError("Install RL extras with: pip install -e '.[ai]'") from exc
    feat = build_features(frame).loc[:, list(DEFAULT_FEATURES)].dropna()
    returns = frame["close"].pct_change().shift(-1).reindex(feat.index)
    env = build_long_only_env(feat, returns)
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=min(1024,max(128,len(feat)//4)),
                batch_size=64, gamma=.995, gae_lambda=.95, ent_coef=.005, seed=17, verbose=0)
    model.learn(total_timesteps=int(total_timesteps))
    root = settings.project_root/"output/crypto_ai_swing/agents/rl"
    root.mkdir(parents=True, exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path=root/f"ppo_challenger_{stamp}"
    model.save(str(path))
    manifest={
        "schema_version":"swing_rl_challenger_v1","algorithm":"PPO","status":"SHADOW",
        "live_decision_influence":False,"long_only":True,"shorting":False,"leverage":False,
        "trained_at":datetime.now(timezone.utc).isoformat(),"timesteps":int(total_timesteps),
        "artifact":str(path.with_suffix(".zip").resolve()),"automatic_live_promotion":False,
    }
    (root/f"ppo_challenger_{stamp}.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest
