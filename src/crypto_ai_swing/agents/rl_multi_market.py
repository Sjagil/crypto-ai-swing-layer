from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from crypto_ai_swing.agents.canonical_features import (
    canonical_model_frame,
    select_train_only_features,
)
from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge


CANDIDATE_SEEDS = (17, 29, 43)


def _prepare(
    bridge: CryptoLibraryBridge,
    market: str,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    features = canonical_model_frame(
        bridge,
        frame,
        market=market,
        timeframe=str(frame.attrs.get("timeframe") or "1h"),
        benchmark=None,
    ).replace([np.inf, -np.inf], np.nan)
    next_return = (
        pd.to_numeric(frame["close"], errors="coerce")
        .pct_change()
        .shift(-1)
        .reindex(features.index)
    )
    joined = features.copy()
    joined["__next_return"] = next_return
    joined = joined.dropna(subset=["__next_return"])
    return joined.drop(columns=["__next_return"]), joined["__next_return"]

def _normalize(
    frame: pd.DataFrame,
    mean: pd.Series,
    std: pd.Series,
) -> pd.DataFrame:
    safe_std = std.replace(0.0, 1.0).fillna(1.0)
    return (
        (frame - mean) / safe_std
    ).clip(-10.0, 10.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _env(feature_frame: pd.DataFrame, returns: pd.Series, *, extra_stress_bps: float = 0.0):
    import gymnasium as gym
    from gymnasium import spaces

    x = feature_frame.to_numpy(np.float32)
    r = returns.to_numpy(np.float32)
    normal_turnover_cost = 35.0 / 10_000.0
    extra = float(extra_stress_bps) / 10_000.0

    class LongOnlyEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(
                -np.inf,
                np.inf,
                shape=(x.shape[1] + 2,),
                dtype=np.float32,
            )
            self.i = 0
            self.position = 0.0
            self.equity = 1.0
            self.peak = 1.0

        def _obs(self):
            dd = self.equity / max(self.peak, 1e-9) - 1.0
            return np.concatenate(
                [x[self.i], np.asarray([self.position, dd], np.float32)]
            ).astype(np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.i = 0
            self.position = 0.0
            self.equity = 1.0
            self.peak = 1.0
            return self._obs(), {}

        def step(self, action):
            target = 1.0 if int(action) == 1 else 0.0
            turnover = abs(target - self.position)
            gross = target * float(r[self.i])
            normal_cost = turnover * normal_turnover_cost
            stress_cost = turnover * extra
            net = gross - normal_cost - stress_cost
            self.equity *= max(1e-8, 1.0 + net)
            self.peak = max(self.peak, self.equity)
            drawdown = max(0.0, 1.0 - self.equity / max(self.peak, 1e-9))
            # Reward favors net return but penalizes persistent drawdown and churn.
            reward = net - 0.15 * drawdown - 0.10 * turnover * normal_turnover_cost
            self.position = target
            self.i += 1
            done = self.i >= len(x) - 1
            return self._obs(), float(reward), done, False, {
                "equity": self.equity,
                "drawdown": drawdown,
                "net_return": net,
                "turnover": turnover,
            }

    return LongOnlyEnv()


def _evaluate(model, x: pd.DataFrame, r: pd.Series, *, stress_bps: float = 0.0):
    env = _env(x, r, extra_stress_bps=stress_bps)
    obs, _ = env.reset()
    done = False
    max_dd = 0.0
    net_returns = []
    turnovers = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        max_dd = max(max_dd, float(info.get("drawdown", 0.0)))
        net_returns.append(float(info.get("net_return", 0.0)))
        turnovers.append(float(info.get("turnover", 0.0)))
    strategy = float(env.equity - 1.0)
    buy_hold = float(np.prod(1.0 + r.iloc[:len(net_returns)].to_numpy(float)) - 1.0)
    index = x.index[:len(net_returns)]
    return {
        "strategy_return": strategy,
        "buy_hold_return": buy_hold,
        "excess_vs_buy_hold": strategy - buy_hold,
        "maximum_drawdown": max_dd,
        "turnover_events": int(np.sum(np.asarray(turnovers) > 0)),
        "step_returns": pd.Series(net_returns, index=index, dtype=float),
    }


def _metrics(model, split: dict[str, tuple[pd.DataFrame, pd.Series]]):
    per_market = {}
    series = {}
    for market, (x, r) in sorted(split.items()):
        result = _evaluate(model, x, r)
        series[market] = result.pop("step_returns")
        per_market[market] = result
    returns = np.asarray([row["strategy_return"] for row in per_market.values()], dtype=float)
    excess = np.asarray([row["excess_vs_buy_hold"] for row in per_market.values()], dtype=float)
    drawdowns = np.asarray([row["maximum_drawdown"] for row in per_market.values()], dtype=float)
    portfolio = pd.concat(series, axis=1).mean(axis=1).dropna() if series else pd.Series(dtype=float)
    return {
        "market_count": len(per_market),
        "mean_return": float(returns.mean()) if len(returns) else -1.0,
        "median_return": float(np.median(returns)) if len(returns) else -1.0,
        "positive_market_fraction": float(np.mean(returns > 0.0)) if len(returns) else 0.0,
        "mean_excess_vs_buy_hold": float(excess.mean()) if len(excess) else -1.0,
        "worst_maximum_drawdown": float(drawdowns.max()) if len(drawdowns) else 1.0,
        "per_market": per_market,
        "portfolio_step_returns": portfolio,
    }


def _bayesian_market_returns(metrics: dict[str, Any], draws: int = 5000) -> dict[str, Any]:
    values = np.asarray(
        [row["strategy_return"] for row in metrics["per_market"].values()],
        dtype=float,
    )
    if len(values) == 0:
        return {"status": "NO_DATA"}
    rng = np.random.default_rng(2201)
    weights = rng.dirichlet(np.ones(len(values)), size=draws)
    means = weights @ values
    return {
        "status": "READY",
        "markets": len(values),
        "mean_return": float(values.mean()),
        "median_return": float(np.median(values)),
        "probability_cross_market_mean_positive": float(np.mean(means > 0.0)),
        "cross_market_mean_p05": float(np.quantile(means, 0.05)),
        "cross_market_mean_p50": float(np.quantile(means, 0.50)),
        "cross_market_mean_p95": float(np.quantile(means, 0.95)),
    }


class MultiMarketRLTrainer:
    """Leakage-controlled shared PPO tournament.

    Candidate seeds are selected on validation only. The final test set is
    touched once by the selected seed. Feature scaling is fitted on train only.
    """

    def __init__(self, settings) -> None:
        self.settings = settings
        self.crypto = CryptoLibraryBridge(settings.crypto_repo_root)

    def _stochastic(self, normal: pd.Series, stressed: pd.Series) -> dict[str, Any]:
        if len(normal) < 30:
            return {
                "status": "COLLECTING",
                "passed": False,
                "observations": len(normal),
                "required": 30,
            }
        try:
            module = self.crypto.import_module("research.stochastic_validation")
            policy = module.StochasticValidationPolicy(
                simulations=10_000,
                expected_block_length=max(5, min(20, len(normal) // 20)),
                maximum_drawdown=0.15,
                maximum_drawdown_breach_probability=0.01,
                maximum_terminal_loss_probability=0.05,
                minimum_p05_total_return=0.0,
                dirichlet_blocks=12,
                dirichlet_concentrations=(0.5, 1.0, 5.0),
                minimum_observations=30,
                confidence_level=0.95,
                seed=2201,
                batch_size=256,
            )
            return module.validate_strategy_return_paths(
                normal.to_numpy(float),
                stressed.to_numpy(float),
                policy=policy,
                seed_offset=0,
            )
        except Exception as exc:
            return {
                "status": "ERROR",
                "passed": False,
                "error": f"{type(exc).__name__}:{str(exc)[:300]}",
            }

    def train(
        self,
        *,
        markets: list[str],
        timeframe: str = "1h",
        total_timesteps: int = 50_000,
        minimum_rows_per_market: int = 900,
    ) -> dict[str, Any]:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        frames = self.crypto.ohlcv_many(
            markets,
            timeframe,
            persist=False,
            concurrency=4,
        )
        raw = {}
        for market, frame in frames.items():
            if frame is None or frame.empty:
                continue
            x, r = _prepare(self.crypto, market, frame)
            if len(x) >= minimum_rows_per_market:
                raw[market] = (x, r)

        if len(raw) < 6:
            raise ValueError(f"RL usable markets {len(raw)} < 6")

        train_raw = {}
        validation_raw = {}
        test_raw = {}
        for market, (x, r) in raw.items():
            n = len(x)
            train_end = int(n * 0.60)
            validation_end = int(n * 0.80)
            if train_end < 400 or validation_end - train_end < 150 or n - validation_end < 150:
                continue
            train_raw[market] = (x.iloc[:train_end].copy(), r.iloc[:train_end].copy())
            validation_raw[market] = (
                x.iloc[train_end:validation_end].copy(),
                r.iloc[train_end:validation_end].copy(),
            )
            test_raw[market] = (
                x.iloc[validation_end:].copy(),
                r.iloc[validation_end:].copy(),
            )

        if len(train_raw) < 6:
            raise ValueError(f"RL split-usable markets {len(train_raw)} < 6")

        train_matrix_full = pd.concat([x for x, _ in train_raw.values()], axis=0, sort=False)
        selected_features = select_train_only_features(
            train_matrix_full,
            tuple(train_matrix_full.columns),
            target=None,
            maximum_features=64,
        )
        if len(selected_features) < 8:
            raise ValueError(
                f"RL canonical train-only feature selection left {len(selected_features)} features"
            )

        def align(split):
            return {
                market: (x.reindex(columns=list(selected_features)), r.copy())
                for market, (x, r) in split.items()
            }

        train_raw = align(train_raw)
        validation_raw = align(validation_raw)
        test_raw = align(test_raw)
        train_matrix = pd.concat([x for x, _ in train_raw.values()], axis=0)
        feature_mean = train_matrix.mean(axis=0)
        feature_std = train_matrix.std(axis=0, ddof=0).replace(0.0, 1.0)

        def norm_split(split):
            return {
                market: (_normalize(x, feature_mean, feature_std), r.copy())
                for market, (x, r) in split.items()
            }

        train = norm_split(train_raw)
        validation = norm_split(validation_raw)
        test = norm_split(test_raw)

        per_candidate_timesteps = max(
            10_000,
            int(total_timesteps) // len(CANDIDATE_SEEDS),
        )
        candidate_rows = []
        candidate_models = {}
        for seed in CANDIDATE_SEEDS:
            factories = []
            for market in sorted(train):
                x, r = train[market]
                factories.append(
                    lambda x=x.copy(), r=r.copy(): _env(x, r)
                )
            vector_env = DummyVecEnv(factories)
            model = PPO(
                "MlpPolicy",
                vector_env,
                learning_rate=2.5e-4,
                n_steps=512,
                batch_size=128,
                gamma=0.995,
                gae_lambda=0.95,
                ent_coef=0.003,
                vf_coef=0.5,
                max_grad_norm=0.5,
                seed=seed,
                verbose=0,
            )
            model.learn(total_timesteps=per_candidate_timesteps)
            metrics = _metrics(model, validation)
            portfolio = metrics.pop("portfolio_step_returns")
            objective = (
                metrics["mean_return"]
                + 0.35 * metrics["mean_excess_vs_buy_hold"]
                + 0.20 * metrics["median_return"]
                - 0.50 * metrics["worst_maximum_drawdown"]
            )
            candidate_rows.append(
                {
                    "seed": seed,
                    "objective": float(objective),
                    "metrics": metrics,
                    "validation_portfolio_observations": len(portfolio),
                }
            )
            candidate_models[seed] = model

        selected = max(candidate_rows, key=lambda row: row["objective"])
        selected_seed = int(selected["seed"])
        model = candidate_models[selected_seed]

        test_metrics = _metrics(model, test)
        normal_portfolio = test_metrics.pop("portfolio_step_returns")

        # Stress the selected final-test policy with an extra 25 bps per
        # turnover event, without re-selecting the model.
        stressed_series = {}
        for market, (x, r) in test.items():
            stressed = _evaluate(model, x, r, stress_bps=25.0)
            stressed_series[market] = stressed.pop("step_returns")
        stressed_portfolio = (
            pd.concat(stressed_series, axis=1).mean(axis=1).dropna()
            if stressed_series
            else pd.Series(dtype=float)
        )

        bayesian = _bayesian_market_returns(test_metrics)
        stochastic = self._stochastic(normal_portfolio, stressed_portfolio)
        checks = {
            "mean_test_return": test_metrics["mean_return"] > 0.0,
            "median_test_return": test_metrics["median_return"] > 0.0,
            "positive_market_fraction": test_metrics["positive_market_fraction"] >= 0.55,
            "mean_excess_vs_buy_hold": test_metrics["mean_excess_vs_buy_hold"] > 0.0,
            "maximum_drawdown": test_metrics["worst_maximum_drawdown"] <= 0.15,
            "bayesian_probability": (
                float(bayesian.get("probability_cross_market_mean_positive", 0.0)) >= 0.95
            ),
            "bayesian_p05": float(bayesian.get("cross_market_mean_p05", -1.0)) > 0.0,
            "stochastic_validation": bool(stochastic.get("passed", False)),
        }
        qualified = all(checks.values())

        root = self.settings.project_root / "output/crypto_ai_swing/agents/rl"
        root.mkdir(parents=True, exist_ok=True)
        trained_at = datetime.now(UTC)
        identity = sha256(
            json.dumps(
                {
                    "selected_seed": selected_seed,
                    "validation": selected,
                    "test": test_metrics,
                    "qualified": qualified,
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()[:20]
        artifact = root / f"ppo_tournament_{identity}.zip"
        model.save(str(artifact))

        manifest = {
            "schema_version": "swing_rl_ppo_tournament_v3",
            "status": "SHADOW",
            "trained_at": trained_at.isoformat(),
            "algorithm": "PPO",
            "markets": sorted(train),
            "timeframe": timeframe,
            "requested_total_timesteps": int(total_timesteps),
            "candidate_timesteps": per_candidate_timesteps,
            "candidate_seeds": list(CANDIDATE_SEEDS),
            "candidate_validation": candidate_rows,
            "selected_seed": selected_seed,
            "feature_columns": list(selected_features),
            "feature_source": "canonical_feature_pipeline_v1",
            "feature_mean": {
                key: float(value)
                for key, value in feature_mean.items()
            },
            "feature_std": {
                key: float(value)
                for key, value in feature_std.items()
            },
            "test_metrics": test_metrics,
            "bayesian": bayesian,
            "stochastic_validation": stochastic,
            "qualification_checks": checks,
            "qualified": qualified,
            "long_only": True,
            "spot_only": True,
            "leverage": False,
            "shorting": False,
            "live_decision_influence": False,
            "automatic_live_promotion": False,
            "final_test_used_for_model_selection": False,
        }
        manifest_path = root / f"ppo_tournament_{identity}.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str),
            encoding="utf-8",
        )
        (root / "latest.pointer.json").write_text(
            json.dumps(
                {
                    "schema_version": "swing_rl_pointer_v3",
                    "artifact_path": str(artifact.resolve()),
                    "manifest_path": str(manifest_path.resolve()),
                    "qualified": qualified,
                    "updated_at": trained_at.isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return manifest
