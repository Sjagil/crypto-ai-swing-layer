# Reference repository roles

## NautilusTrader

Use for independent event-driven replay, fill-model tests, order lifecycle semantics and backtest cross-checks. Run it in `.venvs/nautilus`.

## vectorbt

Use for fast research scans and independent signal-return checks. It is useful for screening large parameter sets, but a vectorized pass is not enough for promotion.

## Optuna

Use for bounded parameter search. Search space, objective and trial history must be persisted. The untouched test set may not participate in optimization.

## skfolio

Use as a portfolio challenger for minimum variance, risk budgeting, hierarchical risk parity and related allocation methods. It does not choose trade direction.

## Stable-Baselines3 Contrib

Use MaskablePPO or RecurrentPPO as an RL challenger. The default config gives RL zero live authority.

## Kronos

Use as a forecast challenger through an isolated worker. Keep its forecast separate from the deterministic strategy until out-of-sample evidence justifies weight.

## FinRL-Trading

Use for benchmark environments, reward-design comparisons and RL experiment patterns. Do not let a FinRL agent call Bitvavo directly.

## LEAN

Optional cross-engine validation. It is heavier and is disabled by default.

## MoonDev Trading AI Agents

Optional research-agent patterns. It is disabled by default because an agent framework is not required to make the trading pipeline intelligent.

## Explicitly disabled by default

Qlib, PyBroker, VeighNa and vnpy_ib are not needed for the first crypto-only architecture. They can be added later if a concrete research role appears.
