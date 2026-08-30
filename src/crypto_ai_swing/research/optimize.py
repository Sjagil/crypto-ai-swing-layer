from __future__ import annotations

from typing import Callable, Any


def optimize(
    objective: Callable[[Any], float],
    trials: int = 100,
    direction: str = "maximize",
    seed: int = 17,
):
    import optuna
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=trials)
    return study
