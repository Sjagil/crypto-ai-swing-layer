from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge


def probability_diagnostics(y_true, probability, bins: int = 10) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.clip(np.asarray(probability, dtype=float).reshape(-1), 1e-6, 1 - 1e-6)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("unaligned probability arrays")
    base = float(y.mean())
    baseline = np.full_like(p, base, dtype=float)
    brier = float(brier_score_loss(y, p))
    base_brier = float(brier_score_loss(y, baseline))
    edges = np.linspace(0, 1, max(2, int(bins)) + 1)
    rows = []
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (p >= lo) & ((p <= hi) if i == len(edges) - 2 else (p < hi))
        count = int(mask.sum())
        if not count:
            continue
        pred = float(p[mask].mean())
        obs = float(y[mask].mean())
        ece += count / len(y) * abs(pred - obs)
        rows.append(
            {
                "lower": float(lo),
                "upper": float(hi),
                "count": count,
                "mean_probability": pred,
                "observed_rate": obs,
                "absolute_gap": abs(pred - obs),
            }
        )
    return {
        "observations": len(y),
        "positive_rate": base,
        "brier": brier,
        "baseline_brier": base_brier,
        "brier_skill": float(1 - brier / max(base_brier, 1e-12)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "average_precision": (
            float(average_precision_score(y, p))
            if len(np.unique(y)) > 1
            else None
        ),
        "expected_calibration_error": float(ece),
        "reliability_bins": rows,
    }


def _trial_path(
    frame: pd.DataFrame,
    mask: np.ndarray,
    cost: float,
    horizon: int,
) -> pd.Series:
    realized = frame["target_forward_return"].to_numpy(float)
    mask = np.asarray(mask, bool)
    table = pd.DataFrame(
        {
            "t": pd.to_datetime(frame["feature_time"], utc=True),
            "r": np.where(mask, realized - cost, 0.0),
        }
    )
    return (
        table.groupby("t")["r"]
        .mean()
        .sort_index()
        .iloc[:: max(1, int(horizon))]
        .astype(float)
    )


def native_model_selection_evidence(
    validation: pd.DataFrame,
    probabilities: Mapping[str, np.ndarray],
    threshold_rows: Mapping[str, list[dict]],
    *,
    cost_floor: float,
    horizon_bars: int,
    crypto_repo_root: Path,
    known_trial_count: int | None = None,
    bootstrap_samples: int = 2_000,
    block_size: int = 5,
) -> dict[str, Any]:
    paths = {}
    for name, probability in probabilities.items():
        for row in threshold_rows.get(name, []):
            threshold = float(row["threshold"])
            key = f"{name}@{threshold:.3f}"
            paths[key] = _trial_path(
                validation,
                np.asarray(probability) >= threshold,
                cost_floor,
                horizon_bars,
            ).rename(key)

    if not paths:
        return {"status": "NO_TRIAL_PATHS", "known_trial_count": 0}

    matrix = pd.concat(paths.values(), axis=1).fillna(0.0)
    current_trials = len(paths)
    global_trials = max(current_trials, int(known_trial_count or current_trials))
    if len(matrix) < 12:
        return {
            "status": "INSUFFICIENT_NON_OVERLAPPING_OBSERVATIONS",
            "known_trial_count": global_trials,
            "current_trial_count": current_trials,
            "observation_count": len(matrix),
        }

    bridge = CryptoLibraryBridge(Path(crypto_repo_root))
    fn = bridge.import_module(
        "research.optimization"
    ).multiple_testing_bootstrap
    result = fn(
        matrix,
        bootstrap_samples=max(2_000, int(bootstrap_samples)),
        block_size=max(1, int(block_size)),
        seed=17,
        known_trial_count=global_trials,
    )
    return {
        "status": "READY",
        **asdict(result),
        "current_trial_count": current_trials,
        "global_known_trial_count": global_trials,
        "native_backend": (
            "Sjagil/crypto:research.optimization.multiple_testing_bootstrap"
        ),
    }


def native_selected_return_evidence(
    test: pd.DataFrame,
    selected: np.ndarray,
    *,
    cost_floor: float,
    horizon_bars: int,
    crypto_repo_root: Path,
    simulations: int = 10_000,
) -> dict[str, Any]:
    path = _trial_path(test, selected, cost_floor, horizon_bars)
    if len(path) < 3:
        return {
            "status": "INSUFFICIENT_PATH",
            "path_observations": len(path),
            "passed": False,
        }

    bridge = CryptoLibraryBridge(Path(crypto_repo_root))
    statistical = bridge.import_module("research.statistical_evidence")
    stochastic = bridge.import_module("research.stochastic_validation")
    native_settings = bridge.settings().research

    hac = statistical.hac_effective_sample_size(path)
    policy = stochastic.StochasticValidationPolicy(
        simulations=max(
            2_000,
            int(simulations),
            int(native_settings.monte_carlo_runs),
        ),
        expected_block_length=max(2, int(horizon_bars)),
        maximum_drawdown=float(native_settings.maximum_drawdown),
        maximum_drawdown_breach_probability=float(
            native_settings.maximum_monte_carlo_probability_of_20pct_drawdown
        ),
        maximum_terminal_loss_probability=float(
            native_settings.maximum_monte_carlo_probability_of_loss
        ),
        minimum_p05_total_return=float(
            native_settings.minimum_stochastic_p05_total_return
        ),
        dirichlet_blocks=int(native_settings.dirichlet_block_count),
        minimum_observations=30,
        confidence_level=float(native_settings.confidence_level),
        seed=17,
    )
    stationary = stochastic.stationary_bootstrap_monte_carlo(
        path.to_numpy(float),
        policy=policy,
    )
    dirichlet = stochastic.dirichlet_time_concentration_stress(
        path.to_numpy(float),
        policy=policy,
    )
    passed = bool(
        float(path.mean()) > 0.0
        and stationary.get("passed") is True
        and dirichlet.get("passed") is True
    )
    return {
        "status": "READY",
        "path_observations": len(path),
        "mean_net": float(path.mean()),
        "median_net": float(path.median()),
        "hac": hac,
        "stationary_bootstrap": stationary,
        "dirichlet_time_concentration": dirichlet,
        "passed": passed,
        "native_backends": [
            "Sjagil/crypto:research.statistical_evidence.hac_effective_sample_size",
            "Sjagil/crypto:research.stochastic_validation.stationary_bootstrap_monte_carlo",
            "Sjagil/crypto:research.stochastic_validation.dirichlet_time_concentration_stress",
        ],
    }


def native_hac_evidence(
    test: pd.DataFrame,
    selected: np.ndarray,
    *,
    cost_floor: float,
    horizon_bars: int,
    crypto_repo_root: Path,
) -> dict[str, Any]:
    evidence = native_selected_return_evidence(
        test,
        selected,
        cost_floor=cost_floor,
        horizon_bars=horizon_bars,
        crypto_repo_root=crypto_repo_root,
        simulations=2_000,
    )
    return {
        "status": evidence.get("status"),
        "path_observations": evidence.get("path_observations"),
        "mean_net": evidence.get("mean_net"),
        "hac": evidence.get("hac"),
        "native_backend": (
            "Sjagil/crypto:research.statistical_evidence.hac_effective_sample_size"
        ),
    }
