from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_ai_swing.agents.feature_denoising import feature_group
from crypto_ai_swing.agents.hpo import HPOModelFactory
from crypto_ai_swing.agents.label_noise import LABEL_NOISE_VERSION
from crypto_ai_swing.agents.multitimeframe import (
    MTF_FUSION_VERSION,
    align_closed_feature_frame,
    horizon_bars_for_hours,
    required_context_health,
    resolve_mtf_policy,
    validate_selected_mtf_features,
)


def test_round47h1_preserves_four_hour_target_on_15m():
    assert horizon_bars_for_hours("15m", 4.0) == 16


def test_round47h1_closed_bar_alignment_blocks_open_1h_candle():
    target = pd.date_range(
        "2026-01-01 10:00:00",
        periods=8,
        freq="15min",
        tz="UTC",
    )
    source = pd.DataFrame(
        {"value": [10.0, 11.0]},
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2026-01-01 10:00:00", tz="UTC"),
                pd.Timestamp("2026-01-01 11:00:00", tz="UTC"),
            ]
        ),
    )
    aligned = align_closed_feature_frame(
        target,
        target_timeframe="15m",
        source=source,
        source_timeframe="1h",
        prefix="mtf_1h__",
    )
    assert np.isnan(
        aligned.loc[
            pd.Timestamp("2026-01-01 10:30:00", tz="UTC"),
            "mtf_1h__value",
        ]
    )
    assert aligned.loc[
        pd.Timestamp("2026-01-01 10:45:00", tz="UTC"),
        "mtf_1h__value",
    ] == 10.0
    assert aligned.loc[
        pd.Timestamp("2026-01-01 11:30:00", tz="UTC"),
        "mtf_1h__value",
    ] == 10.0


def test_round47h1_selector_has_timeframe_groups():
    assert feature_group("mtf_1h__crypto_idx_rsi_14") == "mtf_1h"
    assert feature_group("mtf_2h__trend_score") == "mtf_2h"
    assert feature_group("mtf_4h__crypto_vwap_distance_20") == "mtf_4h"
    assert feature_group("mtf_1d__swing_trend") == "mtf_1d"
    assert feature_group("mtf_1w__trend_score") == "mtf_1w"
    assert feature_group("mtf_cross__trend_mean") == "mtf_cross"
    assert feature_group("htf_4h_trend_bullish") == "mtf_4h"


def test_round47h1_selected_feature_gate_requires_all_required_timeframes():
    policy = resolve_mtf_policy(
        {
            "minimum_selected_per_required_timeframe": 1,
            "minimum_selected_cross_timeframe": 1,
        }
    )
    features = [
        "mtf_1h__a",
        "mtf_2h__b",
        "mtf_4h__c",
        "mtf_1d__d",
        "mtf_cross__e",
    ]
    counts = validate_selected_mtf_features(features, policy=policy)
    assert counts["mtf_1h"] == 1
    assert counts["mtf_2h"] == 1
    assert counts["mtf_4h"] == 1
    assert counts["mtf_1d"] == 1
    assert counts["mtf_cross"] == 1


def test_round47h1_runtime_health_fails_closed_when_required_context_missing():
    policy = resolve_mtf_policy({})
    frame = pd.DataFrame(
        {
            "mtf_1h__present": [1.0],
            "mtf_1h__age_bars": [0.5],
            "mtf_2h__present": [1.0],
            "mtf_2h__age_bars": [0.5],
            "mtf_4h__present": [0.0],
            "mtf_4h__age_bars": [0.5],
            "mtf_1d__present": [1.0],
            "mtf_1d__age_bars": [0.5],
        },
        index=[pd.Timestamp("2026-01-01", tz="UTC")],
    )
    result = required_context_health(frame, policy=policy)
    assert result["ready"] is False
    assert result["required"]["4h"]["ready"] is False


def test_round47h1_stale_single_timeframe_hpo_is_rejected(tmp_path):
    root = Path(tmp_path)
    state_dir = root / "output/crypto_ai_swing/hpo"
    state_dir.mkdir(parents=True)
    (state_dir / "best.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "dataset_id": "old",
                "timeframe": "1h",
                "horizon_bars": 4,
                "feature_columns": ["a", "b"],
                "label_noise_version": LABEL_NOISE_VERSION,
                "heads": {
                    "alpha": {"family": "extra_trees", "params": {}}
                },
            }
        )
    )

    class Settings:
        project_root = root
        agents = {"hpo": {"maximum_state_age_hours": 9999}}

    factory = HPOModelFactory(
        Settings(),
        timeframe="15m",
        horizon_bars=16,
        dataset_id="new",
        feature_columns=("mtf_1h__a", "mtf_cross__b"),
        mtf_version=MTF_FUSION_VERSION,
        label_noise_version=LABEL_NOISE_VERSION,
    )
    summary = factory.summary()
    assert summary["contract_compatible"] is False
    assert factory.alpha_candidate() is None
    assert "TIMEFRAME_MISMATCH" in summary["contract_mismatches"]


def test_round47h1_config_is_15m_but_rl_waits_for_h2():
    root = Path(__file__).resolve().parents[1]
    text = (root / "config/agents.yaml").read_text()
    assert "timeframe: 15m" in text
    assert "horizon_bars: 16" in text
    assert "multitimeframe:" in text
    rl_block = text.split("rl:", 1)[1].split("research_agent_panel:", 1)[0]
    assert "timeframe: 1h" in rl_block

def test_round47h1_legacy_factory_without_explicit_contract_remains_compatible(tmp_path):
    root = Path(tmp_path)
    state_dir = root / "output/crypto_ai_swing/hpo"
    state_dir.mkdir(parents=True)
    (state_dir / "best.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "heads": {
                    "alpha": {
                        "family": "mlp",
                        "params": {
                            "hidden_layer_sizes": [64, 32],
                            "alpha": 0.0001,
                            "batch_size": 64,
                            "learning_rate_init": 0.0005,
                            "max_iter": 240,
                            "activation": "relu",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    class Settings:
        project_root = root
        agents = {
            "hpo": {
                "maximum_state_age_hours": 9999,
            }
        }

    factory = HPOModelFactory(Settings())
    summary = factory.summary()

    assert summary["strict_contract_validation"] is False
    assert summary["contract_compatible"] is True
    assert factory.alpha_candidate() is not None

