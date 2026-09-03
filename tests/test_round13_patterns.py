from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_ai_swing.research.patterns import (
    latest_pattern_snapshot,
)


def test_pattern_snapshot_is_research_only_and_causal():
    periods = 260
    index = pd.date_range(
        "2025-01-01",
        periods=periods,
        freq="1h",
        tz="UTC",
    )
    close = np.linspace(100.0, 180.0, periods)
    open_ = np.r_[close[0], close[:-1]]
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 1.0,
            "low": np.minimum(open_, close) - 1.0,
            "close": close,
            "volume": np.linspace(
                1000.0,
                3000.0,
                periods,
            ),
        },
        index=index,
    )
    payload = latest_pattern_snapshot(frame)
    assert payload["status"] == "READY"
    assert payload["authority"] == "RESEARCH_ONLY"
    assert payload["live_decision_influence"] is False
    assert payload["causality"]["future_features_used"] is False
    assert "bullish_structure" in payload["patterns"]
