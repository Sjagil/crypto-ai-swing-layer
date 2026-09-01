from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd

from crypto_ai_swing.research.bootstrap import (
    Candidate,
    ColdStartResearchRunner,
    _raw_signals,
)


def _frame(seed: int, rows: int = 520):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=rows, freq="1h", tz="UTC")
    ret = rng.normal(0.0002, 0.012, rows)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0005, 0.006, rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0005, 0.006, rows))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": rng.uniform(10, 100, rows)},
        index=idx,
    )


class FakeCrypto:
    def __init__(self, frames):
        self.frames = frames

    def ohlcv_many(self, markets, timeframe, persist=False, concurrency=4):
        return {market: self.frames[market] for market in markets}


class FakeUniverse:
    def __init__(self, markets):
        self.markets = markets

    def current(self):
        return {
            "markets": self.markets,
            "policy": {"hard_maximum_spread_bps": 35},
            "candidates": [
                {"market": market, "spread_bps": 2.0 + i}
                for i, market in enumerate(self.markets)
            ],
        }


def test_research_signals_have_no_fillna_futurewarning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _raw_signals(
            _frame(1),
            Candidate("MOMENTUM_24", {"ret_24": 0.02, "rsi_max": 75, "maximum_bars": 48}),
        )
    assert not [item for item in caught if issubclass(item.category, FutureWarning)]


def test_final_holdout_is_only_evaluated_for_validation_shortlist(tmp_path):
    markets = [f"M{i}-EUR" for i in range(5)]
    frames = {market: _frame(10 + i) for i, market in enumerate(markets)}
    settings = SimpleNamespace(
        project_root=tmp_path,
        crypto_repo_root=tmp_path,
        universe={},
        execution={"costs": {"fee_bps_per_side": 25, "base_slippage_bps": 2}},
    )
    runner = ColdStartResearchRunner(settings)
    runner.crypto = FakeCrypto(frames)
    runner.universe = FakeUniverse(markets)
    payload = runner.run(final_holdout_shortlist_size=3)
    assert payload["test_used_for_candidate_selection"] is False
    assert payload["holdout_tested_count"] == 3
    evaluated = [row for row in payload["top_candidates"] if row["final_holdout_evaluated"]]
    assert len(evaluated) == 3
    not_evaluated = [row for row in payload["top_candidates"] if not row["final_holdout_evaluated"]]
    assert all(row["aggregate"]["test"]["evaluated"] is False for row in not_evaluated)
