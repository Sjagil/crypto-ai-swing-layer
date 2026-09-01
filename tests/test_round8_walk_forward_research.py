import pandas as pd

from crypto_ai_swing.research.bootstrap import (
    _candidate_grid,
    _diversified_shortlist,
    _walk_forward_slices,
)


def test_round8_candidate_grid_is_broader_but_bounded():
    candidates = _candidate_grid()
    assert len(candidates) == 38
    families = {candidate.family for candidate in candidates}
    assert families == {
        "TREND_PULLBACK",
        "DONCHIAN_BREAKOUT",
        "MOMENTUM_24",
        "MEAN_REVERSION_UPTREND",
        "VOL_EXPANSION",
    }


def test_final_holdout_shortlist_has_family_diversity():
    rows = []
    for index in range(8):
        rows.append({"candidate_id": index, "family": "DONCHIAN_BREAKOUT"})
    for index in range(8, 12):
        rows.append({"candidate_id": index, "family": "MOMENTUM_24"})
    for index in range(12, 16):
        rows.append({"candidate_id": index, "family": "TREND_PULLBACK"})
    selected = _diversified_shortlist(rows, 5, maximum_per_family=2)
    counts = {}
    for row in selected:
        counts[row["family"]] = counts.get(row["family"], 0) + 1
    assert len(selected) == 5
    assert max(counts.values()) <= 2
    assert len(counts) >= 3


def test_walk_forward_development_windows_never_touch_final_holdout():
    index = pd.date_range("2025-01-01", periods=1000, freq="1h", tz="UTC")
    frame = pd.DataFrame({"close": range(1000)}, index=index)
    folds = _walk_forward_slices(
        frame,
        folds=3,
        purge=4,
        development_fraction=0.80,
    )
    assert len(folds) == 3
    final_holdout_start = frame.index[800]
    for fold in folds:
        validation = fold["validation"]
        assert not validation.empty
        assert validation.index.max() < final_holdout_start
