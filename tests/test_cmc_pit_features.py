from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from crypto_ai_swing.agents.cmc_pit_features import augment_cmc_pit_features


class _Bridge:
    def __init__(self, root: Path):
        self._settings = SimpleNamespace(paths=SimpleNamespace(data_dir=root))
    def settings(self):
        return self._settings


def test_cmc_pit_merge_never_uses_future_available_rows(tmp_path: Path):
    path = tmp_path / "coinmarketcap_startup" / "features" / "pit_features.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "market":"BTC-EUR",
                "available_at":"2020-01-02T00:10:00Z",
                "data_classification":"TRUE_HISTORICAL_SOURCE",
                "cmc_close_eur_ret_1d":0.10,
            },
            {
                "market":"BTC-EUR",
                "available_at":"2020-01-03T00:10:00Z",
                "data_classification":"TRUE_HISTORICAL_SOURCE",
                "cmc_close_eur_ret_1d":0.20,
            },
            {
                "market":"BTC-EUR",
                "available_at":"2020-01-01T00:00:00Z",
                "data_classification":"PROSPECTIVE_COLLECTION",
                "cmc_close_eur_ret_1d":999.0,
            },
        ]
    ).to_parquet(path, index=False)
    idx = pd.to_datetime([
        "2020-01-01T23:00:00Z",
        "2020-01-02T12:00:00Z",
        "2020-01-03T12:00:00Z",
    ])
    base = pd.DataFrame({"technical": [1.0,2.0,3.0]}, index=idx)
    out = augment_cmc_pit_features(_Bridge(tmp_path), base, market="BTC-EUR")
    assert np.isnan(out.iloc[0]["cmc_close_eur_ret_1d"])
    assert out.iloc[1]["cmc_close_eur_ret_1d"] == 0.10
    assert out.iloc[2]["cmc_close_eur_ret_1d"] == 0.20
    assert not (out.get("cmc_close_eur_ret_1d") == 999.0).any()
    assert out.attrs["cmc_pit_point_in_time"] is True
