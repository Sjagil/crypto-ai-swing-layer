from __future__ import annotations

from typing import Protocol, Mapping, Any
from pathlib import Path
import pandas as pd

from crypto_ai_swing.contracts import TradeIntent


class MarketDataAdapter(Protocol):
    def bars(self, market: str, timeframe: str) -> pd.DataFrame: ...


class UniverseAdapter(Protocol):
    def native_markets(self) -> pd.DataFrame: ...
    def cmc_point_in_time(self, timestamp: pd.Timestamp) -> pd.DataFrame: ...


class ContextAdapter(Protocol):
    def latest_context(self, market: str) -> Mapping[str, Any]: ...


class IntentSink(Protocol):
    def write(self, intent: TradeIntent) -> Path: ...
