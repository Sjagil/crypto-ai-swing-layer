from __future__ import annotations

import pandas as pd
from .canonical import quality_report


def passes_quality(df: pd.DataFrame, config: dict, expected_frequency: str | None = None) -> tuple[bool, dict]:
    report = quality_report(df, expected_frequency)
    limits = config.get("market_data", {})
    blockers: list[str] = []
    if report.get("duplicate_ratio", 0.0) > limits.get("maximum_duplicate_ratio", 0.0):
        blockers.append("DUPLICATES")
    if report.get("ohlc_violation_ratio", 0.0) > limits.get("maximum_ohlc_violation_ratio", 0.0):
        blockers.append("OHLC_VIOLATION")
    if report.get("negative_volume_ratio", 0.0) > limits.get("maximum_negative_volume_ratio", 0.0):
        blockers.append("NEGATIVE_VOLUME")
    if report.get("gap_ratio", 0.0) > limits.get("maximum_gap_ratio", 1.0):
        blockers.append("GAPS")
    report["blockers"] = blockers
    return not blockers, report
