from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge


@dataclass(frozen=True)
class NativeCampaignDependencySpec:
    reports: tuple[str, ...] = ()
    datasets: tuple[tuple[str, str], ...] = ()


NATIVE_CAMPAIGN_DEPENDENCIES: dict[str, NativeCampaignDependencySpec] = {
    "multi-alpha-v2": NativeCampaignDependencySpec(
        reports=(
            "absolute_momentum_campaign_v1.json",
            "residual_reversal_campaign_v1.json",
        ),
        datasets=(
            ("BTC-EUR", "1d"),
            ("ETH-EUR", "1d"),
            ("SOL-EUR", "1d"),
            ("LINK-EUR", "1d"),
        ),
    ),
}


REPORT_REPAIR_COMMANDS: dict[str, str] = {
    "absolute_momentum_campaign_v1.json": (
        "python main.py lab campaign run --name absolute-momentum-v1 --yes"
    ),
    "residual_reversal_campaign_v1.json": (
        "python main.py lab campaign run --name residual-reversal-v1 --yes"
    ),
}


def _normalized_timeframe(timeframe: str) -> str:
    return "1W" if str(timeframe).lower() == "1w" else str(timeframe)


def inspect_native_campaign_dependencies(
    *,
    crypto_repo_root: Path,
    campaign: str,
    native_settings: Any | None = None,
) -> dict[str, Any]:
    """Inspect native campaign prerequisites without generating research evidence."""

    key = str(campaign).strip().lower()
    spec = NATIVE_CAMPAIGN_DEPENDENCIES.get(
        key,
        NativeCampaignDependencySpec(),
    )
    settings = (
        native_settings
        if native_settings is not None
        else CryptoLibraryBridge(Path(crypto_repo_root)).settings()
    )

    reports_dir = Path(settings.paths.lab_dir) / "reports"
    processed_dir = Path(settings.paths.processed_data_dir)

    report_rows: list[dict[str, Any]] = []
    for name in spec.reports:
        path = reports_dir / name
        report_rows.append(
            {
                "name": name,
                "path": str(path),
                "present": path.is_file(),
                "repair_command": REPORT_REPAIR_COMMANDS.get(name),
            }
        )

    dataset_rows: list[dict[str, Any]] = []
    for market, timeframe in spec.datasets:
        normalized_tf = _normalized_timeframe(timeframe)
        path = processed_dir / f"{market}_{normalized_tf}.parquet"
        dataset_rows.append(
            {
                "market": market,
                "timeframe": timeframe,
                "path": str(path),
                "present": path.is_file(),
            }
        )

    missing_reports = [row for row in report_rows if not row["present"]]
    missing_datasets = [row for row in dataset_rows if not row["present"]]
    repair_commands = list(
        dict.fromkeys(
            str(row["repair_command"])
            for row in missing_reports
            if row.get("repair_command")
        )
    )

    ready = not missing_reports and not missing_datasets
    return {
        "schema_version": "crypto_ai_swing_native_campaign_dependencies_v1",
        "campaign": key,
        "ready": ready,
        "classification": (
            "PREREQUISITES_READY"
            if ready
            else "BLOCKED_PREREQUISITE_EVIDENCE"
        ),
        "dependency_spec": asdict(spec),
        "reports": report_rows,
        "datasets": dataset_rows,
        "missing_report_count": len(missing_reports),
        "missing_dataset_count": len(missing_datasets),
        "repair_commands": repair_commands,
        "authority": "RESEARCH_ONLY",
        "automatic_live_promotion": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
