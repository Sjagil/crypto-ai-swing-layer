from __future__ import annotations

from pathlib import Path
from decimal import Decimal
import json
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .settings import Settings
from .bridge.discovery import write_report
from .bridge.crypto_library import CryptoLibraryBridge
from .orchestration.pipeline import SwingPipeline
from .orchestration.proactive import ProactiveTrader
from .execution.bitvavo import live_gate_status
from .nlp.engine import NLPMarketEngine

app = typer.Typer(no_args_is_help=True)
console = Console()


def _settings() -> Settings:
    return Settings.load(Path.cwd())


@app.command()
def doctor() -> None:
    """Discover existing crypto repository integration surfaces."""
    s = _settings()
    report_path = s.project_root / "output/crypto_ai_swing/doctor.json"
    report = write_report(s.crypto_repo_root, report_path)
    table = Table(title="Crypto repository discovery")
    table.add_column("Capability")
    table.add_column("Found")
    table.add_column("Path")
    for item in report["items"]:
        table.add_row(
            item["capability"],
            "YES" if item["found"] else "NO",
            ", ".join(item["paths"]) or "-",
        )
    console.print(table)
    console.print(f"Report: {report_path}")


@app.command("crypto-library-doctor")
def crypto_library_doctor() -> None:
    """Import Sjagil/crypto as a Python library and expose exact interfaces."""
    s = _settings()
    bridge = CryptoLibraryBridge(s.crypto_repo_root)
    out = (
        s.project_root
        / "output/crypto_ai_swing/crypto_library_doctor.json"
    )
    report = bridge.write_integration_report(out)
    table = Table(title="Sjagil/crypto library imports")
    table.add_column("Module")
    table.add_column("Imported")
    table.add_column("Source / error")
    for item in report["modules"]:
        table.add_row(
            item["module"],
            "YES" if item["imported"] else "NO",
            item.get("file") or item.get("error") or "-",
        )
    console.print(table)
    console.print(
        f"Imported {report['imported_modules']}/{report['required_modules']} modules"
    )
    console.print("Execution authority public interfaces:")
    console.print_json(
        json.dumps(
            report.get("execution_authority_interfaces", {}),
            default=str,
        )
    )
    console.print(f"Report: {out}")


@app.command()
def smoke(rows: int = 800) -> None:
    """Run an offline synthetic end-to-end shadow smoke test."""
    s = _settings()
    rng = np.random.default_rng(17)
    idx = pd.date_range(
        "2025-01-01", periods=rows, freq="1h", tz="UTC"
    )
    ret = rng.normal(0.0003, 0.008, size=rows)
    close = 50000 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    span = np.maximum(
        close * rng.uniform(0.001, 0.01, size=rows), 1.0
    )
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    volume = rng.lognormal(mean=7.0, sigma=0.8, size=rows)
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    result = SwingPipeline(s).run(
        {"BTC-EUR": df},
        equity_eur=Decimal("10000"),
        cash_eur=Decimal("10000"),
    )
    console.print(
        json.dumps(
            {
                "signals": len(result.signals),
                "intents": len(result.intents),
                "blocked": result.blocked,
                "last_signal": (
                    result.signals[-1].side.value
                    if result.signals
                    else None
                ),
                "live_order_calls": 0,
            },
            indent=2,
        )
    )


@app.command()
def shadow(
    data_dir: Path = typer.Option(..., exists=True, file_okay=False),
    pattern: str = "*_1h.parquet",
    equity_eur: float = 10000.0,
) -> None:
    """Run one shadow decision pass over local Parquet files."""
    s = _settings()
    frames = {}
    for path in sorted(data_dir.glob(pattern)):
        market = (
            path.stem.replace("_1h", "")
            .replace("_", "-")
            .upper()
        )
        frames[market] = pd.read_parquet(path)
    if not frames:
        raise typer.BadParameter(f"No files matched {pattern}")
    result = SwingPipeline(s).run(
        frames,
        equity_eur=Decimal(str(equity_eur)),
        cash_eur=Decimal(str(equity_eur)),
    )
    payload = {
        "markets": len(frames),
        "signals": [
            {
                "market": signal.market,
                "side": signal.side.value,
                "score": signal.score,
                "expected_edge_bps": signal.expected_edge_bps,
            }
            for signal in result.signals
        ],
        "intents": [x.to_dict() for x in result.intents],
        "blocked": result.blocked,
        "execution_authority": "NONE",
    }
    out = s.project_root / "output/crypto_ai_swing/latest_shadow.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    console.print(f"Wrote {out}")


@app.command("live-preflight")
def live_preflight() -> None:
    """Check live gates without placing an order."""
    s = _settings()
    status = live_gate_status(s.execution)
    extra = []
    try:
        bridge = CryptoLibraryBridge(s.crypto_repo_root)
        library = bridge.integration_status()
        if not library.get("ready"):
            extra.append("CRYPTO_LIBRARY_NOT_READY")
        if not bridge.execution_authority_interfaces():
            extra.append("CRYPTO_EXECUTION_AUTHORITY_INTERFACE_MISSING")
    except Exception:
        extra.append("CRYPTO_LIBRARY_IMPORT_FAILED")
    blockers = list(status.blockers) + extra
    console.print(
        json.dumps(
            {"ready": not blockers, "blockers": blockers},
            indent=2,
        )
    )


@app.command()
def nlp(text: str) -> None:
    """Analyze text with FinBERT, deterministic fallback otherwise."""
    s = _settings()
    assessment = NLPMarketEngine(s.nlp).assess_text(text)
    console.print(
        json.dumps(
            {
                "score": assessment.score,
                "confidence": assessment.confidence,
                "severe_negative": assessment.severe_negative,
                "event_tags": list(assessment.event_tags),
                "assets": list(assessment.assets),
                "model": assessment.model,
            },
            indent=2,
        )
    )


@app.command()
def proactive(
    mode: str = typer.Option(
        "shadow", help="shadow, paper, or live"
    ),
    once: bool = typer.Option(False, help="Run one cycle and exit"),
    interval_seconds: int = typer.Option(60, min=10),
) -> None:
    """Run proactive crypto-library-backed swing decisions."""
    if mode not in {"shadow", "paper", "live"}:
        raise typer.BadParameter(
            "mode must be shadow, paper, or live"
        )
    s = _settings()
    trader = ProactiveTrader(s, mode=mode)
    try:
        if once:
            console.print_json(
                json.dumps(trader.cycle(), default=str)
            )
        else:
            console.print(
                f"Starting proactive runtime in {mode.upper()} mode"
            )
            trader.run_forever(
                interval_seconds=interval_seconds
            )
    finally:
        trader.close()


if __name__ == "__main__":
    app()
