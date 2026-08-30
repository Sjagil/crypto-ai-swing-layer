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
from .orchestration.pipeline import SwingPipeline

app = typer.Typer(no_args_is_help=True)
console = Console()


def _settings() -> Settings:
    return Settings.load(Path.cwd())


@app.command()
def doctor() -> None:
    """Discover the existing crypto repository and known integration surfaces."""
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


@app.command()
def smoke(rows: int = 800) -> None:
    """Run an offline synthetic end-to-end shadow smoke test."""
    s = _settings()
    rng = np.random.default_rng(17)
    idx = pd.date_range("2025-01-01", periods=rows, freq="1h", tz="UTC")
    ret = rng.normal(0.0003, 0.008, size=rows)
    close = 50000 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    span = np.maximum(close * rng.uniform(0.001, 0.01, size=rows), 1.0)
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    volume = rng.lognormal(mean=7.0, sigma=0.8, size=rows)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
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
                "last_signal": result.signals[-1].side.value if result.signals else None,
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
        market = path.stem.replace("_1h", "").replace("_", "-").upper()
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
                "market": s.market,
                "side": s.side.value,
                "score": s.score,
                "expected_edge_bps": s.expected_edge_bps,
            }
            for s in result.signals
        ],
        "intents": [x.to_dict() for x in result.intents],
        "blocked": result.blocked,
        "execution_authority": "NONE",
    }
    out = s.project_root / "output/crypto_ai_swing/latest_shadow.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"Wrote {out}")


if __name__ == "__main__":
    app()
