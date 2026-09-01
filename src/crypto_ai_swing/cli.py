from __future__ import annotations

from pathlib import Path
from decimal import Decimal
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
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
from .intelligence.crypto_news import CryptoNewsCollector
from .research.native import NativeResearchBridge
from .research.forward import ForwardEvidenceLedger
from .agents.training import AgentTrainer
from .agents.runtime import AgentRuntime
from .agents.rl_training import train_ppo_challenger
from .execution.crypto_authority import CryptoAuthorityAdapter
from .orchestration.supervisor import AutonomousSupervisor
from .universe.runtime import UniverseManager
from .research.bootstrap import ColdStartResearchRunner

app = typer.Typer(no_args_is_help=True)
console = Console()


def _settings() -> Settings:
    return Settings.load(Path.cwd())


def _forward_ledger(settings: Settings) -> ForwardEvidenceLedger:
    cfg = settings.autonomy.get("forward_evidence", {})
    rel = cfg.get("path", "output/crypto_ai_swing/forward/forward.sqlite")
    return ForwardEvidenceLedger(
        settings.project_root / rel,
        decision_bucket_minutes=int(cfg.get("decision_bucket_minutes", 15)),
    )


def _configured_canary_readiness(
    settings: Settings,
    ledger: ForwardEvidenceLedger,
) -> dict:
    cfg = dict(
        settings.autonomy.get("forward_evidence", {})
        .get("canary_readiness", {})
        or {}
    )
    return ledger.canary_readiness(
        primary_horizon_hours=int(cfg.get("primary_horizon_hours", 4)),
        minimum_unblocked_buy_outcomes=int(
            cfg.get("minimum_unblocked_buy_outcomes", 30)
        ),
        minimum_distinct_markets=int(cfg.get("minimum_distinct_markets", 5)),
        minimum_observation_span_hours=float(
            cfg.get("minimum_observation_span_hours", 72)
        ),
        minimum_mean_return_bps=float(cfg.get("minimum_mean_return_bps", 0.0)),
        minimum_positive_return_rate=float(
            cfg.get("minimum_positive_return_rate", 0.50)
        ),
    )


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


@app.command("market-data-audit")
def market_data_audit(
    market: str = typer.Argument("BTC-EUR"),
    timeframe: str = typer.Option("1h"),
) -> None:
    """Inspect normalized public market data from Sjagil/crypto."""
    s = _settings()
    bridge = CryptoLibraryBridge(s.crypto_repo_root)
    bundle = bridge.market_bundle(
        market.upper(),
        timeframe,
        mode="shadow",
        persist=False,
        depth=100,
    )
    audit = bridge.market_bundle_audit(bundle)
    out = s.project_root / "output/crypto_ai_swing/market_data_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    console.print_json(json.dumps(audit, default=str))


@app.command("news-scan")
def news_scan(force: bool = typer.Option(True)) -> None:
    """Collect causal RSS intelligence through Sjagil/crypto and cache it."""
    s = _settings()
    bridge = CryptoLibraryBridge(s.crypto_repo_root)
    cfg = s.autonomy.get("news", {})
    collector = CryptoNewsCollector(
        bridge,
        output_path=s.project_root / "output/crypto_ai_swing/intelligence/news.jsonl",
        cache_seconds=float(cfg.get("cache_seconds", 300)),
        maximum_documents=int(cfg.get("maximum_documents", 500)),
    )
    snapshot = collector.collect(force=force, persist=bool(cfg.get("persist", True)))
    payload = {
        "status": snapshot.status,
        "documents": len(snapshot.documents),
        "observed_at": snapshot.observed_at,
        "source": snapshot.source,
        "feeds": list(snapshot.source_statuses),
        "latest": [
            {
                "usable_at": doc.usable_at.isoformat(),
                "source": doc.source,
                "title": doc.title,
                "url": doc.url,
            }
            for doc in snapshot.documents[:20]
        ],
    }
    console.print_json(json.dumps(payload, default=str))


@app.command("research-doctor")
def research_doctor() -> None:
    """Inspect the canonical Sjagil/crypto research/backtest lifecycle."""
    s = _settings()
    bridge = NativeResearchBridge(s.crypto_repo_root)
    out = s.project_root / "output/crypto_ai_swing/research/native_research_doctor.json"
    payload = bridge.write_status(out)
    table = Table(title="Sjagil/crypto native research stack")
    table.add_column("Module")
    table.add_column("Imported")
    table.add_column("Public interfaces")
    for row in payload["modules"]:
        table.add_row(
            row["module"],
            "YES" if row["imported"] else "NO",
            ", ".join(sorted(row.get("interfaces", {}).keys()))[:150] or row.get("error", "-"),
        )
    console.print(table)
    console.print(f"Imported {payload['imported_modules']}/{payload['required_modules']} modules")
    console.print("Promotion lifecycle:")
    console.print(" -> ".join(payload.get("promotion_states", [])))
    console.print(f"Report: {out}")


@app.command("strategy-factory-plan")
def strategy_factory_plan(
    trials: int = typer.Option(2000, min=1, max=100000),
) -> None:
    """Generate the deterministic canonical strategy-DNA research plan."""
    s = _settings()
    bridge = NativeResearchBridge(s.crypto_repo_root)
    payload = bridge.classical_factory_plan(trial_count=trials)
    out = s.project_root / "output/crypto_ai_swing/research/classical_factory_plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    summary = {
        "status": payload.get("status"),
        "campaign": payload.get("campaign"),
        "trial_count": payload.get("trial_count"),
        "economic_family_count": payload.get("economic_family_count"),
        "research_universe": payload.get("research_universe"),
        "timeframe_routes": payload.get("timeframe_routes"),
        "evaluation_contract": payload.get("evaluation_contract"),
        "orders_generated": payload.get("orders_generated"),
        "orders_submitted": payload.get("orders_submitted"),
        "output": str(out),
    }
    console.print_json(json.dumps(summary, default=str))


@app.command("research-run")
def research_run(
    exact: bool = typer.Option(
        False,
        "--exact/--stage0-only",
        help="Run native exact validation after Stage 0 survivors. Default is Stage 0 only.",
    ),
    maximum_rows: int = typer.Option(20_000, min=1_000, max=500_000),
) -> None:
    """Run the canonical bounded Sjagil/crypto research factory."""
    s = _settings()
    bridge = NativeResearchBridge(s.crypto_repo_root)
    out = s.project_root / "output/crypto_ai_swing/research/native_campaign_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = bridge.run_factory_campaign(
            maximum_rows=maximum_rows,
            execute_exact=exact,
        )
        summary = bridge.factory_summary(payload)
        if str(payload.get("status") or "").startswith("COLD_START"):
            bootstrap = ColdStartResearchRunner(s).run()
            summary = {
                **summary,
                "status": payload.get("status"),
                "exact_requested": bool(exact),
                "cold_start_bootstrap": bootstrap,
            }
        else:
            summary["status"] = "COMPLETED"
            summary["exact_requested"] = bool(exact)
    except Exception as exc:
        summary = {
            "status": "BLOCKED",
            "exact_requested": bool(exact),
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
            "automatic_live_promotion": False,
            "orders_submitted": 0,
        }
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["output"] = str(out)
    console.print_json(json.dumps(summary, default=str))


@app.command("forward-mature")
def forward_mature(
    horizons: str = typer.Option("1,4,24"),
) -> None:
    """Causally mature past forward observations whose horizons have elapsed."""
    s = _settings()
    cfg = s.autonomy.get("forward_evidence", {})
    rel = cfg.get("path", "output/crypto_ai_swing/forward/forward.sqlite")
    ledger = ForwardEvidenceLedger(
        s.project_root / rel,
        decision_bucket_minutes=int(cfg.get("decision_bucket_minutes", 15)),
    )
    bridge = CryptoLibraryBridge(s.crypto_repo_root)
    try:
        parsed = tuple(
            sorted(
                {
                    int(value.strip())
                    for value in horizons.split(",")
                    if value.strip() and int(value.strip()) > 0
                }
            )
        )
        frames = {
            market: bridge.ohlcv(market, "1h", persist=False)
            for market in ledger.observation_markets()
        }
        payload = ledger.mature_from_frames(
            frames,
            horizons_hours=parsed or (1, 4, 24),
        )
        console.print_json(json.dumps(payload, default=str))
    finally:
        ledger.close()


@app.command("forward-report")
def forward_report() -> None:
    """Report matured prospective outcomes without granting promotion authority."""
    s = _settings()
    ledger = _forward_ledger(s)
    try:
        payload = ledger.report()
        payload["canary_readiness"] = _configured_canary_readiness(s, ledger)
        out = s.project_root / "output/crypto_ai_swing/forward/report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        payload["output"] = str(out)
        console.print_json(json.dumps(payload, default=str))
    finally:
        ledger.close()


@app.command("edge-calibration-status")
def edge_calibration_status() -> None:
    """Show prospective 4h canary evidence and edge calibration readiness."""
    s = _settings()
    ledger = _forward_ledger(s)
    try:
        payload = {
            "readiness": _configured_canary_readiness(s, ledger),
            "calibration": ledger.report().get("calibration", {}),
            "authority": "EVIDENCE_ONLY",
            "orders_submitted": 0,
        }
        console.print_json(json.dumps(payload, default=str))
    finally:
        ledger.close()


@app.command("forward-status")
def forward_status() -> None:
    """Show append-only prospective signal evidence collected by this layer."""
    s = _settings()
    cfg = s.autonomy.get("forward_evidence", {})
    rel = cfg.get("path", "output/crypto_ai_swing/forward/forward.sqlite")
    ledger = ForwardEvidenceLedger(
        s.project_root / rel,
        decision_bucket_minutes=int(cfg.get("decision_bucket_minutes", 15)),
    )
    try:
        console.print_json(json.dumps(ledger.status()))
    finally:
        ledger.close()


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



@app.command("universe-refresh")
def universe_refresh() -> None:
    """Refresh the public Bitvavo EUR spot universe and select exactly 25 markets."""
    s = _settings()
    payload = UniverseManager(s).current(force_refresh=True)
    console.print_json(json.dumps(payload, default=str))


@app.command("universe-status")
def universe_status() -> None:
    """Show the current cached 25-market runtime universe."""
    s = _settings()
    payload = UniverseManager(s).current()
    console.print_json(json.dumps(payload, default=str))


@app.command("bootstrap-research")
def bootstrap_research(
    timeframe: str = typer.Option("1h"),
    cost_bps_per_side: float | None = typer.Option(
        None, min=0.0,
        help="Optional uniform one-way cost. Empty uses current per-market spread floor.",
    ),
    final_holdout_shortlist_size: int = typer.Option(5, min=1, max=20),
) -> None:
    """Run leakage-hardened cold-start research over the runtime universe."""
    s = _settings()
    payload = ColdStartResearchRunner(s).run(
        timeframe=timeframe,
        cost_bps_per_side=cost_bps_per_side,
        final_holdout_shortlist_size=final_holdout_shortlist_size,
    )
    console.print_json(json.dumps(payload, default=str))

@app.command("economics-bootstrap")
def economics_bootstrap() -> None:
    s=_settings();bridge=NativeResearchBridge(s.crypto_repo_root)
    try: payload=bridge.bootstrap_economics()
    except Exception as exc: payload={"status":"BLOCKED","error_type":type(exc).__name__,"error":str(exc)[:1000],"orders_submitted":0}
    console.print_json(json.dumps(payload,default=str))

@app.command("agent-train")
def agent_train(
    markets: str = typer.Option("", help="Comma list. Empty = current dynamic universe"),
    timeframe: str = typer.Option("1h"),
    horizon_bars: int = typer.Option(4,min=1,max=168),
    minimum_rows: int = typer.Option(1200,min=300),
) -> None:
    s=_settings();trainer=AgentTrainer(s)
    result=trainer.train(
        markets=[x.strip().upper() for x in markets.split(",") if x.strip()] or None,
        timeframe=timeframe,horizon_bars=horizon_bars,minimum_rows=minimum_rows,
        minimum_net_move_bps=float(s.agents.get("minimum_net_move_bps",65)),
    )
    console.print_json(json.dumps({
        "status":result.status,"artifact":str(result.artifact_path),"pointer":str(result.pointer_path),
        "dataset_id":result.dataset_id,"rows":result.row_count,"markets":list(result.markets),
        "metrics":result.metrics,"live_decision_influence":False,"automatic_live_promotion":False,
    },default=str))

@app.command("agent-status")
def agent_status() -> None:
    s=_settings();console.print_json(json.dumps(AgentRuntime(s).status(),default=str))

@app.command("agent-infer")
def agent_infer(market: str = typer.Argument("BTC-EUR"), timeframe: str = typer.Option("1h")) -> None:
    s=_settings();bridge=CryptoLibraryBridge(s.crypto_repo_root)
    decision=AgentRuntime(s,mode="shadow").predict_frame(
        market.upper(),bridge.ohlcv(market.upper(),timeframe,persist=False),{"spread_bps":1.0}
    )
    console.print_json(json.dumps(asdict(decision),default=str))

@app.command("rl-train")
def rl_train(
    market: str = typer.Argument("BTC-EUR"),
    timeframe: str = typer.Option("1h"),
    timesteps: int = typer.Option(50000,min=1000),
) -> None:
    s=_settings();bridge=CryptoLibraryBridge(s.crypto_repo_root)
    payload=train_ppo_challenger(s,frame=bridge.ohlcv(market.upper(),timeframe,persist=False),total_timesteps=timesteps)
    console.print_json(json.dumps(payload,default=str))

@app.command("live-canary-status")
def live_canary_status() -> None:
    s=_settings();console.print_json(json.dumps(CryptoAuthorityAdapter(s.crypto_repo_root).authority_status(),default=str))

@app.command("live-canary-approve")
def live_canary_approve(
    markets: str = typer.Option("", help="Comma list. Empty = current dynamic universe"),
    approval: str = typer.Option(...),
) -> None:
    s = _settings()
    ledger = _forward_ledger(s)
    try:
        readiness = _configured_canary_readiness(s, ledger)
    finally:
        ledger.close()
    if not bool(readiness.get("eligible")):
        console.print_json(json.dumps({
            "status": "BLOCKED_PROSPECTIVE_EVIDENCE",
            "canary_approved": False,
            "readiness": readiness,
            "orders_submitted": 0,
            "automatic_live_promotion": False,
        }, default=str))
        return
    selected = [x.strip().upper() for x in markets.split(",") if x.strip()]
    if not selected:
        selected = list(UniverseManager(s).current()["markets"])
    payload = CryptoAuthorityAdapter(s.crypto_repo_root).approve(
        markets=selected,
        approval=approval,
    )
    payload["prospective_readiness"] = readiness
    console.print_json(json.dumps(payload, default=str))


@app.command("live-canary-deactivate")
def live_canary_deactivate() -> None:
    s=_settings();console.print_json(json.dumps(CryptoAuthorityAdapter(s.crypto_repo_root).deactivate(),default=str))

@app.command("live-canary-preflight")
def live_canary_preflight() -> None:
    s = _settings()
    ledger = _forward_ledger(s)
    try:
        readiness = _configured_canary_readiness(s, ledger)
    finally:
        ledger.close()
    if not bool(readiness.get("eligible")):
        console.print_json(json.dumps({
            "ready": False,
            "status": "BLOCKED_PROSPECTIVE_EVIDENCE",
            "blockers": list(readiness.get("blockers") or []),
            "prospective_readiness": readiness,
            "orders_submitted": 0,
        }, default=str))
        return

    latest = s.project_root / "output/crypto_ai_swing/proactive/latest.json"
    if not latest.is_file():
        raise typer.BadParameter("Run proactive --mode shadow --once first")
    report = json.loads(latest.read_text())
    candidates = [
        dict(x.get("intent") or {})
        for x in report.get("executions") or []
        if str((x.get("intent") or {}).get("side") or "").upper() == "BUY"
    ]
    if not candidates:
        console.print_json(json.dumps({
            "ready": False,
            "blockers": ["NO_CURRENT_NATURAL_BUY_INTENT"],
            "prospective_readiness": readiness,
            "orders_submitted": 0,
        }))
        return
    raw = candidates[0]
    now = datetime.now(timezone.utc)
    from crypto_ai_swing.contracts import Authority as A, Side as S, TradeIntent as T
    intent = T(
        intent_id=str(raw["intent_id"]),
        created_at=now,
        market=str(raw["market"]),
        side=S.BUY,
        notional_eur=min(Decimal("10"), Decimal(str(raw["notional_eur"]))),
        expected_edge_bps=float(raw["expected_edge_bps"]),
        estimated_round_trip_cost_bps=float(raw["estimated_round_trip_cost_bps"]),
        net_edge_bps=float(raw["net_edge_bps"]),
        stop_pct=float(raw["stop_pct"]),
        take_profit_pct=float(raw["take_profit_pct"]),
        trailing_stop_pct=float(raw["trailing_stop_pct"]),
        strategy=str(raw["strategy"]),
        authority=A.LIVE,
        expires_at=now + timedelta(seconds=120),
        metadata=dict(raw.get("metadata") or {}),
    )
    payload = CryptoAuthorityAdapter(s.crypto_repo_root).preflight(intent)
    payload["prospective_readiness"] = readiness
    console.print_json(json.dumps(payload, default=str))


@app.command("supervisor")
def supervisor(
    mode: str = typer.Option("shadow",help="shadow, paper, or live"),
    once: bool = typer.Option(False),
) -> None:
    if mode not in {"shadow","paper","live"}: raise typer.BadParameter("invalid mode")
    s=_settings();runner=AutonomousSupervisor(s,mode=mode)
    try:
        if once: console.print_json(json.dumps(runner.run_once(),default=str))
        else: runner.run_forever()
    finally: runner.close()

if __name__ == "__main__":
    app()
