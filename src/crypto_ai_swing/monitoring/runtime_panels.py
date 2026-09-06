from __future__ import annotations

import json
from pathlib import Path

from rich import box
from rich.table import Table


def load_latest_proactive(project_root: Path) -> dict:
    candidates = [
        project_root / "output/crypto_ai_swing/proactive/latest.json",
        project_root / "output/crypto_ai_swing/modes/shadow/proactive/latest.json",
        project_root / "output/crypto_ai_swing/modes/paper/proactive/latest.json",
        project_root / "output/crypto_ai_swing/modes/canary/proactive/latest.json",
        project_root / "output/crypto_ai_swing/modes/live/proactive/latest.json",
    ]
    available = [path for path in candidates if path.is_file()]
    if not available:
        return {}
    path = max(available, key=lambda item: item.stat().st_mtime)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(payload, dict):
        payload["_dashboard_source"] = str(path)
        return payload
    return {}


def signals_table(proactive: dict) -> Table:
    table = Table(title="Signals / decisions", box=box.SIMPLE, expand=True)
    table.add_column("Market")
    table.add_column("Side")
    table.add_column("Detail")
    screen = dict(proactive.get("screen") or {})
    signals = sorted(
        list(proactive.get("signals") or []),
        key=lambda row: float(row.get("score") or 0),
        reverse=True,
    )
    for row in signals[:10]:
        market = str(row.get("market") or "-")
        state = dict(screen.get(market) or {})
        technical = dict(state.get("technical") or {})
        agent = dict(state.get("agent_preview") or {})
        rl = dict(state.get("rl_preview") or {})
        detail = (
            f"score={float(row.get('score') or 0):.3f} "
            f"{technical.get('breakout_state') or '-'} "
            f"alpha={agent.get('alpha_probability')} "
            f"RL={rl.get('long_probability')}"
        )
        table.add_row(market, str(row.get("side") or "-"), detail[:120])
    for row in list(proactive.get("exit_events") or [])[-3:]:
        table.add_row(
            str(row.get("market") or "-"),
            "SELL/EXIT",
            str(row.get("reason") or row.get("action") or "-")[:100],
        )
    if not signals and not proactive.get("exit_events"):
        table.add_row("-", "no rows", proactive.get("_dashboard_source", "-"))
    return table


def orders_table(proactive: dict, account: dict) -> Table:
    table = Table(title="Orders / fills / positions", box=box.SIMPLE, expand=True)
    table.add_column("Item")
    table.add_column("State")
    table.add_column("Detail")
    rows = list(proactive.get("executions") or [])[-4:]
    rows += list(proactive.get("exit_events") or [])[-4:]
    for row in rows[-8:]:
        intent = dict(row.get("intent") or {})
        execution = dict(row.get("execution") or {})
        market = row.get("market") or intent.get("market") or "-"
        state = (
            row.get("action")
            or ("ACCEPTED" if execution.get("accepted") else "BLOCKED")
        )
        detail = (
            row.get("reason")
            or execution.get("reason_code")
            or execution.get("order_id")
            or "-"
        )
        table.add_row(str(market), str(state), str(detail)[:100])
    positions = dict(account.get("positions") or {})
    for market, value in list(positions.items())[:4]:
        table.add_row(str(market), "POSITION", str(value)[:100])
    if not rows and not positions:
        table.add_row("-", "no rows", "-")
    return table


def cmc_table(proactive: dict) -> Table:
    table = Table(title="CMC / macro", box=box.SIMPLE, expand=True)
    table.add_column("Metric")
    table.add_column("Value")
    context = dict(proactive.get("cmc_context") or {})
    global_metrics = dict(context.get("global_metrics") or {})
    fear = context.get("fear_and_greed")
    fear_value = fear.get("value") if isinstance(fear, dict) else None
    fear_class = (
        fear.get("value_classification") or fear.get("classification")
        if isinstance(fear, dict)
        else None
    )
    alt = context.get("altcoin_season")
    alt_value = alt.get("altcoin_index") if isinstance(alt, dict) else None
    breadth = dict(context.get("breadth_top250") or {})
    rows = [
        ("F&G", f"{fear_value} {fear_class or ''}".strip()),
        ("Alt season", alt_value),
        ("BTC dominance", global_metrics.get("btc_dominance")),
        ("Total cap EUR", global_metrics.get("total_market_cap")),
        ("24h volume EUR", global_metrics.get("total_volume_24h")),
        ("Breadth 24h", breadth.get("positive_fraction_24h")),
        ("Trending #1", (context.get("trending") or [{}])[0].get("symbol") if context.get("trending") else None),
        ("Gainer #1", (context.get("gainers") or [{}])[0].get("symbol") if context.get("gainers") else None),
    ]
    for name, value in rows:
        table.add_row(name, str(value if value is not None else "-"))
    return table


def wallet_table(account: dict) -> Table:
    table = Table(title="Wallet / account", box=box.SIMPLE, expand=True)
    table.add_column("Metric")
    table.add_column("Value")
    for name, key in (
        ("Private read", "status"),
        ("Authority", "authority_active"),
        ("Reconcile", "authority_state"),
        ("EUR available", "eur_available"),
        ("Equity EUR", "estimated_equity_eur"),
        ("Crypto exposure", "wallet_crypto_exposure_eur"),
        ("Managed positions", "managed_position_count"),
        ("Entry allowed", "entry_allowed"),
    ):
        value = account.get(key)
        table.add_row(name, str(value if value is not None else "-"))
    errors = list(account.get("errors") or [])
    if errors:
        table.add_row("Account error", str(errors[0])[:100])
    return table
