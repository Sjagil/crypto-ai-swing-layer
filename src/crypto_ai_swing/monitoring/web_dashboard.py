from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import webbrowser
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from crypto_ai_swing.agents.runtime import AgentRuntime
from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.bridge.native_foundation import NativeFoundationBridge
from crypto_ai_swing.execution.crypto_authority import CryptoAuthorityAdapter
from crypto_ai_swing.monitoring.ai_training import collect_ai_training_state
from crypto_ai_swing.orchestration.control_plane import ModeController
from crypto_ai_swing.research.forward import ForwardEvidenceLedger
from crypto_ai_swing.universe.runtime import UniverseManager


SCHEMA = "crypto_ai_swing_web_dashboard_v1"
SERVICE_LABELS = {
    "round43": "com.sjagil.crypto-ai-swing.round43",
    "mdpro": "com.sjagil.crypto.round44-mdpro",
}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "path": str(path), "age_seconds": None, "payload": {}}
    try:
        age = max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        age = None
    return {
        "exists": True,
        "path": str(path),
        "age_seconds": age,
        "payload": _read_json(path),
    }


def _tail(path: Path, lines: int = 80) -> list[str]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            rows = handle.readlines()
    except OSError:
        return []
    return [row.rstrip("\n") for row in rows[-lines:]]


def _run_text(args: list[str], timeout: float = 2.5) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 1, ""
    return int(completed.returncode), (completed.stdout or completed.stderr or "").strip()


def _launchctl(label: str) -> dict[str, Any]:
    domain = f"gui/{os.getuid()}/{label}"
    code, text = _run_text(["launchctl", "print", domain])
    if code != 0:
        return {"label": label, "loaded": False, "state": "NOT_LOADED", "pid": None}
    state_match = re.search(r"^\s*state = (.+)$", text, re.MULTILINE)
    pid_match = re.search(r"^\s*pid = (\d+)$", text, re.MULTILINE)
    exit_match = re.search(r"^\s*last exit code = (.+)$", text, re.MULTILINE)
    pid = int(pid_match.group(1)) if pid_match else None
    result: dict[str, Any] = {
        "label": label,
        "loaded": True,
        "state": state_match.group(1).strip() if state_match else "LOADED",
        "pid": pid,
        "last_exit_code": exit_match.group(1).strip() if exit_match else None,
    }
    if pid:
        _, ps = _run_text(["ps", "-p", str(pid), "-o", "etime=,%cpu=,%mem=,rss=,state="])
        bits = ps.split()
        if len(bits) >= 5:
            result.update(
                {
                    "elapsed": bits[0],
                    "cpu_pct": bits[1],
                    "mem_pct": bits[2],
                    "rss_kb": bits[3],
                    "process_state": bits[4],
                }
            )
    return result


def _latest_proactive(project_root: Path) -> dict[str, Any]:
    candidates = [
        project_root / "output/crypto_ai_swing/proactive/latest.json",
        project_root / "output/crypto_ai_swing/modes/shadow/proactive/latest.json",
        project_root / "output/crypto_ai_swing/modes/paper/proactive/latest.json",
        project_root / "output/crypto_ai_swing/modes/canary/proactive/latest.json",
        project_root / "output/crypto_ai_swing/modes/live/proactive/latest.json",
    ]
    available = [path for path in candidates if path.is_file()]
    if not available:
        return {"exists": False, "path": None, "age_seconds": None, "payload": {}}
    latest = max(available, key=lambda item: item.stat().st_mtime)
    return _artifact(latest)


def _configured_readiness(settings, ledger: ForwardEvidenceLedger) -> dict[str, Any]:
    cfg = dict(
        (getattr(settings, "autonomy", {}) or {})
        .get("forward_evidence", {})
        .get("canary_readiness", {})
        or {}
    )
    return ledger.canary_readiness(
        primary_horizon_hours=int(cfg.get("primary_horizon_hours", 4)),
        minimum_unblocked_buy_outcomes=int(cfg.get("minimum_unblocked_buy_outcomes", 30)),
        minimum_distinct_markets=int(cfg.get("minimum_distinct_markets", 5)),
        minimum_observation_span_hours=float(cfg.get("minimum_observation_span_hours", 72)),
        minimum_mean_return_bps=float(cfg.get("minimum_mean_return_bps", 0.0)),
        minimum_positive_return_rate=float(cfg.get("minimum_positive_return_rate", 0.50)),
    )


def _mdpro_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    manager = dict(payload.get("manager") or payload.get("market_data") or payload)
    realtime = dict(payload.get("realtime") or payload.get("orderflow") or {})
    manager_markets = manager.get("markets") or {}
    realtime_markets = realtime.get("markets") or []
    rt_by_market = {
        str(row.get("market") or "").upper(): dict(row)
        for row in realtime_markets
        if isinstance(row, dict) and row.get("market")
    }
    rows: list[dict[str, Any]] = []
    if isinstance(manager_markets, dict):
        iterable = manager_markets.items()
    elif isinstance(manager_markets, list):
        iterable = [
            (str(row.get("market") or ""), row)
            for row in manager_markets
            if isinstance(row, dict)
        ]
    else:
        iterable = []
    for market, raw in iterable:
        item = dict(raw or {})
        market_name = str(market or item.get("market") or "").upper()
        rt = rt_by_market.get(market_name, {})
        rows.append(
            {
                "market": market_name,
                "status": item.get("status") or item.get("state") or "UNKNOWN",
                "sequence_valid": rt.get("sequence_valid", item.get("sequence_valid")),
                "fresh": rt.get("fresh"),
                "spread_bps": item.get("spread_bps") or rt.get("spread_bps"),
                "bid_levels": ((item.get("book_health") or {}).get("bid_levels") if isinstance(item.get("book_health"), dict) else None),
                "ask_levels": ((item.get("book_health") or {}).get("ask_levels") if isinstance(item.get("book_health"), dict) else None),
                "last_error": item.get("last_error") or rt.get("last_error"),
            }
        )
    rows.sort(key=lambda row: row["market"])
    return rows


def _round44_payload(proactive: dict[str, Any]) -> dict[str, Any]:
    for key in ("round44_comprehensive", "round44", "comprehensive_intelligence"):
        value = proactive.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _screen_rows(proactive: dict[str, Any]) -> list[dict[str, Any]]:
    screen = proactive.get("screen") or {}
    rows: list[dict[str, Any]] = []
    if isinstance(screen, dict):
        for market, raw in screen.items():
            if not isinstance(raw, dict):
                continue
            technical = dict(raw.get("technical") or {})
            indicators = dict(technical.get("indicators") or {})
            rows.append(
                {
                    "market": str(market),
                    "score": raw.get("execution_adjusted_score", raw.get("score")),
                    "breakout": technical.get("breakout_state"),
                    "rsi": indicators.get("rsi_14"),
                    "adx": indicators.get("adx_14"),
                    "status": raw.get("status") or raw.get("decision") or "WATCH",
                }
            )
    rows.sort(key=lambda row: float(row.get("score") or -999), reverse=True)
    return rows[:25]


def _overall_state(snapshot: dict[str, Any]) -> str:
    live = snapshot.get("live", {})
    if live.get("preflight_ready") is True and snapshot.get("mdpro", {}).get("ready_count") == 25:
        return "CANARY_READY"
    if live.get("authority_active") is True:
        return "CANARY_BLOCKED"
    return "SHADOW_SAFE"


class DashboardCollector:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.project_root = Path(settings.project_root)
        self.crypto_root = Path(settings.crypto_repo_root)
        self.bridge = CryptoLibraryBridge(self.crypto_root)
        self.controller = ModeController(settings)
        self.authority = CryptoAuthorityAdapter(self.crypto_root)
        self.lock = threading.Lock()
        self.cached: dict[str, Any] = {}
        self.cached_at = 0.0

    def collect(self, *, ttl: float = 1.0) -> dict[str, Any]:
        now = time.monotonic()
        with self.lock:
            if self.cached and now - self.cached_at < ttl:
                return self.cached
            self.cached = self._collect_uncached()
            self.cached_at = now
            return self.cached

    def _collect_uncached(self) -> dict[str, Any]:
        mode = self.controller.status()
        try:
            preflight = self.controller.preflight("canary")
        except Exception as exc:
            preflight = {"ready": False, "blockers": [f"{type(exc).__name__}:{exc}"]}
        try:
            authority = self.authority.authority_status()
        except Exception as exc:
            authority = {"active": False, "state_status": "UNAVAILABLE", "error": f"{type(exc).__name__}:{exc}"}
        try:
            portfolio = self.authority.portfolio()
        except Exception as exc:
            portfolio = {"status": "UNAVAILABLE", "positions": {}, "error": f"{type(exc).__name__}:{exc}"}

        forward_cfg = dict((getattr(self.settings, "autonomy", {}) or {}).get("forward_evidence", {}) or {})
        forward_path = self.project_root / forward_cfg.get("path", "output/crypto_ai_swing/forward/forward.sqlite")
        ledger = ForwardEvidenceLedger(
            forward_path,
            decision_bucket_minutes=int(forward_cfg.get("decision_bucket_minutes", 15)),
        )
        try:
            forward_status = ledger.status()
            forward_report = ledger.report()
            readiness = _configured_readiness(self.settings, ledger)
        except Exception as exc:
            forward_status = {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}:{exc}"}
            forward_report = {}
            readiness = {"eligible": False, "blockers": ["FORWARD_EVIDENCE_UNAVAILABLE"]}
        finally:
            ledger.close()

        proactive_artifact = _latest_proactive(self.project_root)
        proactive = proactive_artifact["payload"]
        round44 = _round44_payload(proactive)

        native_settings = self.bridge.settings()
        mdpro_artifact = _artifact(Path(native_settings.paths.context_data_dir) / "mdpro/latest.json")
        mdpro_payload = mdpro_artifact["payload"]
        mdpro_rows = _mdpro_rows(mdpro_payload)
        mdpro_ready = sum(
            str(row.get("status") or "").upper() == "READY" and row.get("sequence_valid") is True
            for row in mdpro_rows
        )
        mdpro_nonready = [
            row
            for row in mdpro_rows
            if not (
                str(row.get("status") or "").upper() == "READY"
                and row.get("sequence_valid") is True
            )
        ]
        manager = dict(mdpro_payload.get("manager") or mdpro_payload.get("market_data") or mdpro_payload)
        health = dict(manager.get("health") or {})

        account_health = _artifact(self.crypto_root / "output/operations/live_account_health.json")
        cash_continuity = _artifact(self.crypto_root / "output/operations/eur_cash_continuity.json")

        try:
            agent = AgentRuntime(self.settings).status()
        except Exception as exc:
            agent = {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}:{exc}"}
        try:
            foundation_obj = NativeFoundationBridge(self.crypto_root).status()
            foundation = _jsonable(foundation_obj)
        except Exception as exc:
            foundation = {"ready": False, "error": f"{type(exc).__name__}:{exc}"}
        try:
            universe = UniverseManager(self.settings).current(force_refresh=False)
        except Exception as exc:
            universe = {"markets": [], "error": f"{type(exc).__name__}:{exc}"}
        try:
            ai_training = collect_ai_training_state(self.project_root, self.crypto_root)
        except Exception as exc:
            ai_training = {"agents": [], "training": [], "error": f"{type(exc).__name__}:{exc}"}

        services = {name: _launchctl(label) for name, label in SERVICE_LABELS.items()}
        round43_heartbeat = _artifact(self.project_root / "output/crypto_ai_swing/autonomy/round43/heartbeat.json")
        supervisor_heartbeat = _artifact(self.project_root / "output/crypto_ai_swing/supervisor/heartbeat.json")

        logs = {
            "round43_stdout": _tail(self.project_root / "output/crypto_ai_swing/autonomy/round43/logs/stdout.log", 60),
            "round43_stderr": _tail(self.project_root / "output/crypto_ai_swing/autonomy/round43/logs/stderr.log", 60),
            "mdpro_stdout": _tail(self.project_root / "output/crypto_ai_swing/round44/logs/stdout.log", 60),
            "mdpro_stderr": _tail(self.project_root / "output/crypto_ai_swing/round44/logs/stderr.log", 60),
        }

        account = dict(account_health.get("payload") or {})
        acct = dict(account.get("account") or {})
        valuation = dict(acct.get("portfolio_valuation") or {})
        live = {
            "preflight_ready": bool(preflight.get("ready")),
            "preflight_blockers": list(preflight.get("blockers") or []),
            "canary_scope": preflight.get("canary_scope"),
            "mode": mode,
            "authority_active": authority.get("active") is True,
            "authority_state": authority.get("state_status"),
            "execution_environment_ready": authority.get("execution_environment_ready") is True,
            "maximum_order_eur": authority.get("maximum_order_eur"),
            "maximum_total_exposure_eur": authority.get("maximum_total_exposure_eur"),
            "maximum_open_positions": authority.get("maximum_open_positions"),
            "maximum_new_orders_per_day": authority.get("maximum_new_orders_per_day"),
            "maximum_risk_per_trade_eur": authority.get("maximum_risk_per_trade_eur"),
            "spot_only": authority.get("spot_only"),
            "margin": authority.get("margin"),
            "leverage": authority.get("leverage"),
            "shorting": authority.get("shorting"),
            "withdrawals": authority.get("withdrawals"),
            "autoscale": authority.get("autoscale"),
        }

        snapshot: dict[str, Any] = {
            "schema_version": SCHEMA,
            "generated_at": datetime.now(UTC).isoformat(),
            "live": live,
            "account": {
                "status": account.get("status"),
                "failures": list(account.get("failures") or []),
                "entry_allowed": account.get("entry_allowed"),
                "entry_blockers": list(account.get("entry_blockers") or []),
                "eur_available": acct.get("eur_available"),
                "equity_eur": valuation.get("estimated_total_equity_eur"),
                "valuation_status": valuation.get("status"),
                "reconciliation": account.get("reconciliation"),
                "configuration": account.get("configuration"),
                "clock_sync": account.get("clock_sync"),
                "transaction_reconciliation": account.get("transaction_reconciliation"),
                "preexisting_inventory": account.get("preexisting_inventory"),
                "cash_continuity": cash_continuity.get("payload"),
                "artifact_age_seconds": account_health.get("age_seconds"),
            },
            "portfolio": portfolio,
            "forward": {
                "status": forward_status,
                "report": forward_report,
                "readiness": readiness,
            },
            "round44": round44,
            "proactive": {
                "path": proactive_artifact.get("path"),
                "age_seconds": proactive_artifact.get("age_seconds"),
                "mode": proactive.get("mode"),
                "signals": list(proactive.get("signals") or []),
                "executions": list(proactive.get("executions") or []),
                "screen": _screen_rows(proactive),
                "prospective_canary_readiness": proactive.get("prospective_canary_readiness"),
                "runtime_state": proactive.get("runtime_state") or proactive.get("state"),
            },
            "mdpro": {
                "artifact_path": mdpro_artifact.get("path"),
                "age_seconds": mdpro_artifact.get("age_seconds"),
                "ready_count": mdpro_ready,
                "total_count": len(mdpro_rows),
                "non_ready": mdpro_nonready,
                "markets": mdpro_rows,
                "health": health,
                "l3_status": mdpro_payload.get("l3_status") or "L3_UNSUPPORTED_BY_EXECUTION_VENUE",
            },
            "services": services,
            "heartbeats": {
                "round43": round43_heartbeat,
                "supervisor": supervisor_heartbeat,
            },
            "universe": universe,
            "agent": agent,
            "foundation": foundation,
            "ai_training": ai_training,
            "logs": logs,
            "safety": {
                "automatic_live_authority": False,
                "automatic_live_promotion": False,
                "orders_generated_by_dashboard": 0,
                "orders_submitted_by_dashboard": 0,
                "dashboard_read_only": True,
            },
        }
        snapshot["overall_state"] = _overall_state(snapshot)
        return _jsonable(snapshot)


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crypto AI Swing Operations</title>
<style>
:root{--bg:#070b12;--panel:#0e1521;--panel2:#111b2b;--line:#20304a;--text:#e8edf5;--muted:#8fa2bc;--good:#38d996;--warn:#f6c85f;--bad:#ff6b7a;--blue:#66a6ff;--purple:#ad8cff}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(180deg,#070b12,#09101b 45%,#070b12);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:10;padding:18px 24px;background:rgba(7,11,18,.92);backdrop-filter:blur(16px);border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:center;justify-content:space-between}.brand{font-weight:800;letter-spacing:.08em}.sub{color:var(--muted);font-size:12px}.wrap{max-width:1700px;margin:auto;padding:18px}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:12px}.card{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:14px;padding:14px;min-width:0}.kpi{grid-column:span 2}.half{grid-column:span 6}.third{grid-column:span 4}.full{grid-column:1/-1}.title{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}.value{font-size:26px;font-weight:800}.pill{display:inline-flex;align-items:center;padding:4px 9px;border-radius:999px;border:1px solid var(--line);font-size:12px;font-weight:700}.good{color:var(--good);border-color:#1f6d54;background:#0b2d24}.warn{color:var(--warn);border-color:#67572a;background:#2b2411}.bad{color:var(--bad);border-color:#6b3038;background:#2a1115}.neutral{color:var(--blue);border-color:#2a4e7b;background:#0e2037}.muted{color:var(--muted)}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px 7px;border-bottom:1px solid #1a2940;vertical-align:top}th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.06em}td{font-variant-numeric:tabular-nums}.scroll{max-height:420px;overflow:auto}.bar{height:8px;background:#19263a;border-radius:99px;overflow:hidden;margin-top:6px}.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--purple));border-radius:99px}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;white-space:pre-wrap;word-break:break-word}.log{background:#060a10;border:1px solid #18253a;border-radius:10px;padding:10px;max-height:330px;overflow:auto}.row{display:flex;justify-content:space-between;gap:12px;margin:6px 0}.blockers{display:flex;flex-wrap:wrap;gap:6px}.small{font-size:12px}.tabs{display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap}.tab{cursor:pointer;background:#101a29;border:1px solid var(--line);color:var(--text);padding:6px 10px;border-radius:8px}.tab.active{border-color:#477bc0;color:#9cc7ff}.hidden{display:none}details{margin-top:8px}summary{cursor:pointer;color:var(--muted)}
@media(max-width:1100px){.kpi{grid-column:span 4}.half,.third{grid-column:span 12}}@media(max-width:700px){header{align-items:flex-start;flex-direction:column}.wrap{padding:10px}.kpi{grid-column:span 6}.value{font-size:20px}}
</style>
</head>
<body><header><div><div class="brand">CRYPTO AI SWING · OPERATIONS</div><div class="sub">Read-only production control plane · no order actions exposed</div></div><div><span id="overall" class="pill neutral">LOADING</span> <span id="updated" class="sub"></span></div></header>
<div class="wrap"><div id="root" class="grid"></div></div>
<script>
const esc=x=>String(x??'-').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
const n=(x,d=2)=>{const v=Number(x);return Number.isFinite(v)?v.toLocaleString(undefined,{maximumFractionDigits:d}):'-'};
const pct=(x,d=1)=>{const v=Number(x);return Number.isFinite(v)?(v*100).toFixed(d)+'%':'-'};
const cls=s=>{s=String(s??'').toUpperCase();if(['READY','HEALTHY','RUNNING','CONNECTED','CANARY_READY','PASS','PASSED','SELECTED'].some(x=>s.includes(x)))return'good';if(['BLOCK','ERROR','FAILED','DEGRADED','REQUIRED','NOT_LOADED','UNAVAILABLE'].some(x=>s.includes(x)))return'bad';if(['COLLECT','BUSY','RESEARCH','SHADOW','START','WAIT'].some(x=>s.includes(x)))return'warn';return'neutral'};
const pill=s=>`<span class="pill ${cls(s)}">${esc(s)}</span>`;
const card=(title,body,c='third')=>`<section class="card ${c}"><div class="title">${esc(title)}</div>${body}</section>`;
const rows=(obj,keys)=>keys.map(([k,l])=>`<div class="row"><span class="muted">${esc(l||k)}</span><b>${esc(obj?.[k])}</b></div>`).join('');
const blockers=a=>(a||[]).length?`<div class="blockers">${a.map(x=>`<span class="pill bad">${esc(x)}</span>`).join('')}</div>`:'<span class="pill good">NO BLOCKERS</span>';
const progress=(v,max)=>{let p=Math.max(0,Math.min(100,Number(v||0)/Number(max||1)*100));return `<div class="bar"><i style="width:${p}%"></i></div><div class="sub">${n(v,1)} / ${n(max,1)}</div>`};
function render(d){document.querySelector('#overall').className='pill '+cls(d.overall_state);document.querySelector('#overall').textContent=d.overall_state;document.querySelector('#updated').textContent='updated '+new Date(d.generated_at).toLocaleTimeString();let h='';
const live=d.live||{}, md=d.mdpro||{}, acc=d.account||{}, rd=d.forward?.readiness||{}, buy=rd.unblocked_buy||{}, mode=live.mode||{};
h+=card('Mode',`<div class="value">${esc(mode.selected_mode||'-')}</div>${pill(mode.runtime_mode||'-')}<div class="sub">automatic live promotion: ${esc(d.safety?.automatic_live_promotion)}</div>`,'kpi');
h+=card('Live gate',`<div class="value">${live.preflight_ready?'READY':'BLOCKED'}</div>${pill(live.authority_state||'-')}<div class="sub">scope ${esc(live.canary_scope||'-')}</div>`,'kpi');
h+=card('MDPro',`<div class="value">${esc(md.ready_count)}/${esc(md.total_count||25)}</div>${pill((md.ready_count===25)?'READY':'PARTIAL')}<div class="sub">age ${n(md.age_seconds,1)}s</div>`,'kpi');
h+=card('Round44',`<div class="value">${esc(d.round44?.stage||d.round44?.readiness?.stage||'-')}</div><div class="sub">next ${esc(d.round44?.next_stage||d.round44?.readiness?.next_stage||'-')}</div>`,'kpi');
h+=card('Equity',`<div class="value">€${n(acc.equity_eur,2)}</div><div class="sub">cash €${n(acc.eur_available,2)} · ${esc(acc.valuation_status||'-')}</div>`,'kpi');
h+=card('Forward evidence',`<div class="value">${n(buy.mean_return_bps,1)} bps</div><div class="sub">${esc(buy.observations||0)} obs · hit ${pct(buy.positive_return_rate)}</div>`,'kpi');

h+=card('Live authority & safety',`${rows(live,[['authority_active','authority active'],['execution_environment_ready','execution env'],['maximum_order_eur','max order EUR'],['maximum_total_exposure_eur','max exposure EUR'],['maximum_open_positions','max positions'],['maximum_new_orders_per_day','orders/day'],['maximum_risk_per_trade_eur','risk/trade EUR'],['spot_only','spot only'],['leverage','leverage'],['shorting','shorting'],['withdrawals','withdrawals'],['autoscale','autoscale']])}<div class="title" style="margin-top:12px">Current blockers</div>${blockers(live.preflight_blockers)}`,'half');
h+=card('Account / reconciliation',`${pill(acc.status||'UNKNOWN')}${rows(acc,[['entry_allowed','entry allowed'],['equity_eur','equity EUR'],['eur_available','EUR available'],['valuation_status','valuation'],['artifact_age_seconds','health age sec']])}<div class="title" style="margin-top:12px">Failures</div>${blockers(acc.failures)}<div class="title" style="margin-top:12px">Entry blockers</div>${blockers(acc.entry_blockers)}<details><summary>Configuration / reconciliation JSON</summary><pre class="mono">${esc(JSON.stringify({configuration:acc.configuration,reconciliation:acc.reconciliation,clock_sync:acc.clock_sync,transaction_reconciliation:acc.transaction_reconciliation,preexisting_inventory:acc.preexisting_inventory},null,2))}</pre></details>`,'half');

const req=rd.requirements||{};h+=card('Prospective readiness',`${pill(rd.status||'-')} ${rd.eligible?pill('ELIGIBLE'):pill('NOT ELIGIBLE')}<div class="row"><span>Observations</span><b>${n(buy.observations,0)}</b></div>${progress(buy.observations,req.minimum_unblocked_buy_outcomes||30)}<div class="row"><span>Market breadth</span><b>${n(buy.markets,0)}</b></div>${progress(buy.markets,req.minimum_distinct_markets||5)}<div class="row"><span>Observation span h</span><b>${n(rd.observation_span_hours,1)}</b></div>${progress(rd.observation_span_hours,req.minimum_observation_span_hours||72)}<div class="row"><span>Mean return bps</span><b>${n(buy.mean_return_bps,1)}</b></div><div class="row"><span>Median bps</span><b>${n(buy.median_return_bps,1)}</b></div><div class="row"><span>Positive rate</span><b>${pct(buy.positive_return_rate)}</b></div><div class="title" style="margin-top:10px">Blockers</div>${blockers(rd.blockers)}`,'third');
const mh=md.health||{};h+=card('MDPro transport',`${rows(mh,[['state','state'],['authenticated','authenticated'],['ready','ready'],['messages','messages'],['book_updates','book updates'],['queue','queue'],['queue_size','queue capacity'],['connections','connections'],['reconnects','reconnects'],['sequence_gaps','sequence gaps'],['dropped_messages','drops'],['resyncs','resyncs'],['last_error','last error']])}`,'third');
let nr=(md.non_ready||[]).map(r=>`<tr><td>${esc(r.market)}</td><td>${pill(r.status)}</td><td>${esc(r.sequence_valid)}</td><td>${esc(r.fresh)}</td><td>${n(r.spread_bps,2)}</td><td>${esc(r.last_error||'')}</td></tr>`).join('');h+=card('MDPro non-ready markets',`<div class="scroll"><table><thead><tr><th>Market</th><th>Status</th><th>Seq</th><th>Fresh</th><th>Spread</th><th>Error</th></tr></thead><tbody>${nr||'<tr><td colspan="6">All markets ready</td></tr>'}</tbody></table></div>`,'third');

let svc=Object.entries(d.services||{}).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${pill(v.state)}</td><td>${esc(v.pid||'-')}</td><td>${esc(v.elapsed||'-')}</td><td>${esc(v.cpu_pct||'-')}</td><td>${esc(v.rss_kb||'-')}</td><td>${esc(v.last_exit_code||'-')}</td></tr>`).join('');h+=card('Services / processes',`<table><thead><tr><th>Service</th><th>State</th><th>PID</th><th>Elapsed</th><th>CPU%</th><th>RSS KB</th><th>Exit</th></tr></thead><tbody>${svc}</tbody></table>`,'full');

let sig=(d.proactive?.signals||[]).slice(0,25).map(r=>`<tr><td>${esc(r.market)}</td><td>${pill(r.side||r.status||'-')}</td><td>${n(r.score,3)}</td><td>${n(r.expected_edge_bps,1)}</td><td>${esc(r.reason||'')}</td></tr>`).join('');let scr=(d.proactive?.screen||[]).map(r=>`<tr><td>${esc(r.market)}</td><td>${n(r.score,3)}</td><td>${esc(r.status)}</td><td>${esc(r.breakout||'-')}</td><td>${n(r.rsi,1)}</td><td>${n(r.adx,1)}</td></tr>`).join('');h+=card('Signals / opportunity screen',`<div class="tabs"><button class="tab active" onclick="tab(this,'signals')">Signals</button><button class="tab" onclick="tab(this,'screen')">Top screen</button></div><div id="signals"><table><thead><tr><th>Market</th><th>Side</th><th>Score</th><th>Edge bps</th><th>Reason</th></tr></thead><tbody>${sig||'<tr><td colspan="5">No current signals</td></tr>'}</tbody></table></div><div id="screen" class="hidden"><table><thead><tr><th>Market</th><th>Score</th><th>Status</th><th>Structure</th><th>RSI</th><th>ADX</th></tr></thead><tbody>${scr||'<tr><td colspan="6">No screen artifact</td></tr>'}</tbody></table></div>`,'half');
let ex=(d.proactive?.executions||[]).slice(0,25).map(r=>{let i=r.intent||{};return `<tr><td>${esc(i.market||r.market||'-')}</td><td>${pill(i.side||r.status||'-')}</td><td>€${n(i.notional_eur,2)}</td><td>${n(i.net_edge_bps,1)}</td><td>${esc(r.status||r.reason||'-')}</td></tr>`}).join('');h+=card('Execution intents / decisions',`<table><thead><tr><th>Market</th><th>Side</th><th>Notional</th><th>Net edge</th><th>State</th></tr></thead><tbody>${ex||'<tr><td colspan="5">No intents</td></tr>'}</tbody></table><div class="sub">Dashboard itself exposes no order endpoint.</div>`,'half');

let agents=(d.ai_training?.agents||[]).map(r=>`<tr><td>${esc(r.component)}</td><td>${pill(r.state)}</td><td>${esc(r.authority||'-')}</td><td>${esc(r.detail||'-')}</td></tr>`).join('');let train=(d.ai_training?.training||[]).map(r=>`<tr><td>${esc(r.component)}</td><td>${pill(r.state)}</td><td>${esc(r.detail||'-')}</td><td>${esc(r.updated||'-')}</td></tr>`).join('');h+=card('AI / agents',`<div class="scroll"><table><thead><tr><th>Component</th><th>State</th><th>Authority</th><th>Detail</th></tr></thead><tbody>${agents||'<tr><td colspan="4">No agent artifacts</td></tr>'}</tbody></table></div>`,'half');h+=card('Training / evidence artifacts',`<div class="scroll"><table><thead><tr><th>Component</th><th>State</th><th>Detail</th><th>Updated</th></tr></thead><tbody>${train||'<tr><td colspan="4">No training artifacts</td></tr>'}</tbody></table></div>`,'half');

let mk=(md.markets||[]).map(r=>`<tr><td>${esc(r.market)}</td><td>${pill(r.status)}</td><td>${esc(r.sequence_valid)}</td><td>${n(r.spread_bps,2)}</td><td>${esc(r.bid_levels||'-')}</td><td>${esc(r.ask_levels||'-')}</td></tr>`).join('');h+=card('25-market microstructure',`<div class="scroll"><table><thead><tr><th>Market</th><th>Status</th><th>Sequence</th><th>Spread bps</th><th>Bid levels</th><th>Ask levels</th></tr></thead><tbody>${mk}</tbody></table></div>`,'full');

const logNames=['mdpro_stderr','mdpro_stdout','round43_stderr','round43_stdout'];h+=card('Logs',`<div class="tabs">${logNames.map((x,i)=>`<button class="tab ${i===0?'active':''}" onclick="logtab(this,'${x}')">${x}</button>`).join('')}</div>${logNames.map((x,i)=>`<pre id="log-${x}" class="mono log ${i===0?'':'hidden'}">${esc((d.logs?.[x]||[]).join('\n'))}</pre>`).join('')}`,'full');
h+=card('Raw control snapshot',`<details><summary>Open JSON</summary><pre class="mono log">${esc(JSON.stringify(d,null,2))}</pre></details>`,'full');document.querySelector('#root').innerHTML=h;}
function tab(btn,id){btn.parentElement.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));btn.classList.add('active');['signals','screen'].forEach(x=>document.getElementById(x)?.classList.toggle('hidden',x!==id))}function logtab(btn,id){btn.parentElement.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));btn.classList.add('active');['mdpro_stderr','mdpro_stdout','round43_stderr','round43_stdout'].forEach(x=>document.getElementById('log-'+x)?.classList.toggle('hidden',x!==id))}
async function tick(){try{const r=await fetch('/api/state',{cache:'no-store'});if(!r.ok)throw new Error('HTTP '+r.status);render(await r.json())}catch(e){document.querySelector('#overall').className='pill bad';document.querySelector('#overall').textContent='DASHBOARD ERROR';document.querySelector('#updated').textContent=String(e)}}tick();setInterval(tick,2000);
</script></body></html>'''


class DashboardHandler(BaseHTTPRequestHandler):
    collector: DashboardCollector

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", HTML.encode("utf-8"))
            return
        if path == "/api/state":
            try:
                payload = self.collector.collect()
                body = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
            except Exception as exc:  # noqa: BLE001
                body = json.dumps(
                    {
                        "schema_version": SCHEMA,
                        "generated_at": datetime.now(UTC).isoformat(),
                        "overall_state": "DASHBOARD_ERROR",
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                ).encode("utf-8")
                self._send(500, "application/json; charset=utf-8", body)
                return
            self._send(200, "application/json; charset=utf-8", body)
            return
        if path == "/healthz":
            self._send(200, "application/json; charset=utf-8", b'{"status":"READY","read_only":true}')
            return
        self._send(404, "text/plain; charset=utf-8", b"not found")


def run_web_dashboard(
    settings,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    collector = DashboardCollector(settings)
    handler = type("ConfiguredDashboardHandler", (DashboardHandler,), {"collector": collector})
    server = ThreadingHTTPServer((host, int(port)), handler)
    url = f"http://{host}:{int(port)}/"
    print(json.dumps({"status": "READY", "url": url, "read_only": True}, indent=2))
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
