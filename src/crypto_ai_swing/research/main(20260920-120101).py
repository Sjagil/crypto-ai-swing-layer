#!/usr/bin/env python3
"""
Unified launcher for:
  - Sjagil/crypto
  - Sjagil/crypto-ai-swing-layer

Recommended location:
    crypto-ai-swing-layer/main.py

Examples:
    python main.py
    python main.py --mode shadow
    python main.py --mode paper
    python main.py --once
    python main.py --doctor
    python main.py --no-dashboard

Live mode does NOT bypass any existing authority, reconciliation, canary,
risk, kill-switch, credential, or execution gates in either repository.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _candidate_roots(start: Path) -> list[Path]:
    start = start.expanduser().resolve()
    candidates: list[Path] = []

    for base in (start, start.parent, start.parent.parent):
        candidates.extend(
            [
                base,
                base / "crypto-ai-swing-layer",
                base / "crypto",
            ]
        )

    home = Path.home()
    for base in (
        home / "Documents",
        home / "Downloads",
        home / "Projects",
        home / "Code",
        home,
    ):
        candidates.extend(
            [
                base / "crypto-ai-swing-layer",
                base / "crypto",
            ]
        )

    unique: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        key = str(item)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _is_swing_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "src" / "crypto_ai_swing").is_dir()
        and (path / "config").is_dir()
    )


def _is_crypto_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "core").is_dir()
        and (path / "execution").is_dir()
        and (path / "risk").is_dir()
        and (path / "config" / "settings.py").is_file()
    )


def _resolve_repo(
    explicit: str | None,
    env_name: str,
    predicate,
    cwd: Path,
    label: str,
) -> Path:
    values: list[Path] = []

    if explicit:
        values.append(Path(explicit))
    if os.getenv(env_name):
        values.append(Path(os.environ[env_name]))
    values.extend(_candidate_roots(cwd))

    for candidate in values:
        candidate = candidate.expanduser().resolve()
        if predicate(candidate):
            return candidate

    searched = "\n".join(f"  - {p}" for p in values[:25])
    raise FileNotFoundError(
        f"Could not locate {label} repository.\n"
        f"Set {env_name} or pass the corresponding CLI path.\n"
        f"Searched:\n{searched}"
    )


def _load_env_file(path: Path, *, override: bool = False) -> None:
    if not path.is_file():
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=override)
        return
    except ImportError:
        pass

    # Minimal fallback so bootstrapping does not depend on python-dotenv.
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value


@dataclass(frozen=True)
class RepoPaths:
    swing: Path
    crypto: Path


def bootstrap_paths(args: argparse.Namespace) -> RepoPaths:
    cwd = Path.cwd()

    swing = _resolve_repo(
        args.swing_repo,
        "SWING_REPO_PATH",
        _is_swing_root,
        cwd,
        "crypto-ai-swing-layer",
    )

    # Load swing env first. Existing shell variables remain authoritative.
    _load_env_file(swing / ".env", override=False)

    crypto = _resolve_repo(
        args.crypto_repo,
        "CRYPTO_REPO_PATH",
        _is_crypto_root,
        cwd,
        "crypto",
    )

    # crypto contains the canonical API/provider credentials.
    _load_env_file(crypto / ".env", override=False)

    os.environ["SWING_REPO_PATH"] = str(swing)
    os.environ["CRYPTO_REPO_PATH"] = str(crypto)

    swing_src = str((swing / "src").resolve())
    crypto_root = str(crypto.resolve())

    # Swing package first, then canonical crypto top-level packages.
    if swing_src not in sys.path:
        sys.path.insert(0, swing_src)
    if crypto_root not in sys.path:
        sys.path.insert(1, crypto_root)

    return RepoPaths(swing=swing, crypto=crypto)


# ---------------------------------------------------------------------------
# Runtime state shared between the worker and terminal dashboard
# ---------------------------------------------------------------------------


class RuntimeState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.payload: dict[str, Any] = {
            "status": "STARTING",
            "started_at": _now(),
            "last_cycle_started_at": None,
            "last_cycle_finished_at": None,
            "last_cycle_seconds": None,
            "last_error": None,
            "cycles": 0,
        }

    def update(self, **values: Any) -> None:
        with self._lock:
            self.payload.update(values)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.payload)


# ---------------------------------------------------------------------------
# Terminal formatting
# ---------------------------------------------------------------------------


def _side_label(side: Any) -> str:
    value = str(side or "").upper()
    if value in {"BUY", "LONG"}:
        return "🟢 BUY"
    if value in {"SELL", "EXIT", "SELL/EXIT", "CLOSE"}:
        return "🔴 SELL"
    if value in {"HOLD", "WAIT", "ABSTAIN", "NONE"}:
        return "⚪ WAIT"
    return value or "-"


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_price(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "-"
    if abs(number) >= 1000:
        return f"{number:,.2f}"
    if abs(number) >= 1:
        return f"{number:.5f}"
    return f"{number:.8f}"


def _find_dict(obj: Any, *keys: str, depth: int = 0) -> dict[str, Any]:
    if depth > 8:
        return {}
    if isinstance(obj, dict):
        for key in keys:
            value = obj.get(key)
            if isinstance(value, dict):
                return value
        for value in obj.values():
            result = _find_dict(value, *keys, depth=depth + 1)
            if result:
                return result
    elif isinstance(obj, (list, tuple)):
        for value in obj[:50]:
            result = _find_dict(value, *keys, depth=depth + 1)
            if result:
                return result
    return {}


def _latest_proactive(project_root: Path) -> dict[str, Any]:
    candidates = [
        project_root / "output/crypto_ai_swing/proactive/latest.json",
        project_root / "output/crypto_ai_swing/modes/shadow/proactive/latest.json",
        project_root / "output/crypto_ai_swing/modes/paper/proactive/latest.json",
        project_root / "output/crypto_ai_swing/modes/canary/proactive/latest.json",
        project_root / "output/crypto_ai_swing/modes/live/proactive/latest.json",
    ]
    existing = [p for p in candidates if p.is_file()]
    if not existing:
        return {}

    path = max(existing, key=lambda p: p.stat().st_mtime)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}
    payload["_source"] = str(path)
    return payload


def _inject_pretty_signal_labels(proactive: dict[str, Any]) -> dict[str, Any]:
    # Work on a detached JSON-compatible copy. This never mutates repo artifacts.
    try:
        payload = json.loads(json.dumps(proactive, default=str))
    except Exception:
        payload = dict(proactive)

    for row in payload.get("signals") or []:
        if isinstance(row, dict):
            row["side"] = _side_label(row.get("side"))

    for row in payload.get("exit_events") or []:
        if isinstance(row, dict):
            row["action"] = _side_label(row.get("action") or "SELL")

    for row in payload.get("executions") or []:
        if not isinstance(row, dict):
            continue
        intent = row.get("intent")
        if isinstance(intent, dict):
            intent["side"] = _side_label(intent.get("side"))

    return payload


# ---------------------------------------------------------------------------
# Runtime worker
# ---------------------------------------------------------------------------


class RuntimeWorker:
    def __init__(
        self,
        runtime: Any,
        state: RuntimeState,
        stop_event: threading.Event,
        *,
        interval_seconds: int,
    ) -> None:
        self.runtime = runtime
        self.state = state
        self.stop_event = stop_event
        self.interval_seconds = max(10, int(interval_seconds))
        self.thread = threading.Thread(
            target=self._run,
            name="unified-autonomy-runtime",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            started = time.monotonic()
            self.state.update(
                status="RUNNING",
                last_cycle_started_at=_now(),
                last_error=None,
            )
            try:
                result = self.runtime.run_once(one_shot=False)
                elapsed = time.monotonic() - started
                current = self.state.snapshot()
                self.state.update(
                    status="HEALTHY",
                    cycles=int(current.get("cycles") or 0) + 1,
                    last_cycle_finished_at=_now(),
                    last_cycle_seconds=round(elapsed, 3),
                    last_result=result,
                )
            except Exception as exc:
                elapsed = time.monotonic() - started
                self.state.update(
                    status="ERROR",
                    last_cycle_finished_at=_now(),
                    last_cycle_seconds=round(elapsed, 3),
                    last_error=f"{type(exc).__name__}: {exc}",
                    last_traceback=traceback.format_exc(limit=8),
                )

            remaining = max(0.0, self.interval_seconds - (time.monotonic() - started))
            self.stop_event.wait(remaining)


# ---------------------------------------------------------------------------
# Rich dashboard extension
# ---------------------------------------------------------------------------


def build_runtime_table(state: RuntimeState, mode: str):
    from rich import box
    from rich.table import Table

    s = state.snapshot()
    table = Table(title="Unified runtime", box=box.SIMPLE, expand=True)
    table.add_column("Component")
    table.add_column("State")
    table.add_column("Detail")

    status = str(s.get("status") or "-")
    table.add_row(
        "crypto + swing",
        status,
        f"mode={mode.upper()} cycles={s.get('cycles', 0)} "
        f"last={s.get('last_cycle_seconds') or '-'}s",
    )
    table.add_row(
        "Last cycle",
        str(s.get("last_cycle_finished_at") or "waiting"),
        "",
    )
    if s.get("last_error"):
        table.add_row("Runtime error", "ERROR", str(s["last_error"])[:120])
    return table


def build_trade_table(proactive: dict[str, Any]):
    from rich import box
    from rich.table import Table

    table = Table(title="🟢 BUY / 🔴 SELL / SL / TP", box=box.SIMPLE, expand=True)
    table.add_column("Market", no_wrap=True)
    table.add_column("Action", no_wrap=True)
    table.add_column("Strategy")
    table.add_column("Score")
    table.add_column("Entry")
    table.add_column("SL")
    table.add_column("TP")
    table.add_column("State / reason")

    rows: list[dict[str, Any]] = []

    for item in proactive.get("executions") or []:
        if not isinstance(item, dict):
            continue
        intent = dict(item.get("intent") or {})
        execution = dict(item.get("execution") or {})
        rows.append(
            {
                "market": item.get("market") or intent.get("market"),
                "side": intent.get("side") or item.get("action"),
                "strategy": intent.get("strategy"),
                "score": intent.get("score")
                or intent.get("signal_score")
                or intent.get("expected_edge_bps"),
                "entry": intent.get("entry_price")
                or intent.get("reference_price")
                or execution.get("price"),
                "stop": intent.get("stop_price")
                or intent.get("stop_loss")
                or intent.get("stop_pct"),
                "tp": intent.get("take_profit_price")
                or intent.get("take_profit")
                or intent.get("take_profit_pct"),
                "state": item.get("action")
                or execution.get("reason_code")
                or ("ACCEPTED" if execution.get("accepted") else "BLOCKED"),
            }
        )

    if not rows:
        for item in proactive.get("signals") or []:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "market": item.get("market"),
                    "side": item.get("side"),
                    "strategy": item.get("strategy")
                    or item.get("family")
                    or item.get("signal_name"),
                    "score": item.get("score"),
                    "entry": item.get("entry_price"),
                    "stop": item.get("stop_price")
                    or item.get("stop_pct"),
                    "tp": item.get("take_profit_price")
                    or item.get("take_profit_pct"),
                    "state": item.get("reason") or "SIGNAL",
                }
            )

    for item in (proactive.get("exit_events") or [])[-5:]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "market": item.get("market"),
                "side": "SELL",
                "strategy": item.get("strategy"),
                "score": None,
                "entry": item.get("price"),
                "stop": None,
                "tp": None,
                "state": item.get("reason") or item.get("action") or "EXIT",
            }
        )

    for row in rows[-10:]:
        score = _float(row.get("score"))
        table.add_row(
            str(row.get("market") or "-"),
            _side_label(row.get("side")),
            str(row.get("strategy") or "-")[:26],
            "-" if score is None else f"{score:.3f}",
            _fmt_price(row.get("entry")),
            _fmt_price(row.get("stop")),
            _fmt_price(row.get("tp")),
            str(row.get("state") or "-")[:55],
        )

    if not rows:
        table.add_row("-", "⚪ WAIT", "-", "-", "-", "-", "-", "No current trade decision")

    return table


def build_dashboard_class(runtime_state: RuntimeState, mode: str):
    import crypto_ai_swing.monitoring.terminal_dashboard as terminal_dashboard
    from rich.layout import Layout

    Base = terminal_dashboard.RealtimeTerminalDashboard

    class IntegratedDashboard(Base):
        def render(self):
            proactive = _latest_proactive(self.settings.project_root)
            self.local_proactive = _inject_pretty_signal_labels(proactive)

            account = _find_dict(
                self.snapshot,
                "account",
                "account_snapshot",
                "wallet",
            )
            if account:
                self.account_state = account

            base = super().render()

            # Replace lower-left health section with a split containing the
            # canonical health plus unified runtime state.
            try:
                health = self._health_table()
                extra = Layout()
                extra.split_column(
                    Layout(health, ratio=2),
                    Layout(build_runtime_table(runtime_state, mode), ratio=1),
                    Layout(build_trade_table(self.local_proactive), ratio=2),
                )
                base["left"].children[1].update(extra)
            except Exception:
                # Dashboard display must never break the trading/runtime worker.
                pass

            return base

    return IntegratedDashboard


# ---------------------------------------------------------------------------
# Preflight / doctor
# ---------------------------------------------------------------------------


def print_preflight(settings: Any, repos: RepoPaths, mode: str) -> dict[str, Any]:
    from rich.console import Console
    from rich.table import Table

    from crypto_ai_swing.agents.runtime import AgentRuntime
    from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
    from crypto_ai_swing.execution.crypto_authority import CryptoAuthorityAdapter

    console = Console()
    bridge = CryptoLibraryBridge(repos.crypto)

    integration = bridge.integration_status()

    try:
        authority = CryptoAuthorityAdapter(repos.crypto).authority_status()
    except Exception as exc:
        authority = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        agents = AgentRuntime(settings, mode="shadow" if mode == "live" else mode).status()
    except Exception as exc:
        agents = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }

    table = Table(title="Unified crypto runtime preflight")
    table.add_column("Check")
    table.add_column("State")
    table.add_column("Detail")

    table.add_row("crypto repo", "✅ FOUND", str(repos.crypto))
    table.add_row("swing repo", "✅ FOUND", str(repos.swing))
    table.add_row(
        "crypto imports",
        "✅ READY" if integration.get("ready") else "❌ BLOCKED",
        f"{integration.get('imported_modules', 0)}/"
        f"{integration.get('required_modules', 0)} modules",
    )
    table.add_row(
        "execution authority",
        str(
            authority.get("status")
            or authority.get("state")
            or authority.get("control_state")
            or "-"
        ),
        "existing crypto authority, never bypassed",
    )
    table.add_row(
        "AI agents",
        str(agents.get("status") or agents.get("state") or "LOADED"),
        "shadow/advisory unless existing promotion gates say otherwise",
    )
    table.add_row("selected mode", mode.upper(), "shadow is default")

    console.print(table)

    if not integration.get("ready"):
        failed = [
            x for x in integration.get("modules", [])
            if not x.get("imported")
        ]
        failure_table = Table(title="Failed crypto imports")
        failure_table.add_column("Module")
        failure_table.add_column("Error")
        for row in failed[:20]:
            failure_table.add_row(
                str(row.get("module")),
                str(row.get("error") or "-")[:120],
            )
        console.print(failure_table)

    return {
        "integration": integration,
        "authority": authority,
        "agents": agents,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sjagil/crypto + crypto-ai-swing-layer as one integrated process."
    )
    parser.add_argument(
        "--mode",
        choices=("shadow", "paper", "live"),
        default=os.getenv("CRYPTO_MAIN_MODE", "shadow"),
        help="Execution mode. Default: shadow.",
    )
    parser.add_argument(
        "--swing-repo",
        help="Absolute path to crypto-ai-swing-layer.",
    )
    parser.add_argument(
        "--crypto-repo",
        help="Absolute path to crypto.",
    )
    parser.add_argument(
        "--cycle-seconds",
        type=int,
        default=None,
        help="Override unified autonomy cycle interval.",
    )
    parser.add_argument(
        "--refresh-hz",
        type=float,
        default=4.0,
        help="Terminal dashboard refresh rate, 1-10 Hz.",
    )
    parser.add_argument(
        "--no-public-feed",
        action="store_true",
        help="Disable the extra read-only dashboard Bitvavo WebSocket.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Run integrated autonomy without the fullscreen terminal dashboard.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one complete unified autonomy cycle and exit.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Validate both repository integrations and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repos = bootstrap_paths(args)

    # Imports deliberately happen only after both repositories are on sys.path.
    from rich.console import Console

    from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
    from crypto_ai_swing.orchestration.control_plane import ModeController
    from crypto_ai_swing.orchestration.unified_runtime import UnifiedAutonomyRuntime
    from crypto_ai_swing.settings import Settings

    console = Console()

    # Settings.load must use the swing root. It then consumes CRYPTO_REPO_PATH
    # and imports the canonical crypto repo through CryptoLibraryBridge.
    settings = Settings.load(repos.swing)

    # Hard sanity check against accidental import from another checkout.
    bridge = CryptoLibraryBridge(settings.crypto_repo_root)
    if bridge.root != repos.crypto.resolve():
        raise RuntimeError(
            "Settings resolved a different crypto repository:\n"
            f"expected: {repos.crypto.resolve()}\n"
            f"actual:   {bridge.root}"
        )

    preflight = print_preflight(settings, repos, args.mode)

    if args.doctor:
        return 0 if preflight["integration"].get("ready") else 2

    if not preflight["integration"].get("ready"):
        console.print(
            "[bold red]Cannot start unified runtime: canonical crypto imports are incomplete.[/]"
        )
        return 2

    # Live remains explicitly controlled by the existing control plane.
    if args.mode == "live":
        controller = ModeController(settings)
        state = controller.status()
        preflight_live = controller.preflight("canary")
        if state.get("selected_mode") != "canary" or not preflight_live.get("ready"):
            console.print("[bold red]LIVE blocked by the existing canary control plane.[/]")
            console.print_json(json.dumps({
                "selected_mode": state.get("selected_mode"),
                "preflight": preflight_live,
                "orders_submitted": 0,
            }, default=str))
            return 2

    runtime = UnifiedAutonomyRuntime(settings, mode=args.mode)

    if args.once:
        try:
            result = runtime.run_once(one_shot=True)
            console.print_json(json.dumps(result, default=str))
            return 0
        finally:
            runtime.close()

    interval = (
        int(args.cycle_seconds)
        if args.cycle_seconds is not None
        else max(
            10,
            int(
                dict(getattr(settings, "supervisor", {}) or {}).get(
                    "cycle_interval_seconds",
                    60,
                )
            ),
        )
    )

    runtime_state = RuntimeState()
    stop_event = threading.Event()
    worker = RuntimeWorker(
        runtime,
        runtime_state,
        stop_event,
        interval_seconds=interval,
    )

    def request_stop(*_: Any) -> None:
        stop_event.set()

    try:
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
    except ValueError:
        # signal.signal only works in the main thread.
        pass

    worker.start()

    console.print(
        f"[bold green]Unified runtime started[/] "
        f"mode={args.mode.upper()} cycle={interval}s"
    )

    try:
        if args.no_dashboard:
            while not stop_event.is_set():
                state = runtime_state.snapshot()
                status = state.get("status")
                cycles = state.get("cycles")
                error = state.get("last_error")
                console.print(
                    f"[{_now()}] runtime={status} cycles={cycles}"
                    + (f" error={error}" if error else "")
                )
                stop_event.wait(5.0)
        else:
            Dashboard = build_dashboard_class(runtime_state, args.mode)
            dashboard = Dashboard(
                settings,
                refresh_hz=max(1.0, min(10.0, float(args.refresh_hz))),
                public_feed=not args.no_public_feed,
            )
            try:
                asyncio.run(dashboard.run())
            except KeyboardInterrupt:
                pass
    finally:
        stop_event.set()
        try:
            runtime.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
