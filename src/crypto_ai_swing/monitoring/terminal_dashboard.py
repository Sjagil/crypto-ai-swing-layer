from __future__ import annotations

import json
import asyncio
import time
from collections import deque
from datetime import timedelta
from typing import Any

from rich import box
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.monitoring.ai_training import collect_ai_training_state
from crypto_ai_swing.monitoring.runtime_panels import (
    cmc_table,
    orders_table,
    signals_table,
    wallet_table,
)
from crypto_ai_swing.universe.runtime import UniverseManager


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find(obj: Any, key: str, depth: int = 0):
    if depth > 7:
        return None
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find(value, key, depth + 1)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj[:30]:
            found = _find(value, key, depth + 1)
            if found is not None:
                return found
    return None


class RealtimeTerminalDashboard:
    """Read-only 25-market realtime operations dashboard."""

    def __init__(self, settings, *, refresh_hz=4.0, public_feed=True, markets=None):
        if not 1.0 <= float(refresh_hz) <= 10.0:
            raise ValueError("refresh_hz must be between 1 and 10")
        self.settings = settings
        self.refresh_hz = float(refresh_hz)
        self.public_feed = bool(public_feed)
        self.bridge = CryptoLibraryBridge(settings.crypto_repo_root)
        self.markets = tuple(markets or ())
        self.prices = {}
        self.snapshot = {}
        self.feed_health = {}
        self.feed_error = None
        self.snapshot_error = None
        self.manager = None
        self.stop = asyncio.Event()
        self.recent = deque(maxlen=8)
        self.started = time.monotonic()
        self.ai_training_state = {
            "agents": [],
            "training": [],
        }
        self._last_ai_scan = 0.0
        self._last_account_scan = 0.0
        self.account_state = {}
        self.local_proactive = {}

    def _universe(self):
        payload = UniverseManager(self.settings).current(force_refresh=False)
        markets = tuple(payload.get("markets") or ())
        if len(markets) != 25:
            raise RuntimeError(f"expected 25 runtime markets, got {len(markets)}")
        return markets

    async def _feed(self):
        if not self.public_feed:
            return
        module = self.bridge.import_module("data.websocket_manager")
        self.manager = module.WebSocketManager(
            queue_size=10000,
            maximum_connection_attempts=20,
            inactivity_timeout=20.0,
            heartbeat=10.0,
            ticker_minimum_interval_seconds=0.0,
        )
        await self.manager.start({"bitvavo": {"ticker": list(self.markets)}})
        while not self.stop.is_set():
            try:
                event = await self.manager.next_event(timeout=1.0)
                self.feed_health = self.manager.health_state["bitvavo"].snapshot(
                    timedelta(seconds=20)
                )
                etype = getattr(event.event_type, "value", str(event.event_type))
                if etype != "ticker":
                    continue
                p = dict(event.payload or {})
                market = str(event.canonical_market)
                bid = _num(p.get("best_bid") or p.get("bid"))
                ask = _num(p.get("best_ask") or p.get("ask"))
                spread = None
                if bid and ask and ask >= bid:
                    mid = (bid + ask) / 2
                    spread = (ask - bid) / mid * 10000
                previous = dict(self.prices.get(market) or {})
                last = _num(p.get("last_price") or p.get("price"))
                if last is None:
                    last = previous.get("last")
                if bid is None:
                    bid = previous.get("bid")
                if ask is None:
                    ask = previous.get("ask")
                if bid and ask and ask >= bid:
                    mid = (bid + ask) / 2
                    spread = (
                        (ask - bid) / mid * 10000
                        if mid > 0
                        else None
                    )
                else:
                    spread = previous.get("spread")
                self.prices[market] = {
                    "last": last,
                    "bid": bid,
                    "ask": ask,
                    "spread": spread,
                    "seen": time.monotonic(),
                }
                self.recent.appendleft(market)
                self.feed_health = self.manager.health_state["bitvavo"].snapshot(
                    timedelta(seconds=20)
                )
            except TimeoutError:
                if self.manager is not None:
                    self.feed_health = self.manager.health_state[
                        "bitvavo"
                    ].snapshot(timedelta(seconds=20))
                continue
            except Exception as exc:
                self.feed_error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(1)

    def _canonical_sync(self):
        ui = self.bridge.import_module("ui.server")
        return dict(ui.build_ui_snapshot(self.bridge.settings()))

    async def _snapshot_loop(self):
        while not self.stop.is_set():
            try:
                self.snapshot = await asyncio.to_thread(self._canonical_sync)
                self.snapshot_error = None
            except Exception as exc:
                self.snapshot_error = f"{type(exc).__name__}: {exc}"

            now = time.monotonic()
            if now - self._last_ai_scan >= 2.0:
                try:
                    self.ai_training_state = await asyncio.to_thread(
                        collect_ai_training_state,
                        self.settings.project_root,
                        self.settings.crypto_repo_root,
                    )
                except Exception as exc:
                    self.ai_training_state = {
                        "agents": [],
                        "training": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                self._last_ai_scan = now
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=1.0)
            except TimeoutError:
                pass

    @staticmethod
    def _fmt(value):
        if value is None:
            return "-"
        if value >= 1000:
            return f"{value:,.2f}"
        if value >= 1:
            return f"{value:.4f}"
        return f"{value:.8f}"

    def _price_table(self):
        table = Table(
            title="25-market Bitvavo WebSocket",
            box=box.SIMPLE_HEAVY,
            expand=True,
        )
        for column in ("Market", "Mark", "Bid", "Ask", "Spread bps", "Age ms"):
            table.add_column(column, no_wrap=True)
        now = time.monotonic()
        for market in self.markets:
            row = self.prices.get(market, {})
            age = None if not row else (now - row["seen"]) * 1000
            mark = row.get("last")
            if mark is None and row.get("bid") is not None and row.get("ask") is not None:
                mark = (float(row["bid"]) + float(row["ask"])) / 2.0
            table.add_row(
                market,
                self._fmt(mark),
                self._fmt(row.get("bid")),
                self._fmt(row.get("ask")),
                "-" if row.get("spread") is None else f"{row['spread']:.2f}",
                "-" if age is None else f"{age:.0f}",
            )
        return table

    def _health_table(self):
        table = Table(title="Runtime health", box=box.SIMPLE, expand=True)
        table.add_column("Component")
        table.add_column("State")
        table.add_column("Detail")
        deps = _find(self.snapshot, "runtime_dependencies") or {}
        authority = _find(self.snapshot, "execution_authority") or {}
        health = self.feed_health or {}
        table.add_row(
            "Dashboard public WS",
            str(health.get("state", "STARTING" if self.public_feed else "DISABLED")),
            f"{float(health.get('throughput_per_second') or 0):.1f} msg/s, "
            f"lat={health.get('mean_latency_ms')}ms, "
            f"reconnects={health.get('reconnects', 0)}, "
            f"drops={health.get('dropped_messages', 0)}",
        )
        table.add_row(
            "Canonical public WS",
            str(deps.get("public_stream_ready", "-")),
            "",
        )
        table.add_row(
            "Canonical private WS",
            str(deps.get("private_stream_ready", "-")),
            "orders/fills/account",
        )
        table.add_row(
            "Orderflow WS",
            str(deps.get("orderflow_stream_ready", "-")),
            "book/trades",
        )
        table.add_row(
            "Service authority",
            str(deps.get("service_authority_active", "-")),
            str(authority.get("control_state", "-")),
        )
        if self.feed_error:
            table.add_row("Feed error", "ERROR", self.feed_error[:100])
        if self.snapshot_error:
            table.add_row("Snapshot error", "ERROR", self.snapshot_error[:100])
        return table

    def _latest_proactive_snapshot(self):
        candidates = [
            self.settings.project_root
            / "output/crypto_ai_swing/proactive/latest.json",
            self.settings.project_root
            / "output/crypto_ai_swing/modes/shadow/proactive/latest.json",
            self.settings.project_root
            / "output/crypto_ai_swing/modes/paper/proactive/latest.json",
            self.settings.project_root
            / "output/crypto_ai_swing/modes/live/proactive/latest.json",
        ]
        available = [path for path in candidates if path.is_file()]
        if not available:
            return {}
        path = max(available, key=lambda item: item.stat().st_mtime)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return dict(value) if isinstance(value, dict) else {}

    def _generic_table(self, title, keys):
        table = Table(title=title, box=box.SIMPLE, expand=True)
        table.add_column("Item")
        table.add_column("State")
        table.add_column("Detail")
        value = None
        if title == "Signals / decisions":
            proactive = self._latest_proactive_snapshot()
            signals = list(proactive.get("signals") or [])
            screen = dict(proactive.get("screen") or {})
            if signals:
                value = []
                for signal in signals:
                    row = dict(signal)
                    market = str(row.get("market") or "")
                    technical = dict(
                        (screen.get(market) or {}).get("technical") or {}
                    )
                    indicators = dict(technical.get("indicators") or {})
                    row["detail"] = (
                        f"score={float(row.get('score') or 0):.3f} "
                        f"{technical.get('breakout_state') or '-'} "
                        f"RSI={float(indicators.get('rsi_14') or 0):.1f} "
                        f"ADX={float(indicators.get('adx_14') or 0):.1f}"
                    )
                    value.append(row)
            elif screen:
                ranked = sorted(
                    screen.items(),
                    key=lambda item: float(
                        item[1].get("execution_adjusted_score", -999.0)
                        or -999.0
                    ),
                    reverse=True,
                )[:8]
                value = []
                for market, screen_row in ranked:
                    technical = dict(screen_row.get("technical") or {})
                    indicators = dict(technical.get("indicators") or {})
                    value.append(
                        {
                            "market": market,
                            "status": "WATCH",
                            "detail": (
                                f"screen={float(screen_row.get('execution_adjusted_score') or 0):.3f} "
                                f"{technical.get('breakout_state') or '-'} "
                                f"RSI={float(indicators.get('rsi_14') or 0):.1f} "
                                f"ADX={float(indicators.get('adx_14') or 0):.1f}"
                            ),
                        }
                    )

        if value is None:
            for key in keys:
                value = _find(self.snapshot, key)
                if value is not None:
                    break
        rows = []
        if isinstance(value, list):
            rows = [x for x in value if isinstance(x, dict)][:8]
        elif isinstance(value, dict):
            for key, item in list(value.items())[:8]:
                if isinstance(item, dict):
                    rows.append({"id": key, **item})
                else:
                    rows.append({"id": key, "value": item})
        for row in rows:
            item = (
                row.get("market")
                or row.get("symbol")
                or row.get("model")
                or row.get("id")
                or "-"
            )
            state = (
                row.get("status")
                or row.get("state")
                or row.get("action")
                or row.get("side")
                or "-"
            )
            detail = (
                row.get("detail")
                or row.get("score")
                or row.get("confidence")
                or row.get("price")
                or row.get("value")
                or row.get("reason")
                or row.get("reason_codes")
                or "-"
            )
            table.add_row(str(item), str(state), str(detail)[:100])
        if not rows:
            table.add_row("-", "no rows", "-")
        return table

    def _ai_table(self):
        table = Table(title="AI / agents", box=box.SIMPLE, expand=True)
        table.add_column("Agent")
        table.add_column("State")
        table.add_column("Authority")
        table.add_column("Detail")
        rows = list(self.ai_training_state.get("agents") or [])
        for row in rows[:8]:
            table.add_row(
                str(row.get("component") or "-"),
                str(row.get("state") or "-"),
                str(row.get("authority") or "-"),
                str(row.get("detail") or "-")[:80],
            )
        if not rows:
            error = self.ai_training_state.get("error")
            table.add_row(
                "-",
                "no AI artifacts",
                "READ_ONLY",
                str(error or "waiting for config/artifacts")[:80],
            )
        return table

    def _training_table(self):
        table = Table(
            title="Training / evidence",
            box=box.SIMPLE,
            expand=True,
        )
        table.add_column("Artifact")
        table.add_column("State")
        table.add_column("Metric")
        table.add_column("Updated")
        rows = list(self.ai_training_state.get("training") or [])
        for row in rows[:8]:
            table.add_row(
                str(row.get("component") or "-"),
                str(row.get("state") or "-"),
                str(row.get("detail") or "-")[:70],
                str(row.get("updated") or "-"),
            )
        if not rows:
            table.add_row(
                "-",
                "no training artifacts",
                "-",
                "-",
            )
        return table

    def render(self):
        layout = Layout()
        health = self.feed_health or {}
        public_state = health.get(
            "state",
            "STARTING" if self.public_feed else "DISABLED",
        )
        header = (
            f"CRYPTO AI SWING | universe={len(self.markets)} | "
            f"TUI={self.refresh_hz:.1f}Hz | publicWS={public_state} | "
            f"uptime={time.monotonic() - self.started:.0f}s | READ-ONLY"
        )
        layout.split_column(
            Layout(Panel(header, title="Realtime Operations"), size=3),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="right", ratio=2),
        )
        layout["left"].split_column(
            Layout(self._price_table(), ratio=3),
            Layout(self._health_table(), ratio=2),
        )
        layout["right"].split_column(
            Layout(orders_table(self.local_proactive, self.account_state), ratio=2),
            Layout(signals_table(self.local_proactive), ratio=3),
            Layout(cmc_table(self.local_proactive), ratio=2),
            Layout(wallet_table(self.account_state), ratio=2),
            Layout(self._ai_table(), ratio=2),
            Layout(self._training_table(), ratio=2),
        )
        return layout

    async def run(self):
        if not self.markets:
            self.markets = await asyncio.to_thread(self._universe)
        feed_task = asyncio.create_task(self._feed(), name="dashboard-feed")
        snapshot_task = asyncio.create_task(
            self._snapshot_loop(),
            name="dashboard-snapshot",
        )
        try:
            with Live(
                self.render(),
                screen=True,
                refresh_per_second=self.refresh_hz,
            ) as live:
                while True:
                    live.update(self.render(), refresh=True)
                    await asyncio.sleep(1 / self.refresh_hz)
        finally:
            self.stop.set()
            for task in (feed_task, snapshot_task):
                task.cancel()
            await asyncio.gather(
                feed_task,
                snapshot_task,
                return_exceptions=True,
            )
            if self.manager is not None:
                await self.manager.stop()


def run_terminal_dashboard(
    settings,
    *,
    refresh_hz=4.0,
    public_feed=True,
):
    try:
        asyncio.run(
            RealtimeTerminalDashboard(
                settings,
                refresh_hz=refresh_hz,
                public_feed=public_feed,
            ).run()
        )
    except KeyboardInterrupt:
        pass
