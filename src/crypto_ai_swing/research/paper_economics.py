from __future__ import annotations

import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        decoded = json.loads(str(value or "{}"))
    except (TypeError, ValueError):
        return {}
    return dict(decoded) if isinstance(decoded, dict) else {}


def _utc_timestamp(value: Any) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    return stamp.tz_convert("UTC") if stamp.tzinfo else stamp.tz_localize("UTC")


class PaperEconomicEvaluator:
    """Evaluate realized paper trades and delayed active-swing evidence."""

    SCHEMA = "crypto_ai_swing_paper_economics_v1"

    def __init__(self, settings, *, mode: str = "shadow", crypto=None) -> None:
        self.settings = settings
        self.mode = str(mode).lower()
        self.project_root = Path(settings.project_root)
        autonomy = dict(getattr(settings, "autonomy", {}) or {})
        self.cfg = dict(autonomy.get("paper_economics", {}) or {})
        forward_cfg = dict(autonomy.get("forward_evidence", {}) or {})
        public_mode = "canary" if self.mode == "live" else self.mode
        self.paper_path = (
            self.project_root
            / "output/crypto_ai_swing/modes"
            / public_mode
            / "proactive/paper_ledger.sqlite"
        )
        self.forward_path = self.project_root / forward_cfg.get(
            "path",
            "output/crypto_ai_swing/forward/forward_v3.sqlite",
        )
        self.output_root = (
            self.project_root
            / "output/crypto_ai_swing/research/paper_economics"
            / public_mode
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.trade_path = self.output_root / "closed_trades.sqlite"
        self.latest_path = self.output_root / "latest.json"
        self.history_path = self.output_root / "history.jsonl"
        self.crypto = (
            crypto
            if crypto is not None
            else CryptoLibraryBridge(settings.crypto_repo_root)
        )
        self._initialize_trade_store()

    def _initialize_trade_store(self) -> None:
        with sqlite3.connect(self.trade_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS closed_trades (
                    sell_event_id TEXT PRIMARY KEY,
                    buy_event_id TEXT NOT NULL,
                    market TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    entry_notional_eur REAL NOT NULL,
                    ledger_realized_pnl_eur REAL NOT NULL,
                    cost_adjusted_pnl_eur REAL NOT NULL,
                    gross_return_bps REAL NOT NULL,
                    net_return_bps REAL NOT NULL,
                    stop_pct REAL,
                    r_multiple REAL,
                    mfe_bps REAL,
                    mae_bps REAL,
                    holding_hours REAL NOT NULL,
                    exit_reason TEXT NOT NULL,
                    expected_edge_bps REAL,
                    estimated_round_trip_cost_bps REAL
                );
                CREATE INDEX IF NOT EXISTS idx_round43_trade_strategy
                    ON closed_trades(strategy, closed_at);
                CREATE INDEX IF NOT EXISTS idx_round43_trade_market
                    ON closed_trades(market, closed_at);
                """
            )

    def _starting_equity(self) -> float:
        if self.paper_path.is_file():
            try:
                with sqlite3.connect(self.paper_path, timeout=5.0) as conn:
                    row = conn.execute(
                        "SELECT value FROM meta "
                        "WHERE key='starting_equity_eur'"
                    ).fetchone()
                if row:
                    return max(1.0, _number(row[0], 10000.0))
            except sqlite3.Error:
                return 10000.0
        proactive = dict(getattr(self.settings, "proactive", {}) or {})
        return max(
            1.0,
            _number(proactive.get("shadow_equity_eur"), 10000.0),
        )

    def _paper_events(self) -> list[dict[str, Any]]:
        if not self.paper_path.is_file():
            return []
        try:
            with sqlite3.connect(self.paper_path, timeout=5.0) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT rowid,event_id,created_at,side,market,quantity,
                           price,gross_eur,fee_eur,realized_pnl_eur,payload
                    FROM events
                    ORDER BY rowid
                    """
                ).fetchall()
        except sqlite3.Error:
            return []
        events: list[dict[str, Any]] = []
        for row in rows:
            event = dict(row)
            event["payload"] = _mapping(event.get("payload"))
            events.append(event)
        return events

    def _processed_sells(self) -> set[str]:
        with sqlite3.connect(self.trade_path, timeout=5.0) as conn:
            rows = conn.execute(
                "SELECT sell_event_id FROM closed_trades"
            ).fetchall()
        return {str(row[0]) for row in rows}

    def _frames(self, markets: list[str]) -> dict[str, pd.DataFrame]:
        selected = sorted(set(markets))
        if not selected or self.crypto is None:
            return {}
        try:
            return dict(
                self.crypto.ohlcv_many(
                    selected,
                    "1h",
                    persist=False,
                    concurrency=min(4, len(selected)),
                )
                or {}
            )
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _excursions(
        frame: pd.DataFrame | None,
        *,
        opened_at: str,
        closed_at: str,
        entry_price: float,
    ) -> tuple[float | None, float | None]:
        if (
            frame is None
            or frame.empty
            or entry_price <= 0
            or "high" not in frame
            or "low" not in frame
        ):
            return None, None
        opened = _utc_timestamp(opened_at)
        closed = _utc_timestamp(closed_at)
        if opened is None or closed is None:
            return None, None
        selected = frame.copy()
        if not isinstance(selected.index, pd.DatetimeIndex):
            return None, None
        if selected.index.tz is None:
            selected.index = selected.index.tz_localize("UTC")
        else:
            selected.index = selected.index.tz_convert("UTC")
        window = selected.loc[
            (selected.index >= opened.floor("1h"))
            & (selected.index <= closed.ceil("1h"))
        ]
        if window.empty:
            return None, None
        high = pd.to_numeric(window["high"], errors="coerce").max()
        low = pd.to_numeric(window["low"], errors="coerce").min()
        if pd.isna(high) or pd.isna(low):
            return None, None
        return (
            max(0.0, (float(high) / entry_price - 1.0) * 10000.0),
            min(0.0, (float(low) / entry_price - 1.0) * 10000.0),
        )

    def _sync_closed_trades(self) -> int:
        events = self._paper_events()
        if not events:
            return 0
        processed = self._processed_sells()
        open_buys: dict[str, dict[str, Any]] = {}
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for event in events:
            market = str(event.get("market") or "").upper()
            side = str(event.get("side") or "").upper()
            if side == "BUY":
                open_buys[market] = event
                continue
            if side != "SELL" or str(event.get("event_id")) in processed:
                continue
            buy = open_buys.pop(market, None)
            if buy is not None:
                pairs.append((buy, event))

        maximum = max(
            1,
            int(self.cfg.get("maximum_new_closed_trades_per_refresh", 24)),
        )
        pairs = pairs[:maximum]
        frames = self._frames(
            [str(sell["market"]).upper() for _, sell in pairs]
        )
        inserted = 0
        with sqlite3.connect(self.trade_path, timeout=5.0) as conn:
            for buy, sell in pairs:
                entry_price = _number(buy.get("price"))
                exit_price = _number(sell.get("price"))
                quantity = min(
                    _number(buy.get("quantity")),
                    _number(sell.get("quantity")),
                )
                entry_notional = max(
                    _number(buy.get("gross_eur")),
                    quantity * entry_price,
                )
                if (
                    entry_price <= 0
                    or exit_price <= 0
                    or quantity <= 0
                    or entry_notional <= 0
                ):
                    continue

                buy_payload = dict(buy.get("payload") or {})
                sell_payload = dict(sell.get("payload") or {})
                metadata = dict(buy_payload.get("intent_metadata") or {})
                ledger_pnl = _number(sell.get("realized_pnl_eur"))
                gross_return_bps = (
                    exit_price / entry_price - 1.0
                ) * 10000.0
                estimated_cost_bps = _number(
                    buy_payload.get("estimated_round_trip_cost_bps"),
                    float("nan"),
                )
                if math.isfinite(estimated_cost_bps):
                    net_return_bps = gross_return_bps - estimated_cost_bps
                    cost_adjusted_pnl = (
                        entry_notional * net_return_bps / 10000.0
                    )
                else:
                    cost_adjusted_pnl = ledger_pnl
                    net_return_bps = (
                        cost_adjusted_pnl / entry_notional
                    ) * 10000.0

                opened = _utc_timestamp(buy.get("created_at"))
                closed = _utc_timestamp(sell.get("created_at"))
                holding_hours = 0.0
                if opened is not None and closed is not None:
                    holding_hours = max(
                        0.0,
                        float((closed - opened).total_seconds() / 3600.0),
                    )

                stop_pct = _number(
                    buy_payload.get(
                        "stop_pct",
                        metadata.get("stop_pct"),
                    ),
                    0.0,
                )
                initial_risk = entry_notional * stop_pct
                r_multiple = (
                    cost_adjusted_pnl / initial_risk
                    if initial_risk > 0
                    else None
                )
                market = str(sell.get("market") or "").upper()
                mfe_bps, mae_bps = self._excursions(
                    frames.get(market),
                    opened_at=str(buy.get("created_at") or ""),
                    closed_at=str(sell.get("created_at") or ""),
                    entry_price=entry_price,
                )
                strategy = str(
                    buy_payload.get("strategy")
                    or metadata.get("strategy")
                    or "UNKNOWN"
                )
                exit_reason = str(
                    sell_payload.get("reason")
                    or sell_payload.get("exit_reason")
                    or "UNKNOWN"
                )
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO closed_trades(
                        sell_event_id,buy_event_id,market,strategy,
                        opened_at,closed_at,quantity,entry_price,exit_price,
                        entry_notional_eur,ledger_realized_pnl_eur,
                        cost_adjusted_pnl_eur,gross_return_bps,net_return_bps,
                        stop_pct,r_multiple,mfe_bps,mae_bps,holding_hours,
                        exit_reason,expected_edge_bps,
                        estimated_round_trip_cost_bps
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        str(sell.get("event_id")),
                        str(buy.get("event_id")),
                        market,
                        strategy,
                        str(buy.get("created_at")),
                        str(sell.get("created_at")),
                        quantity,
                        entry_price,
                        exit_price,
                        entry_notional,
                        ledger_pnl,
                        cost_adjusted_pnl,
                        gross_return_bps,
                        net_return_bps,
                        stop_pct if stop_pct > 0 else None,
                        r_multiple,
                        mfe_bps,
                        mae_bps,
                        holding_hours,
                        exit_reason,
                        _number(
                            buy_payload.get("expected_edge_bps"),
                            float("nan"),
                        ),
                        estimated_cost_bps,
                    ),
                )
                inserted += int(cursor.rowcount > 0)
            conn.commit()
        return inserted

    def _rows(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.trade_path, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM closed_trades ORDER BY closed_at"
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _maximum_drawdown(
        pnl_values: list[float],
        starting_equity: float,
    ) -> float:
        equity = starting_equity
        peak = starting_equity
        maximum = 0.0
        for pnl in pnl_values:
            equity += pnl
            peak = max(peak, equity)
            if peak > 0:
                maximum = max(maximum, (peak - equity) / peak)
        return float(maximum)

    @classmethod
    def _aggregate(
        cls,
        rows: list[dict[str, Any]],
        starting_equity: float,
    ) -> dict[str, Any]:
        if not rows:
            return {
                "closed_trades": 0,
                "cost_adjusted_pnl_eur": 0.0,
                "ledger_realized_pnl_eur": 0.0,
                "win_rate": None,
                "expectancy_eur": None,
                "expectancy_bps": None,
                "profit_factor": None,
                "payoff_ratio": None,
                "average_r_multiple": None,
                "mean_net_return_bps": None,
                "median_net_return_bps": None,
                "trade_path_compounded_return_fraction": None,
                "maximum_realized_drawdown_fraction": 0.0,
                "average_holding_hours": None,
                "mean_mfe_bps": None,
                "mean_mae_bps": None,
            }

        pnl_values = [
            _number(row["cost_adjusted_pnl_eur"]) for row in rows
        ]
        ledger_values = [
            _number(row["ledger_realized_pnl_eur"]) for row in rows
        ]
        returns = [_number(row["net_return_bps"]) for row in rows]
        wins = [value for value in pnl_values if value > 0]
        losses = [-value for value in pnl_values if value < 0]
        win_returns = [
            _number(row["net_return_bps"])
            for row in rows
            if _number(row["cost_adjusted_pnl_eur"]) > 0
        ]
        loss_returns = [
            -_number(row["net_return_bps"])
            for row in rows
            if _number(row["cost_adjusted_pnl_eur"]) < 0
        ]
        average_win = (
            sum(win_returns) / len(win_returns) if win_returns else 0.0
        )
        average_loss = (
            sum(loss_returns) / len(loss_returns) if loss_returns else 0.0
        )
        win_rate = len(wins) / len(rows)
        expectancy_bps = (
            win_rate * average_win
            - (1.0 - win_rate) * average_loss
        )
        gross_profit = sum(wins)
        gross_loss = sum(losses)
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else None
        )
        payoff_ratio = (
            average_win / average_loss if average_loss > 0 else None
        )
        r_values = [
            _number(row["r_multiple"])
            for row in rows
            if row["r_multiple"] is not None
        ]
        mfe_values = [
            _number(row["mfe_bps"])
            for row in rows
            if row["mfe_bps"] is not None
        ]
        mae_values = [
            _number(row["mae_bps"])
            for row in rows
            if row["mae_bps"] is not None
        ]
        compounded = 1.0
        for net_return in returns:
            compounded *= max(
                1e-9,
                1.0 + net_return / 10000.0,
            )
        return {
            "closed_trades": len(rows),
            "cost_adjusted_pnl_eur": float(sum(pnl_values)),
            "ledger_realized_pnl_eur": float(sum(ledger_values)),
            "win_rate": float(win_rate),
            "expectancy_eur": float(sum(pnl_values) / len(rows)),
            "expectancy_bps": float(expectancy_bps),
            "profit_factor": (
                float(profit_factor)
                if profit_factor is not None
                else None
            ),
            "payoff_ratio": (
                float(payoff_ratio)
                if payoff_ratio is not None
                else None
            ),
            "average_r_multiple": (
                float(sum(r_values) / len(r_values))
                if r_values
                else None
            ),
            "mean_net_return_bps": float(sum(returns) / len(returns)),
            "median_net_return_bps": float(median(returns)),
            "trade_path_compounded_return_fraction": float(
                compounded - 1.0
            ),
            "maximum_realized_drawdown_fraction": (
                cls._maximum_drawdown(
                    pnl_values,
                    starting_equity,
                )
            ),
            "average_holding_hours": float(
                sum(_number(row["holding_hours"]) for row in rows)
                / len(rows)
            ),
            "mean_mfe_bps": (
                float(sum(mfe_values) / len(mfe_values))
                if mfe_values
                else None
            ),
            "mean_mae_bps": (
                float(sum(mae_values) / len(mae_values))
                if mae_values
                else None
            ),
        }

    def _forward_horizons(self) -> dict[str, Any]:
        horizons = tuple(
            int(value)
            for value in self.cfg.get(
                "swing_horizons_hours",
                [4, 24, 72, 168],
            )
        )
        if not self.forward_path.is_file():
            return {
                "status": "NO_FORWARD_DATABASE",
                "by_horizon": {},
                "complete_long_horizon_observations": 0,
            }

        by_horizon: dict[str, Any] = {}
        observation_horizons: dict[str, set[int]] = {}
        required = {value for value in horizons if value >= 24}
        try:
            with sqlite3.connect(self.forward_path, timeout=5.0) as conn:
                rows = conn.execute(
                    """
                    SELECT s.observation_id,s.market,o.horizon_hours,
                           o.return_bps,o.mfe_bps,o.mae_bps
                    FROM forward_outcomes_v2 o
                    JOIN signal_observations s
                      ON s.observation_id=o.observation_id
                    WHERE s.side='BUY' AND s.blocked=0
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            return {
                "status": "ERROR",
                "error": f"{type(exc).__name__}:{str(exc)[:300]}",
                "by_horizon": {},
                "complete_long_horizon_observations": 0,
            }

        grouped: dict[int, list[tuple[Any, ...]]] = {
            value: [] for value in horizons
        }
        for row in rows:
            observation_id = str(row[0])
            horizon = int(row[2])
            if horizon not in grouped:
                continue
            grouped[horizon].append(row)
            observation_horizons.setdefault(
                observation_id,
                set(),
            ).add(horizon)

        for horizon, values in grouped.items():
            returns = [_number(row[3]) for row in values]
            mfe_values = [_number(row[4]) for row in values]
            mae_values = [_number(row[5]) for row in values]
            by_horizon[str(horizon)] = {
                "observations": len(values),
                "markets": len({str(row[1]) for row in values}),
                "mean_return_bps": (
                    float(sum(returns) / len(returns))
                    if returns
                    else None
                ),
                "positive_fraction": (
                    float(
                        sum(value > 0 for value in returns)
                        / len(returns)
                    )
                    if returns
                    else None
                ),
                "mean_mfe_bps": (
                    float(sum(mfe_values) / len(mfe_values))
                    if mfe_values
                    else None
                ),
                "mean_mae_bps": (
                    float(sum(mae_values) / len(mae_values))
                    if mae_values
                    else None
                ),
            }

        complete = sum(
            1
            for values in observation_horizons.values()
            if required and required.issubset(values)
        )
        weights = dict(
            self.cfg.get(
                "swing_horizon_weights",
                {"4": 0.10, "24": 0.35, "72": 0.35, "168": 0.20},
            )
            or {}
        )
        weighted_sum = 0.0
        used_weight = 0.0
        for key, row in by_horizon.items():
            mean_value = row.get("mean_return_bps")
            if mean_value is None:
                continue
            weight = max(0.0, _number(weights.get(key), 0.0))
            weighted_sum += float(mean_value) * weight
            used_weight += weight

        return {
            "status": "READY",
            "by_horizon": by_horizon,
            "complete_long_horizon_observations": complete,
            "available_weighted_mean_return_bps": (
                weighted_sum / used_weight
                if used_weight > 0
                else None
            ),
            "weights": weights,
        }

    def refresh(self) -> dict[str, Any]:
        inserted = self._sync_closed_trades()
        rows = self._rows()
        starting_equity = self._starting_equity()
        overall = self._aggregate(rows, starting_equity)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["strategy"]), []).append(row)
        per_strategy = {
            strategy: self._aggregate(values, starting_equity)
            for strategy, values in sorted(grouped.items())
        }

        minimum_strategy = int(
            self.cfg.get("minimum_strategy_closed_trades", 8)
        )
        leader: dict[str, Any] | None = None
        quarantine: list[dict[str, Any]] = []
        for strategy, metrics in per_strategy.items():
            closed = int(metrics.get("closed_trades") or 0)
            expectancy = _number(
                metrics.get("expectancy_bps"),
                -1e9,
            )
            profit_factor = _number(
                metrics.get("profit_factor"),
                0.0,
            )
            drawdown = _number(
                metrics.get("maximum_realized_drawdown_fraction"),
                1.0,
            )
            no_loss_positive_sample = bool(
                metrics.get("profit_factor") is None
                and _number(metrics.get("win_rate")) >= 1.0
                and _number(
                    metrics.get("cost_adjusted_pnl_eur")
                )
                > 0
            )
            profit_factor_pass = bool(
                no_loss_positive_sample
                or profit_factor > 1
            )
            if (
                closed >= minimum_strategy
                and expectancy > 0
                and profit_factor_pass
            ):
                shrinkage = closed / (closed + minimum_strategy)
                score = (
                    expectancy
                    * shrinkage
                    / (1.0 + 5.0 * max(0.0, drawdown))
                )
                candidate = {
                    "strategy": strategy,
                    "economic_score": float(score),
                    **metrics,
                }
                if (
                    leader is None
                    or score > float(leader["economic_score"])
                ):
                    leader = candidate
            if (
                closed >= minimum_strategy
                and (expectancy < 0 or profit_factor < 0.90)
            ):
                quarantine.append(
                    {
                        "strategy": strategy,
                        "reason": (
                            "NEGATIVE_REALIZED_PAPER_ECONOMICS"
                        ),
                        "closed_trades": closed,
                        "expectancy_bps": expectancy,
                        "profit_factor": metrics.get("profit_factor"),
                    }
                )

        horizons = self._forward_horizons()
        diagnostics_minimum = int(
            self.cfg.get("minimum_closed_trades_for_diagnostics", 10)
        )
        qualification_minimum = int(
            self.cfg.get("minimum_closed_trades_for_qualification", 30)
        )
        minimum_profit_factor = float(
            self.cfg.get("minimum_profit_factor", 1.15)
        )
        minimum_payoff_ratio = float(
            self.cfg.get("minimum_payoff_ratio", 1.05)
        )
        maximum_drawdown = float(
            self.cfg.get(
                "maximum_realized_drawdown_fraction",
                0.20,
            )
        )
        minimum_long_horizon = int(
            self.cfg.get(
                "minimum_complete_long_horizon_observations",
                30,
            )
        )
        closed_trades = int(overall.get("closed_trades") or 0)
        checks = {
            "positive_cost_adjusted_pnl": (
                _number(overall.get("cost_adjusted_pnl_eur")) > 0
            ),
            "positive_expectancy": (
                _number(overall.get("expectancy_bps"), -1.0) > 0
            ),
            "profit_factor": bool(
                (
                    overall.get("profit_factor") is None
                    and _number(overall.get("win_rate")) >= 1.0
                    and _number(
                        overall.get("cost_adjusted_pnl_eur")
                    )
                    > 0
                )
                or _number(overall.get("profit_factor"))
                >= minimum_profit_factor
            ),
            "payoff_ratio": bool(
                (
                    overall.get("payoff_ratio") is None
                    and _number(overall.get("win_rate")) >= 1.0
                    and _number(
                        overall.get("cost_adjusted_pnl_eur")
                    )
                    > 0
                )
                or _number(overall.get("payoff_ratio"))
                >= minimum_payoff_ratio
            ),
            "drawdown": (
                _number(
                    overall.get(
                        "maximum_realized_drawdown_fraction"
                    ),
                    1.0,
                )
                <= maximum_drawdown
            ),
        }
        paper_pass = (
            closed_trades >= diagnostics_minimum
            and all(checks.values())
        )
        if closed_trades < diagnostics_minimum:
            status = "COLLECTING"
        elif paper_pass and closed_trades < qualification_minimum:
            status = "PROMISING"
        elif paper_pass:
            status = "PAPER_ECONOMIC_PASS"
        else:
            status = "DEGRADED"

        complete_long = int(
            horizons.get("complete_long_horizon_observations") or 0
        )
        weighted_return = horizons.get(
            "available_weighted_mean_return_bps"
        )
        long_horizon_positive = (
            complete_long >= minimum_long_horizon
            and weighted_return is not None
            and float(weighted_return) > 0
        )
        economic_qualification = bool(
            status == "PAPER_ECONOMIC_PASS"
            and long_horizon_positive
        )

        priorities: list[str] = []
        if closed_trades >= diagnostics_minimum:
            if not checks["positive_expectancy"]:
                priorities.append("IMPROVE_NET_EXPECTANCY")
            if not checks["profit_factor"]:
                priorities.append("IMPROVE_PROFIT_FACTOR")
            if not checks["payoff_ratio"]:
                priorities.append("IMPROVE_PAYOFF_ASYMMETRY")
            if not checks["drawdown"]:
                priorities.append("REDUCE_PAPER_DRAWDOWN")
            mfe = overall.get("mean_mfe_bps")
            mae = overall.get("mean_mae_bps")
            if (
                mfe is not None
                and mae is not None
                and abs(float(mae)) > float(mfe)
            ):
                priorities.append("IMPROVE_ENTRY_TIMING_REDUCE_MAE")
        four_hour = dict(
            horizons.get("by_horizon") or {}
        ).get("4", {})
        if (
            int(four_hour.get("observations") or 0) >= 20
            and _number(four_hour.get("mean_return_bps")) <= 0
        ):
            priorities.append("IMPROVE_4H_ENTRY_QUALITY")
        if (
            complete_long >= minimum_long_horizon
            and not long_horizon_positive
        ):
            priorities.append("IMPROVE_SWING_FOLLOW_THROUGH")
        if leader is None:
            priorities.append(
                "DISCOVER_ECONOMICALLY_ROBUST_STRATEGY"
            )

        report = {
            "schema_version": self.SCHEMA,
            "generated_at": _utc_now(),
            "mode": self.mode,
            "status": status,
            "new_closed_trades_attributed": inserted,
            "overall": overall,
            "per_strategy": per_strategy,
            "paper_economic_leader": leader,
            "research_quarantine_candidates": quarantine,
            "forward_swing_horizons": horizons,
            "checks": checks,
            "economic_qualification": economic_qualification,
            "research_priorities": list(dict.fromkeys(priorities)),
            "objective": (
                "maximize risk-adjusted compounded net active-swing "
                "pnl after full estimated costs"
            ),
            "ratio_semantics": {
                "profit_factor_null_when_no_losses": True,
                "payoff_ratio_null_when_no_losses": True,
                "all_winning_positive_sample_passes_ratio_checks": True,
            },
            "paper_only": True,
            "live_decision_influence": False,
            "automatic_live_authority": False,
            "automatic_live_promotion": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        self.latest_path.write_text(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    report,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
        return report

    def status(self) -> dict[str, Any]:
        try:
            value = json.loads(
                self.latest_path.read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError):
            value = {}
        if isinstance(value, dict) and value:
            return dict(value)
        return {
            "schema_version": self.SCHEMA,
            "status": "NOT_BUILT",
            "mode": self.mode,
            "automatic_live_authority": False,
            "automatic_live_promotion": False,
        }
