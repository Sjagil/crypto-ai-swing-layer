from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import sqlite3

import pandas as pd


class ForwardEvidenceLedger:
    """Append-only prospective signal evidence with causal outcome maturation."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cycles (
                cycle_id TEXT PRIMARY KEY,
                generated_at TEXT NOT NULL,
                mode TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS signal_observations (
                observation_id TEXT PRIMARY KEY,
                cycle_id TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                market TEXT NOT NULL,
                candle_ts TEXT,
                side TEXT NOT NULL,
                score REAL NOT NULL,
                edge_bps REAL NOT NULL,
                blocked INTEGER NOT NULL,
                blockers TEXT NOT NULL,
                context TEXT NOT NULL,
                FOREIGN KEY(cycle_id) REFERENCES cycles(cycle_id)
            );
            CREATE TABLE IF NOT EXISTS forward_outcomes (
                observation_id TEXT NOT NULL,
                horizon_hours INTEGER NOT NULL,
                matured_at TEXT NOT NULL,
                reference_price REAL NOT NULL,
                outcome_price REAL NOT NULL,
                return_bps REAL NOT NULL,
                mfe_bps REAL NOT NULL,
                mae_bps REAL NOT NULL,
                PRIMARY KEY(observation_id, horizon_hours),
                FOREIGN KEY(observation_id) REFERENCES signal_observations(observation_id)
            );
            CREATE INDEX IF NOT EXISTS idx_forward_signal_market_ts
                ON signal_observations(market, candle_ts);
            CREATE INDEX IF NOT EXISTS idx_forward_outcome_horizon
                ON forward_outcomes(horizon_hours);
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def _id(*parts: Any) -> str:
        import hashlib

        raw = "|".join(str(x) for x in parts).encode()
        return hashlib.sha256(raw).hexdigest()[:40]

    def append_cycle(
        self,
        payload: dict[str, Any],
        candle_times: dict[str, str],
    ) -> dict[str, Any]:
        generated = str(
            payload.get("generated_at") or datetime.now(timezone.utc).isoformat()
        )
        mode = str(payload.get("mode") or "unknown")
        cycle_id = self._id(generated, mode, ",".join(payload.get("markets", [])))
        self.conn.execute(
            "INSERT OR IGNORE INTO cycles VALUES (?,?,?,?)",
            (cycle_id, generated, mode, json.dumps(payload, sort_keys=True, default=str)),
        )
        blocker_map = {
            str(row.get("market")): list(row.get("blockers", []))
            for row in payload.get("blocked", [])
        }
        inserted = 0
        for signal in payload.get("signals", []):
            market = str(signal.get("market"))
            candle_ts = candle_times.get(market)
            blockers = blocker_map.get(market, [])
            observation_id = self._id(
                market,
                candle_ts or generated,
                signal.get("side"),
                signal.get("score"),
                signal.get("edge_bps"),
            )
            before = self.conn.total_changes
            self.conn.execute(
                "INSERT OR IGNORE INTO signal_observations VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    observation_id,
                    cycle_id,
                    generated,
                    market,
                    candle_ts,
                    str(signal.get("side")),
                    float(signal.get("score") or 0.0),
                    float(signal.get("edge_bps") or 0.0),
                    1 if blockers else 0,
                    json.dumps(blockers, sort_keys=True),
                    json.dumps(
                        payload.get("market_context", {}).get(market, {}),
                        sort_keys=True,
                        default=str,
                    ),
                ),
            )
            inserted += int(self.conn.total_changes > before)
        self.conn.commit()
        return {"cycle_id": cycle_id, "signal_observations_inserted": inserted}

    def status(self) -> dict[str, int]:
        cycles = self.conn.execute("SELECT COUNT(*) FROM cycles").fetchone()[0]
        signals = self.conn.execute(
            "SELECT COUNT(*) FROM signal_observations"
        ).fetchone()[0]
        blocked = self.conn.execute(
            "SELECT COUNT(*) FROM signal_observations WHERE blocked=1"
        ).fetchone()[0]
        return {
            "cycles": int(cycles),
            "signals": int(signals),
            "blocked_signals": int(blocked),
        }

    def outcome_status(self) -> dict[str, Any]:
        total = self.conn.execute("SELECT COUNT(*) FROM forward_outcomes").fetchone()[0]
        by_horizon = {
            int(horizon): int(count)
            for horizon, count in self.conn.execute(
                "SELECT horizon_hours, COUNT(*) FROM forward_outcomes "
                "GROUP BY horizon_hours ORDER BY horizon_hours"
            ).fetchall()
        }
        return {"outcomes": int(total), "by_horizon_hours": by_horizon}

    @staticmethod
    def _utc_index(frame: pd.DataFrame) -> pd.DataFrame:
        selected = frame.sort_index().copy()
        if not isinstance(selected.index, pd.DatetimeIndex):
            raise ValueError("forward maturation frame requires DatetimeIndex")
        if selected.index.tz is None:
            selected.index = selected.index.tz_localize("UTC")
        else:
            selected.index = selected.index.tz_convert("UTC")
        return selected

    def mature_from_frames(
        self,
        frames: Mapping[str, pd.DataFrame],
        *,
        horizons_hours: Iterable[int] = (1, 4, 24),
    ) -> dict[str, Any]:
        horizons = tuple(sorted({int(value) for value in horizons_hours if int(value) > 0}))
        if not horizons:
            raise ValueError("at least one positive forward horizon is required")
        normalized = {
            market: self._utc_index(frame)
            for market, frame in frames.items()
            if frame is not None and not frame.empty
        }
        observations = self.conn.execute(
            "SELECT observation_id, market, candle_ts, side FROM signal_observations "
            "WHERE candle_ts IS NOT NULL ORDER BY candle_ts"
        ).fetchall()
        inserted = 0
        evaluated = 0
        skipped_unmatured = 0
        skipped_missing_reference = 0
        now = datetime.now(timezone.utc).isoformat()

        for observation_id, market, candle_ts, side in observations:
            frame = normalized.get(str(market))
            if frame is None or frame.empty:
                continue
            try:
                candle_open = pd.Timestamp(candle_ts)
                candle_open = (
                    candle_open.tz_convert("UTC")
                    if candle_open.tzinfo
                    else candle_open.tz_localize("UTC")
                )
            except Exception:
                skipped_missing_reference += 1
                continue
            if candle_open not in frame.index:
                skipped_missing_reference += 1
                continue
            try:
                reference = float(frame.loc[candle_open, "close"])
            except Exception:
                skipped_missing_reference += 1
                continue
            if reference <= 0:
                skipped_missing_reference += 1
                continue
            decision_at = candle_open + pd.Timedelta(hours=1)
            for horizon in horizons:
                exists = self.conn.execute(
                    "SELECT 1 FROM forward_outcomes WHERE observation_id=? AND horizon_hours=?",
                    (observation_id, horizon),
                ).fetchone()
                if exists:
                    continue
                horizon_end = decision_at + pd.Timedelta(hours=horizon)
                window = frame.loc[
                    (frame.index >= decision_at) & (frame.index < horizon_end)
                ]
                evaluated += 1
                if window.empty:
                    skipped_unmatured += 1
                    continue
                last_close_time = window.index[-1] + pd.Timedelta(hours=1)
                if last_close_time < horizon_end:
                    skipped_unmatured += 1
                    continue
                outcome = float(window["close"].iloc[-1])
                maximum = float(window["high"].max())
                minimum = float(window["low"].min())
                return_bps = (outcome / reference - 1.0) * 10000.0
                mfe_bps = (maximum / reference - 1.0) * 10000.0
                mae_bps = (minimum / reference - 1.0) * 10000.0
                self.conn.execute(
                    "INSERT OR IGNORE INTO forward_outcomes VALUES (?,?,?,?,?,?,?,?)",
                    (
                        observation_id,
                        horizon,
                        now,
                        reference,
                        outcome,
                        return_bps,
                        mfe_bps,
                        mae_bps,
                    ),
                )
                inserted += 1
        self.conn.commit()
        return {
            "horizons_hours": list(horizons),
            "outcomes_inserted": inserted,
            "candidate_evaluations": evaluated,
            "skipped_unmatured": skipped_unmatured,
            "skipped_missing_reference": skipped_missing_reference,
            **self.outcome_status(),
        }

    def observation_markets(self) -> tuple[str, ...]:
        rows = self.conn.execute(
            "SELECT DISTINCT market FROM signal_observations ORDER BY market"
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def report(self) -> dict[str, Any]:
        rows = self.conn.execute(
            """
            SELECT o.horizon_hours, s.market, s.side, s.blocked,
                   COUNT(*) AS n,
                   AVG(o.return_bps), AVG(o.mfe_bps), AVG(o.mae_bps),
                   AVG(CASE WHEN o.return_bps > 0 THEN 1.0 ELSE 0.0 END)
            FROM forward_outcomes o
            JOIN signal_observations s ON s.observation_id=o.observation_id
            GROUP BY o.horizon_hours, s.market, s.side, s.blocked
            ORDER BY o.horizon_hours, s.market, s.side, s.blocked
            """
        ).fetchall()
        groups = [
            {
                "horizon_hours": int(row[0]),
                "market": str(row[1]),
                "side": str(row[2]),
                "blocked": bool(row[3]),
                "observations": int(row[4]),
                "mean_return_bps": float(row[5]),
                "mean_mfe_bps": float(row[6]),
                "mean_mae_bps": float(row[7]),
                "positive_return_rate": float(row[8]),
            }
            for row in rows
        ]
        return {
            "ledger": self.status(),
            "outcomes": self.outcome_status(),
            "groups": groups,
            "authority": "FORWARD_EVIDENCE_ONLY",
            "automatic_live_promotion": False,
        }
