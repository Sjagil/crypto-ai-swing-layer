from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import sqlite3

import numpy as np
import pandas as pd


CANONICAL_OUTCOME_SCHEMA = "forward_outcome_observed_time_next_open_v2"
LEGACY_OUTCOME_SCHEMA = "forward_outcome_candle_close_v1"


class ForwardEvidenceLedger:
    """Append-only prospective evidence with observed-time causal maturation.

    Version 2 never starts an outcome window before the actual signal observation
    time. Hourly evidence uses the first hourly open at or after observed_at as a
    conservative execution reference. The legacy v1 table is retained read-only.
    """

    def __init__(self, path: Path, *, decision_bucket_minutes: int = 15):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.decision_bucket_minutes = max(1, int(decision_bucket_minutes))
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
            CREATE TABLE IF NOT EXISTS forward_outcomes_v2 (
                observation_id TEXT NOT NULL,
                horizon_hours INTEGER NOT NULL,
                matured_at TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                entry_bar_ts TEXT NOT NULL,
                reference_price REAL NOT NULL,
                outcome_price REAL NOT NULL,
                return_bps REAL NOT NULL,
                mfe_bps REAL NOT NULL,
                mae_bps REAL NOT NULL,
                schema_version TEXT NOT NULL,
                PRIMARY KEY(observation_id, horizon_hours),
                FOREIGN KEY(observation_id) REFERENCES signal_observations(observation_id)
            );
            CREATE INDEX IF NOT EXISTS idx_forward_signal_market_ts
                ON signal_observations(market, candle_ts);
            CREATE INDEX IF NOT EXISTS idx_forward_signal_observed
                ON signal_observations(market, observed_at);
            CREATE INDEX IF NOT EXISTS idx_forward_outcome_horizon
                ON forward_outcomes(horizon_hours);
            CREATE INDEX IF NOT EXISTS idx_forward_outcome_v2_horizon
                ON forward_outcomes_v2(horizon_hours);
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

    @staticmethod
    def _utc_timestamp(raw: Any) -> pd.Timestamp:
        value = pd.Timestamp(raw)
        return value.tz_convert("UTC") if value.tzinfo else value.tz_localize("UTC")

    def _decision_bucket(self, observed_at: Any) -> str:
        value = self._utc_timestamp(observed_at)
        bucket = value.floor(f"{self.decision_bucket_minutes}min")
        return bucket.isoformat()

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
        decision_bucket = self._decision_bucket(generated)
        inserted = 0
        for signal in payload.get("signals", []):
            market = str(signal.get("market"))
            candle_ts = candle_times.get(market)
            blockers = blocker_map.get(market, [])
            # One canonical observation per market / side / 15m decision cluster.
            # Score changes inside the same cluster do not manufacture more samples.
            observation_id = self._id(
                "v2",
                market,
                decision_bucket,
                signal.get("side"),
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
        return {
            "cycle_id": cycle_id,
            "signal_observations_inserted": inserted,
            "decision_bucket_minutes": self.decision_bucket_minutes,
        }

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
        total = self.conn.execute(
            "SELECT COUNT(*) FROM forward_outcomes_v2"
        ).fetchone()[0]
        by_horizon = {
            int(horizon): int(count)
            for horizon, count in self.conn.execute(
                "SELECT horizon_hours, COUNT(*) FROM forward_outcomes_v2 "
                "GROUP BY horizon_hours ORDER BY horizon_hours"
            ).fetchall()
        }
        legacy_total = self.conn.execute(
            "SELECT COUNT(*) FROM forward_outcomes"
        ).fetchone()[0]
        return {
            "outcomes": int(total),
            "by_horizon_hours": by_horizon,
            "schema_version": CANONICAL_OUTCOME_SCHEMA,
            "legacy_v1_outcomes_retained": int(legacy_total),
        }

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

    def _deduplicated_observations(self) -> list[tuple[Any, ...]]:
        rows = self.conn.execute(
            "SELECT observation_id, market, candle_ts, side, observed_at "
            "FROM signal_observations ORDER BY observed_at, observation_id"
        ).fetchall()
        selected: list[tuple[Any, ...]] = []
        seen: set[tuple[str, str, str]] = set()
        for row in rows:
            observation_id, market, candle_ts, side, observed_at = row
            try:
                bucket = self._decision_bucket(observed_at)
            except Exception:
                bucket = str(observed_at)
            key = (str(market), bucket, str(side))
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
        return selected

    def mature_from_frames(
        self,
        frames: Mapping[str, pd.DataFrame],
        *,
        horizons_hours: Iterable[int] = (1, 4, 24),
    ) -> dict[str, Any]:
        horizons = tuple(
            sorted({int(value) for value in horizons_hours if int(value) > 0})
        )
        if not horizons:
            raise ValueError("at least one positive forward horizon is required")
        normalized = {
            market: self._utc_index(frame)
            for market, frame in frames.items()
            if frame is not None and not frame.empty
        }
        observations = self._deduplicated_observations()
        inserted = 0
        evaluated = 0
        skipped_unmatured = 0
        skipped_missing_reference = 0
        now = datetime.now(timezone.utc).isoformat()

        for observation_id, market, candle_ts, side, observed_at in observations:
            frame = normalized.get(str(market))
            if frame is None or frame.empty:
                continue
            try:
                observed = self._utc_timestamp(observed_at)
            except Exception:
                skipped_missing_reference += 1
                continue
            # With closed 1h bars, the first fully causal execution proxy is the
            # first hourly open at or after actual observation time.
            entry_bar_open = observed.ceil("1h")
            if entry_bar_open not in frame.index:
                skipped_unmatured += len(horizons)
                continue
            try:
                reference = float(frame.loc[entry_bar_open, "open"])
            except Exception:
                skipped_missing_reference += 1
                continue
            if reference <= 0 or not np.isfinite(reference):
                skipped_missing_reference += 1
                continue

            for horizon in horizons:
                exists = self.conn.execute(
                    "SELECT 1 FROM forward_outcomes_v2 "
                    "WHERE observation_id=? AND horizon_hours=?",
                    (observation_id, horizon),
                ).fetchone()
                if exists:
                    continue
                horizon_end = entry_bar_open + pd.Timedelta(int(horizon), unit="h")
                window = frame.loc[
                    (frame.index >= entry_bar_open) & (frame.index < horizon_end)
                ]
                evaluated += 1
                if window.empty:
                    skipped_unmatured += 1
                    continue
                last_close_time = (
                    pd.Timestamp(window.index[-1]) + pd.Timedelta(1, unit="h")
                )
                if last_close_time < horizon_end:
                    skipped_unmatured += 1
                    continue
                outcome = float(window["close"].iloc[-1])
                maximum = float(window["high"].max())
                minimum = float(window["low"].min())
                return_bps = (outcome / reference - 1.0) * 10_000.0
                mfe_bps = max(0.0, (maximum / reference - 1.0) * 10_000.0)
                mae_bps = min(0.0, (minimum / reference - 1.0) * 10_000.0)
                self.conn.execute(
                    "INSERT OR IGNORE INTO forward_outcomes_v2 "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        observation_id,
                        horizon,
                        now,
                        observed.isoformat(),
                        entry_bar_open.isoformat(),
                        reference,
                        outcome,
                        return_bps,
                        mfe_bps,
                        mae_bps,
                        CANONICAL_OUTCOME_SCHEMA,
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
            "decision_bucket_minutes": self.decision_bucket_minutes,
            "causal_reference": "FIRST_1H_OPEN_AT_OR_AFTER_OBSERVED_AT",
            **self.outcome_status(),
        }

    def observation_markets(self) -> tuple[str, ...]:
        rows = self.conn.execute(
            "SELECT DISTINCT market FROM signal_observations ORDER BY market"
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def _legacy_quality(self) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT COUNT(*), "
            "SUM(CASE WHEN mfe_bps < 0 THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN mae_bps > 0 THEN 1 ELSE 0 END) "
            "FROM forward_outcomes"
        ).fetchone()
        return {
            "schema_version": LEGACY_OUTCOME_SCHEMA,
            "outcomes": int(row[0] or 0),
            "negative_mfe_rows": int(row[1] or 0),
            "positive_mae_rows": int(row[2] or 0),
            "used_for_current_report": False,
        }

    @staticmethod
    def _summary_statistics(group: list[tuple[Any, ...]]) -> dict[str, Any]:
        """Return bounded descriptive prospective metrics for one signal cohort."""
        if not group:
            return {
                "observations": 0,
                "markets": 0,
                "mean_return_bps": None,
                "median_return_bps": None,
                "return_std_bps": None,
                "q25_return_bps": None,
                "q75_return_bps": None,
                "positive_return_rate": None,
                "mean_mfe_bps": None,
                "mean_mae_bps": None,
            }
        realized = np.asarray([float(row[4]) for row in group], dtype=float)
        mfe = np.asarray([float(row[7]) for row in group], dtype=float)
        mae = np.asarray([float(row[8]) for row in group], dtype=float)
        return {
            "observations": len(group),
            "markets": len({str(row[5]) for row in group}),
            "mean_return_bps": float(realized.mean()),
            "median_return_bps": float(np.median(realized)),
            "return_std_bps": (
                float(realized.std(ddof=1)) if len(realized) >= 2 else None
            ),
            "q25_return_bps": float(np.quantile(realized, 0.25)),
            "q75_return_bps": float(np.quantile(realized, 0.75)),
            "positive_return_rate": float(np.mean(realized > 0.0)),
            "mean_mfe_bps": float(np.maximum(mfe, 0.0).mean()),
            "mean_mae_bps": float(np.minimum(mae, 0.0).mean()),
        }

    def _calibration_rows(self) -> list[tuple[Any, ...]]:
        return self.conn.execute(
            """
            SELECT o.horizon_hours, s.side, s.blocked, s.edge_bps,
                   o.return_bps, s.market, s.observed_at, o.mfe_bps, o.mae_bps
            FROM forward_outcomes_v2 o
            JOIN signal_observations s ON s.observation_id=o.observation_id
            ORDER BY o.horizon_hours, s.observed_at, s.observation_id
            """
        ).fetchall()

    def _calibration(self) -> dict[str, Any]:
        rows = self._calibration_rows()
        result: dict[str, Any] = {}
        for horizon in sorted({int(row[0]) for row in rows}):
            selected = [row for row in rows if int(row[0]) == horizon]
            buy = [row for row in selected if str(row[1]).upper() == "BUY"]
            hold = [row for row in selected if str(row[1]).upper() == "HOLD"]
            unblocked_buy = [row for row in buy if not bool(row[2])]
            blocked_buy = [row for row in buy if bool(row[2])]

            calibration: dict[str, Any] = {
                "buy": self._summary_statistics(buy),
                "unblocked_buy": self._summary_statistics(unblocked_buy),
                "blocked_buy": self._summary_statistics(blocked_buy),
                "hold": self._summary_statistics(hold),
            }
            if unblocked_buy:
                predicted = np.asarray(
                    [float(row[3]) for row in unblocked_buy], dtype=float
                )
                realized = np.asarray(
                    [float(row[4]) for row in unblocked_buy], dtype=float
                )
                error = realized - predicted
                std = (
                    float(realized.std(ddof=1))
                    if len(realized) >= 2
                    else None
                )
                se = (
                    std / float(np.sqrt(len(realized)))
                    if std is not None
                    else None
                )
                mean_realized = float(realized.mean())
                mean_predicted = float(predicted.mean())
                calibration["unblocked_buy_edge_calibration"] = {
                    "mean_predicted_edge_bps": mean_predicted,
                    "median_predicted_edge_bps": float(np.median(predicted)),
                    "mean_realized_return_bps": mean_realized,
                    "median_realized_return_bps": float(np.median(realized)),
                    "mean_realized_minus_predicted_bps": float(error.mean()),
                    "mean_absolute_calibration_error_bps": float(
                        np.abs(error).mean()
                    ),
                    "realized_to_predicted_ratio": (
                        mean_realized / mean_predicted
                        if mean_predicted > 0
                        else None
                    ),
                    "mean_return_standard_error_bps": se,
                    "mean_return_lower_90_bps": (
                        mean_realized - 1.645 * se if se is not None else None
                    ),
                    "mean_return_upper_90_bps": (
                        mean_realized + 1.645 * se if se is not None else None
                    ),
                }
            else:
                calibration["unblocked_buy_edge_calibration"] = {
                    "mean_predicted_edge_bps": None,
                    "median_predicted_edge_bps": None,
                    "mean_realized_return_bps": None,
                    "median_realized_return_bps": None,
                    "mean_realized_minus_predicted_bps": None,
                    "mean_absolute_calibration_error_bps": None,
                    "realized_to_predicted_ratio": None,
                    "mean_return_standard_error_bps": None,
                    "mean_return_lower_90_bps": None,
                    "mean_return_upper_90_bps": None,
                }
            buy_n = len(unblocked_buy)
            calibration["sample_readiness"] = (
                "PRELIMINARY"
                if buy_n >= 30
                else "COLLECTING"
            )
            calibration["minimum_unblocked_buy_outcomes_for_preliminary"] = 30
            result[str(horizon)] = calibration
        return result

    def canary_readiness(
        self,
        *,
        primary_horizon_hours: int = 4,
        minimum_unblocked_buy_outcomes: int = 30,
        minimum_distinct_markets: int = 5,
        minimum_observation_span_hours: float = 72.0,
        minimum_mean_return_bps: float = 0.0,
        minimum_positive_return_rate: float = 0.50,
    ) -> dict[str, Any]:
        """Gate real canary approval on actual prospective BUY outcomes.

        This is intentionally a canary-only evidence gate. Passing it does not
        authorize scaling, model promotion, leverage, shorting, or withdrawals.
        """
        horizon = int(primary_horizon_hours)
        rows = [
            row
            for row in self._calibration_rows()
            if int(row[0]) == horizon
            and str(row[1]).upper() == "BUY"
            and not bool(row[2])
        ]
        summary = self._summary_statistics(rows)
        observed_times: list[pd.Timestamp] = []
        for row in rows:
            try:
                observed_times.append(self._utc_timestamp(row[6]))
            except Exception:
                continue
        span_hours = 0.0
        if len(observed_times) >= 2:
            span_hours = float(
                (max(observed_times) - min(observed_times)).total_seconds()
                / 3600.0
            )

        blockers: list[str] = []
        if int(summary["observations"]) < int(minimum_unblocked_buy_outcomes):
            blockers.append("INSUFFICIENT_UNBLOCKED_BUY_OUTCOMES")
        if int(summary["markets"]) < int(minimum_distinct_markets):
            blockers.append("INSUFFICIENT_PROSPECTIVE_MARKET_BREADTH")
        if span_hours < float(minimum_observation_span_hours):
            blockers.append("INSUFFICIENT_PROSPECTIVE_TIME_SPAN")

        mean_return = summary.get("mean_return_bps")
        if mean_return is None or float(mean_return) <= float(minimum_mean_return_bps):
            blockers.append("PROSPECTIVE_MEAN_RETURN_NOT_POSITIVE")
        positive_rate = summary.get("positive_return_rate")
        if (
            positive_rate is None
            or float(positive_rate) < float(minimum_positive_return_rate)
        ):
            blockers.append("PROSPECTIVE_HIT_RATE_WEAK")

        # Evidence that is merely sparse is still collecting. Once the sample
        # requirements are met, negative expectancy is classified separately.
        sparse_codes = {
            "INSUFFICIENT_UNBLOCKED_BUY_OUTCOMES",
            "INSUFFICIENT_PROSPECTIVE_MARKET_BREADTH",
            "INSUFFICIENT_PROSPECTIVE_TIME_SPAN",
        }
        if not blockers:
            status = "PRELIMINARY_CANARY_ELIGIBLE"
        elif any(code in sparse_codes for code in blockers):
            status = "COLLECTING"
        else:
            status = "NEGATIVE_PROSPECTIVE_EVIDENCE"

        calibration = None
        if rows:
            predicted = np.asarray([float(row[3]) for row in rows], dtype=float)
            realized = np.asarray([float(row[4]) for row in rows], dtype=float)
            error = realized - predicted
            std = float(realized.std(ddof=1)) if len(realized) >= 2 else None
            se = std / float(np.sqrt(len(realized))) if std is not None else None
            calibration = {
                "mean_predicted_edge_bps": float(predicted.mean()),
                "mean_realized_return_bps": float(realized.mean()),
                "mean_realized_minus_predicted_bps": float(error.mean()),
                "mean_absolute_calibration_error_bps": float(np.abs(error).mean()),
                "mean_return_lower_90_bps": (
                    float(realized.mean()) - 1.645 * se if se is not None else None
                ),
            }

        return {
            "schema_version": "prospective_canary_readiness_v1",
            "status": status,
            "eligible": not blockers,
            "primary_horizon_hours": horizon,
            "unblocked_buy": summary,
            "observation_span_hours": span_hours,
            "requirements": {
                "minimum_unblocked_buy_outcomes": int(minimum_unblocked_buy_outcomes),
                "minimum_distinct_markets": int(minimum_distinct_markets),
                "minimum_observation_span_hours": float(minimum_observation_span_hours),
                "minimum_mean_return_bps": float(minimum_mean_return_bps),
                "minimum_positive_return_rate": float(minimum_positive_return_rate),
            },
            "calibration": calibration,
            "blockers": blockers,
            "authority": "CANARY_ELIGIBILITY_ONLY",
            "automatic_live_promotion": False,
            "autoscale_authority": False,
        }

    def report(self) -> dict[str, Any]:
        rows = self.conn.execute(
            """
            SELECT o.horizon_hours, s.market, s.side, s.blocked,
                   COUNT(*) AS n,
                   AVG(o.return_bps), AVG(o.mfe_bps), AVG(o.mae_bps),
                   AVG(CASE WHEN o.return_bps > 0 THEN 1.0 ELSE 0.0 END)
            FROM forward_outcomes_v2 o
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
                "mean_mfe_bps": max(0.0, float(row[6])),
                "mean_mae_bps": min(0.0, float(row[7])),
                "positive_return_rate": float(row[8]),
            }
            for row in rows
        ]
        outcome_status = self.outcome_status()
        return {
            "ledger": self.status(),
            "outcomes": outcome_status,
            "groups": groups,
            "calibration": self._calibration(),
            "canary_readiness": self.canary_readiness(),
            "legacy_quality": self._legacy_quality(),
            "causality": {
                "schema_version": CANONICAL_OUTCOME_SCHEMA,
                "reference": "FIRST_1H_OPEN_AT_OR_AFTER_OBSERVED_AT",
                "decision_bucket_minutes": self.decision_bucket_minutes,
                "mfe_non_negative": True,
                "mae_non_positive": True,
                "legacy_v1_excluded_from_current_metrics": True,
            },
            "authority": "FORWARD_EVIDENCE_ONLY",
            "automatic_live_promotion": False,
        }
