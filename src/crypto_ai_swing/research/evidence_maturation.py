from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.research.forward import ForwardEvidenceLedger


def _now() -> datetime:
    return datetime.now(UTC)


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _age_hours(value: Any) -> float | None:
    parsed = _parse_time(value)
    if parsed is None:
        return None
    return max(0.0, (_now() - parsed.astimezone(UTC)).total_seconds() / 3600.0)


class ProspectiveEvidenceMaturation:
    """Persistent causal forward-evidence maturation and sufficiency diagnostics."""

    SCHEMA = "crypto_ai_swing_round43_evidence_maturation_v1"

    def __init__(self, settings) -> None:
        self.settings = settings
        self.project_root = Path(settings.project_root)
        self.autonomy = dict(getattr(settings, "autonomy", {}) or {})
        self.cfg = dict(self.autonomy.get("evidence_maturation", {}) or {})
        self.forward_cfg = dict(self.autonomy.get("forward_evidence", {}) or {})
        self.strategy_cfg = dict(self.autonomy.get("strategy_lab", {}) or {})
        self.selector_cfg = dict(self.autonomy.get("entry_selector", {}) or {})
        self.crypto = CryptoLibraryBridge(settings.crypto_repo_root)
        self.path = self.project_root / self.forward_cfg.get(
            "path",
            "output/crypto_ai_swing/forward/forward_v3.sqlite",
        )
        self.root = (
            self.project_root
            / "output/crypto_ai_swing/research/evidence_maturation"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.root / "latest.json"
        self.history_path = self.root / "history.jsonl"
        self.state_path = self.root / "state.json"
        self.state = self._read_json(self.state_path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return dict(value) if isinstance(value, dict) else {}
        except (OSError, TypeError, ValueError):
            return {}

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        temp.replace(path)

    def _ledger(self) -> ForwardEvidenceLedger:
        return ForwardEvidenceLedger(
            self.path,
            decision_bucket_minutes=int(
                self.forward_cfg.get("decision_bucket_minutes", 15)
            ),
        )

    def _primary_horizon(self) -> int:
        readiness = dict(self.forward_cfg.get("canary_readiness", {}) or {})
        return int(
            readiness.get(
                "primary_horizon_hours",
                self.strategy_cfg.get("horizon_hours", 4),
            )
        )

    def _required_horizons(self) -> tuple[int, ...]:
        values = self.selector_cfg.get("horizons_hours", (24, 72, 168))
        return tuple(sorted({int(value) for value in values if int(value) > 0}))

    def _all_horizons(self) -> tuple[int, ...]:
        values = self.forward_cfg.get("horizons_hours", (1, 4, 24, 72, 168))
        return tuple(sorted({int(value) for value in values if int(value) > 0}))

    def _optimization_minimum(self) -> int:
        configured = self.cfg.get(
            "minimum_complete_horizon_observations_for_optimization"
        )
        if configured is not None:
            return max(1, int(configured))
        minimum_train = int(
            self.selector_cfg.get("minimum_train_observations", 24)
        )
        return max(60, minimum_train * 3)

    def _execution_grace_hours(self) -> float:
        timeframe = str(self.forward_cfg.get("execution_timeframe", "15m"))
        return {
            "5m": 1.0 / 12.0,
            "15m": 0.25,
            "1h": 1.0,
            "2h": 2.0,
            "4h": 4.0,
            "1d": 24.0,
        }.get(timeframe, 1.0)

    def _database_snapshot(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {
                "status": "NO_DATABASE",
                "cycles": 0,
                "signals": 0,
                "unblocked_buy_observations": 0,
                "outcomes": 0,
                "outcomes_by_horizon": {},
                "pending_by_horizon": {},
                "complete_required_horizon_observations": 0,
            }

        conn = sqlite3.connect(self.path)
        try:
            cycles = int(
                conn.execute("SELECT COUNT(*) FROM cycles").fetchone()[0] or 0
            )
            signals = int(
                conn.execute(
                    "SELECT COUNT(*) FROM signal_observations"
                ).fetchone()[0]
                or 0
            )
            blocked = int(
                conn.execute(
                    "SELECT COUNT(*) FROM signal_observations WHERE blocked=1"
                ).fetchone()[0]
                or 0
            )
            buy_summary = conn.execute(
                """
                SELECT COUNT(*), COUNT(DISTINCT market),
                       MIN(observed_at), MAX(observed_at)
                FROM signal_observations
                WHERE side='BUY' AND blocked=0
                """
            ).fetchone()
            unblocked_buy = int(buy_summary[0] or 0)
            unblocked_buy_markets = int(buy_summary[1] or 0)
            first_observed_at = buy_summary[2]
            latest_observed_at = buy_summary[3]

            outcome_rows = conn.execute(
                """
                SELECT o.horizon_hours, COUNT(*), COUNT(DISTINCT s.market)
                FROM forward_outcomes_v2 o
                JOIN signal_observations s
                  ON s.observation_id=o.observation_id
                WHERE s.side='BUY' AND s.blocked=0
                GROUP BY o.horizon_hours
                ORDER BY o.horizon_hours
                """
            ).fetchall()
            outcomes_by_horizon = {
                int(horizon): int(count)
                for horizon, count, _ in outcome_rows
            }
            markets_by_horizon = {
                int(horizon): int(markets)
                for horizon, _, markets in outcome_rows
            }
            outcomes = sum(outcomes_by_horizon.values())

            primary = self._primary_horizon()
            primary_stats = conn.execute(
                """
                SELECT COUNT(*), COUNT(DISTINCT s.market),
                       MIN(s.observed_at), MAX(s.observed_at),
                       AVG(o.return_bps),
                       AVG(CASE WHEN o.return_bps > 0 THEN 1.0 ELSE 0.0 END),
                       MAX(o.matured_at)
                FROM forward_outcomes_v2 o
                JOIN signal_observations s
                  ON s.observation_id=o.observation_id
                WHERE s.side='BUY' AND s.blocked=0
                  AND o.horizon_hours=?
                """,
                (primary,),
            ).fetchone()
            primary_count = int(primary_stats[0] or 0)
            primary_markets = int(primary_stats[1] or 0)
            primary_first = primary_stats[2]
            primary_last = primary_stats[3]
            primary_mean = (
                float(primary_stats[4])
                if primary_stats[4] is not None
                else None
            )
            primary_positive = (
                float(primary_stats[5])
                if primary_stats[5] is not None
                else None
            )
            latest_matured_at = primary_stats[6]

            required = self._required_horizons()
            complete_required = 0
            if required:
                placeholders = ",".join("?" for _ in required)
                complete_required = int(
                    conn.execute(
                        f"""
                        SELECT COUNT(*) FROM (
                            SELECT s.observation_id
                            FROM signal_observations s
                            JOIN forward_outcomes_v2 o
                              ON o.observation_id=s.observation_id
                            WHERE s.side='BUY' AND s.blocked=0
                              AND o.horizon_hours IN ({placeholders})
                            GROUP BY s.observation_id
                            HAVING COUNT(DISTINCT o.horizon_hours)=?
                        )
                        """,
                        (*required, len(required)),
                    ).fetchone()[0]
                    or 0
                )

            observations = conn.execute(
                """
                SELECT observation_id, observed_at
                FROM signal_observations
                WHERE side='BUY' AND blocked=0
                """
            ).fetchall()
            existing = {
                (str(observation_id), int(horizon))
                for observation_id, horizon in conn.execute(
                    """
                    SELECT o.observation_id, o.horizon_hours
                    FROM forward_outcomes_v2 o
                    JOIN signal_observations s
                      ON s.observation_id=o.observation_id
                    WHERE s.side='BUY' AND s.blocked=0
                    """
                ).fetchall()
            }
        finally:
            conn.close()

        grace = self._execution_grace_hours()
        pending_by_horizon: dict[int, int] = {}
        now = _now()
        for horizon in self._all_horizons():
            pending = 0
            maturity_age = timedelta(hours=float(horizon) + grace)
            for observation_id, observed_at in observations:
                observed = _parse_time(observed_at)
                if observed is None:
                    continue
                if now - observed.astimezone(UTC) < maturity_age:
                    continue
                if (str(observation_id), int(horizon)) not in existing:
                    pending += 1
            pending_by_horizon[int(horizon)] = pending

        span_hours = 0.0
        first_dt = _parse_time(primary_first)
        last_dt = _parse_time(primary_last)
        if first_dt is not None and last_dt is not None:
            span_hours = max(
                0.0,
                (
                    last_dt.astimezone(UTC) - first_dt.astimezone(UTC)
                ).total_seconds()
                / 3600.0,
            )

        return {
            "status": "READY",
            "cycles": cycles,
            "signals": signals,
            "blocked_signals": blocked,
            "unblocked_buy_observations": unblocked_buy,
            "unblocked_buy_markets": unblocked_buy_markets,
            "first_observed_at": first_observed_at,
            "latest_observed_at": latest_observed_at,
            "latest_observation_age_hours": _age_hours(latest_observed_at),
            "outcomes": outcomes,
            "outcomes_by_horizon": outcomes_by_horizon,
            "markets_by_horizon": markets_by_horizon,
            "pending_by_horizon": pending_by_horizon,
            "primary_horizon_hours": primary,
            "primary_horizon_outcomes": primary_count,
            "primary_horizon_markets": primary_markets,
            "primary_horizon_span_hours": span_hours,
            "primary_horizon_mean_return_bps": primary_mean,
            "primary_horizon_positive_fraction": primary_positive,
            "latest_primary_matured_at": latest_matured_at,
            "latest_primary_maturation_age_hours": _age_hours(
                latest_matured_at
            ),
            "complete_required_horizons": list(required),
            "complete_required_horizon_observations": complete_required,
        }

    def _research_stage(self, snapshot: dict[str, Any]) -> str:
        if snapshot.get("status") != "READY":
            return "NO_EVIDENCE"
        primary = int(snapshot.get("primary_horizon_outcomes") or 0)
        discovery = int(
            self.strategy_cfg.get("minimum_discovery_observations", 80)
        )
        if primary < discovery:
            return "COLLECTING"
        complete = int(
            snapshot.get("complete_required_horizon_observations") or 0
        )
        if complete < self._optimization_minimum():
            return "DISCOVERY_READY"
        return "OPTIMIZATION_READY"

    def _canary_readiness(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {
                "status": "NO_EVIDENCE",
                "ready": False,
                "automatic_live_authority": False,
            }
        ledger = self._ledger()
        try:
            cfg = dict(
                self.forward_cfg.get("canary_readiness", {}) or {}
            )
            payload = ledger.canary_readiness(
                primary_horizon_hours=int(
                    cfg.get("primary_horizon_hours", 4)
                ),
                minimum_unblocked_buy_outcomes=int(
                    cfg.get("minimum_unblocked_buy_outcomes", 30)
                ),
                minimum_distinct_markets=int(
                    cfg.get("minimum_distinct_markets", 5)
                ),
                minimum_observation_span_hours=float(
                    cfg.get("minimum_observation_span_hours", 72)
                ),
                minimum_mean_return_bps=float(
                    cfg.get("minimum_mean_return_bps", 0.0)
                ),
                minimum_positive_return_rate=float(
                    cfg.get("minimum_positive_return_rate", 0.50)
                ),
            )
            return {
                **dict(payload),
                "automatic_live_authority": False,
            }
        except Exception as exc:  # noqa: BLE001 - diagnostics must fail closed
            return {
                "status": "ERROR",
                "ready": False,
                "error": f"{type(exc).__name__}:{str(exc)[:400]}",
                "automatic_live_authority": False,
            }
        finally:
            ledger.close()

    def snapshot(self, *, persist: bool = True) -> dict[str, Any]:
        raw = self._database_snapshot()
        payload = {
            "schema_version": self.SCHEMA,
            "generated_at": _now().isoformat(),
            **raw,
            "research_stage": self._research_stage(raw),
            "minimum_discovery_observations": int(
                self.strategy_cfg.get(
                    "minimum_discovery_observations",
                    80,
                )
            ),
            "minimum_complete_horizon_observations_for_optimization": (
                self._optimization_minimum()
            ),
            "canary_readiness": self._canary_readiness(),
            "causal_outcomes_only": True,
            "closed_candles_only": True,
            "synthetic_sample_multiplication": False,
            "threshold_relaxation": False,
            "automatic_live_authority": False,
            "automatic_live_promotion": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        if persist:
            self._write_json(self.latest_path, payload)
        return payload

    def _rotating_markets(
        self,
        markets: tuple[str, ...],
    ) -> list[str]:
        if not markets:
            return []
        maximum = max(
            1,
            int(self.cfg.get("maximum_markets_per_maturation_cycle", 40)),
        )
        ordered = list(markets)
        if len(ordered) <= maximum:
            return ordered
        cursor = int(self.state.get("market_cursor", 0)) % len(ordered)
        selected = [
            ordered[(cursor + offset) % len(ordered)]
            for offset in range(maximum)
        ]
        self.state["market_cursor"] = (cursor + maximum) % len(ordered)
        return selected

    def mature(self) -> dict[str, Any]:
        if not bool(self.cfg.get("enabled", True)):
            return {
                **self.snapshot(),
                "maturation_status": "DISABLED",
            }

        ledger = self._ledger()
        try:
            markets = self._rotating_markets(
                ledger.observation_markets()
            )
            if not markets:
                result: dict[str, Any] = {
                    "status": "NO_OBSERVATIONS",
                    "outcomes_inserted": 0,
                }
            else:
                frames = self.crypto.ohlcv_many(
                    markets,
                    str(
                        self.forward_cfg.get(
                            "execution_timeframe",
                            "15m",
                        )
                    ),
                    persist=False,
                    concurrency=max(
                        1,
                        int(self.cfg.get("fetch_concurrency", 4)),
                    ),
                )
                result = ledger.mature_from_frames(
                    frames,
                    horizons_hours=self._all_horizons(),
                    execution_timeframe=str(
                        self.forward_cfg.get(
                            "execution_timeframe",
                            "15m",
                        )
                    ),
                )
                result["status"] = "COMPLETE"
                result["markets_requested"] = markets
        except Exception as exc:  # noqa: BLE001 - persistent daemon fails closed
            result = {
                "status": "ERROR",
                "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                "outcomes_inserted": 0,
            }
        finally:
            ledger.close()

        now = _now().isoformat()
        self.state["last_maturation_at"] = now
        self.state["last_maturation_status"] = result.get("status")
        self._write_json(self.state_path, self.state)
        payload = {
            **self.snapshot(persist=False),
            "maturation_status": result.get("status"),
            "maturation": result,
        }
        self._write_json(self.latest_path, payload)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, sort_keys=True, default=str) + "\n"
            )
        return payload
