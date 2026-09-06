from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from crypto_ai_swing.agents.component_features import (
    COMPONENTS,
    extract_components,
    extract_descriptors,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _seed(label: str) -> int:
    return int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)


def _finite(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def bayesian_mean_posterior(
    values_bps: list[float] | np.ndarray,
    *,
    draws: int = 5000,
    seed: int = 2401,
) -> dict[str, Any]:
    values = _finite(values_bps)
    n = int(values.size)
    if n == 0:
        return {"status": "NO_DATA", "observations": 0}
    rng = np.random.default_rng(seed)
    wins = int(np.sum(values > 0.0))
    positive = rng.beta(1 + wins, 1 + n - wins, size=draws)
    if n == 1:
        means = np.full(draws, values[0])
    else:
        weights = rng.dirichlet(np.ones(n), size=draws)
        means = weights @ values
    return {
        "status": "READY",
        "observations": n,
        "wins": wins,
        "sample_mean_bps": float(values.mean()),
        "sample_median_bps": float(np.median(values)),
        "positive_fraction": float(np.mean(values > 0.0)),
        "positive_rate_posterior_mean": float(positive.mean()),
        "mean_p05_bps": float(np.quantile(means, 0.05)),
        "mean_p50_bps": float(np.quantile(means, 0.50)),
        "mean_p95_bps": float(np.quantile(means, 0.95)),
        "probability_mean_positive": float(np.mean(means > 0.0)),
    }


def bootstrap_difference_probability(
    high: np.ndarray,
    low: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    high = _finite(high)
    low = _finite(low)
    if len(high) < 2 or len(low) < 2:
        return {"status": "INSUFFICIENT", "probability_high_better": None}
    rng = np.random.default_rng(seed)
    high_idx = rng.integers(0, len(high), size=(draws, len(high)))
    low_idx = rng.integers(0, len(low), size=(draws, len(low)))
    delta = high[high_idx].mean(axis=1) - low[low_idx].mean(axis=1)
    return {
        "status": "READY",
        "observations_high": int(len(high)),
        "observations_low": int(len(low)),
        "mean_difference_bps": float(high.mean() - low.mean()),
        "delta_p05_bps": float(np.quantile(delta, 0.05)),
        "delta_p50_bps": float(np.quantile(delta, 0.50)),
        "delta_p95_bps": float(np.quantile(delta, 0.95)),
        "probability_high_better": float(np.mean(delta > 0.0)),
    }


def _summary(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = _finite(values)
    if len(arr) == 0:
        return {"observations": 0}
    winners = arr[arr > 0]
    losers = arr[arr <= 0]
    q05 = float(np.quantile(arr, 0.05))
    tail = arr[arr <= q05]
    gross_profit = float(winners.sum()) if len(winners) else 0.0
    gross_loss = abs(float(losers.sum())) if len(losers) else 0.0
    return {
        "observations": int(len(arr)),
        "mean_bps": float(arr.mean()),
        "median_bps": float(np.median(arr)),
        "std_bps": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "positive_fraction": float(np.mean(arr > 0.0)),
        "q05_bps": q05,
        "q25_bps": float(np.quantile(arr, 0.25)),
        "q75_bps": float(np.quantile(arr, 0.75)),
        "q95_bps": float(np.quantile(arr, 0.95)),
        "cvar05_bps": float(tail.mean()) if len(tail) else q05,
        "mean_winner_bps": float(winners.mean()) if len(winners) else None,
        "mean_loser_bps": float(losers.mean()) if len(losers) else None,
        "payoff_ratio": (
            float(winners.mean() / abs(losers.mean()))
            if len(winners) and len(losers) and abs(float(losers.mean())) > 0
            else None
        ),
        "profit_factor": (
            gross_profit / gross_loss if gross_loss > 0 else None
        ),
    }


class PerformanceAttributionEngine:
    """Cost-aware prospective and paper performance attribution.

    Prospective attribution uses only matured, causal forward outcomes. Paper
    realized attribution uses exact context embedded in future simulated BUY
    intents when available. It never changes live authority.
    """

    SCHEMA = "crypto_ai_swing_performance_attribution_v1"

    def __init__(self, settings, *, mode: str = "paper") -> None:
        self.settings = settings
        self.mode = str(mode).lower()
        cfg = dict((getattr(settings, "autonomy", {}) or {}).get(
            "performance_attribution", {}
        ) or {})
        self.horizon_hours = int(cfg.get("horizon_hours", 4))
        self.minimum_observations = int(cfg.get("minimum_observations", 30))
        self.bootstrap_draws = int(cfg.get("bootstrap_draws", 3000))
        self.root = Path(settings.project_root) / "output/crypto_ai_swing/research/attribution"
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.root / "latest.json"
        self.history_path = self.root / "history.jsonl"

    def forward_database_path(self) -> Path:
        cfg = dict((getattr(self.settings, "autonomy", {}) or {}).get(
            "forward_evidence", {}
        ) or {})
        return Path(self.settings.project_root) / cfg.get(
            "path", "output/crypto_ai_swing/forward/forward.sqlite"
        )

    def _base_costs(self) -> tuple[float, float, float, float]:
        execution = dict(getattr(self.settings, "execution", {}) or {})
        costs = dict(execution.get("costs", {}) or {})
        fee = float(costs.get("fee_bps_per_side", 25.0))
        slippage = float(costs.get("base_slippage_bps", 2.0))
        agents = dict(getattr(self.settings, "agents", {}) or {})
        edge = dict(agents.get("edge_manager", {}) or {})
        fallback_spread = float(edge.get("research_round_trip_spread_bps", 8.0))
        stress = float(edge.get("stressed_extra_cost_bps", 25.0))
        return fee, slippage, fallback_spread, stress

    def _row_costs(self, context: Mapping[str, Any]) -> tuple[float, float]:
        fee, slippage, fallback_spread, stress = self._base_costs()
        screen = dict(context.get("universe_screen") or {})
        challenger = dict(context.get("mtf_challenger") or {})
        diagnostics = dict(challenger.get("diagnostics") or {})
        raw_spread = screen.get("universe_spread_bps", diagnostics.get("spread_bps"))
        try:
            spread = float(raw_spread)
            if not np.isfinite(spread) or spread < 0:
                spread = fallback_spread
        except (TypeError, ValueError):
            spread = fallback_spread
        normal = 2.0 * (fee + slippage) + spread
        return normal, normal + stress

    def load_forward_rows(self) -> list[dict[str, Any]]:
        path = self.forward_database_path()
        if not path.is_file():
            return []
        conn = sqlite3.connect(path)
        try:
            rows = conn.execute(
                """
                SELECT s.observation_id,s.observed_at,s.market,s.score,
                       s.edge_bps,s.edge_source,s.context,
                       o.entry_bar_ts,o.return_bps,o.mfe_bps,o.mae_bps
                FROM signal_observations s
                JOIN forward_outcomes_v2 o
                  ON o.observation_id=s.observation_id
                WHERE s.side='BUY'
                  AND s.blocked=0
                  AND o.horizon_hours=?
                ORDER BY s.observed_at,s.market,s.observation_id
                """,
                (self.horizon_hours,),
            ).fetchall()
        finally:
            conn.close()

        output: list[dict[str, Any]] = []
        for row in rows:
            try:
                context = json.loads(row[6] or "{}")
                gross = float(row[8])
                mfe = float(row[9])
                mae = float(row[10])
            except Exception:
                continue
            if not isinstance(context, dict) or not np.isfinite(gross):
                continue
            normal_cost, stressed_cost = self._row_costs(context)
            output.append({
                "observation_id": str(row[0]),
                "observed_at": str(row[1]),
                "market": str(row[2]),
                "signal_score": float(row[3]),
                "edge_bps": float(row[4]),
                "edge_source": str(row[5]),
                "context": context,
                "entry_bar_ts": str(row[7]),
                "gross_return_bps": gross,
                "mfe_bps": mfe,
                "mae_bps": mae,
                "normal_cost_bps": normal_cost,
                "stressed_cost_bps": stressed_cost,
                "normal_net_bps": gross - normal_cost,
                "stressed_net_bps": gross - stressed_cost,
                "components": extract_components(context),
                "descriptors": extract_descriptors(context),
            })
        return output

    @staticmethod
    def _category(rows: list[dict[str, Any]], key_fn, *, minimum: int = 2) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            value = key_fn(row)
            if value is None or str(value) in {"", "None"}:
                continue
            grouped[str(value)].append(row)
        result = {}
        for key, selected in sorted(grouped.items()):
            if len(selected) < minimum:
                continue
            result[key] = {
                "normal_net": _summary([r["normal_net_bps"] for r in selected]),
                "gross": _summary([r["gross_return_bps"] for r in selected]),
                "mfe_mean_bps": float(np.mean([r["mfe_bps"] for r in selected])),
                "mae_mean_bps": float(np.mean([r["mae_bps"] for r in selected])),
            }
        return result

    def _component_attribution(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        returns = pd.Series([r["normal_net_bps"] for r in rows], dtype=float)
        result: dict[str, Any] = {}
        for name in COMPONENTS:
            values = pd.Series(
                [r["components"].get(name) for r in rows], dtype="float64"
            )
            mask = values.notna() & returns.notna()
            count = int(mask.sum())
            if count < 4:
                result[name] = {
                    "status": "INSUFFICIENT",
                    "observations": count,
                    "coverage_fraction": count / max(1, len(rows)),
                }
                continue
            v = values[mask].to_numpy(dtype=float)
            y = returns[mask].to_numpy(dtype=float)
            corr = pd.Series(v).corr(pd.Series(y), method="spearman")
            median = float(np.median(v))
            high = y[v >= median]
            low = y[v < median]
            delta = bootstrap_difference_probability(
                high,
                low,
                draws=self.bootstrap_draws,
                seed=_seed(f"component:{name}"),
            )
            result[name] = {
                "status": "READY",
                "observations": count,
                "coverage_fraction": count / max(1, len(rows)),
                "spearman_ic": (
                    float(corr) if corr is not None and np.isfinite(corr) else 0.0
                ),
                "median_split": median,
                "high": _summary(high),
                "low": _summary(low),
                "high_mean_posterior": bayesian_mean_posterior(
                    high,
                    draws=max(1000, self.bootstrap_draws),
                    seed=_seed(f"posterior:{name}"),
                ),
                "high_vs_low": delta,
            }
        return result

    def _paper_path(self) -> Path:
        public_mode = "canary" if self.mode == "live" else self.mode
        return (
            Path(self.settings.project_root)
            / "output/crypto_ai_swing/modes"
            / public_mode
            / "proactive/paper_ledger.sqlite"
        )

    def _paper_attribution(self) -> dict[str, Any]:
        path = self._paper_path()
        if not path.is_file():
            return {"status": "NOT_AVAILABLE", "events": 0}
        conn = sqlite3.connect(path)
        try:
            rows = conn.execute(
                """
                SELECT event_id,created_at,side,market,quantity,price,
                       gross_eur,fee_eur,realized_pnl_eur,payload
                FROM events ORDER BY rowid
                """
            ).fetchall()
        finally:
            conn.close()
        open_buys: dict[str, list[dict[str, Any]]] = defaultdict(list)
        closed: list[dict[str, Any]] = []
        total_fees = 0.0
        total_realized = 0.0
        for raw in rows:
            try:
                payload = json.loads(raw[9] or "{}")
            except Exception:
                payload = {}
            event = {
                "event_id": str(raw[0]),
                "created_at": str(raw[1]),
                "side": str(raw[2]),
                "market": str(raw[3]),
                "quantity": float(raw[4]),
                "price": float(raw[5]),
                "gross_eur": float(raw[6]),
                "fee_eur": float(raw[7]),
                "realized_pnl_eur": float(raw[8]),
                "payload": payload if isinstance(payload, dict) else {},
            }
            total_fees += event["fee_eur"]
            if event["side"] == "BUY":
                open_buys[event["market"]].append(event)
            elif event["side"] == "SELL":
                total_realized += event["realized_pnl_eur"]
                buy = open_buys[event["market"]].pop(0) if open_buys[event["market"]] else None
                context = None
                if buy:
                    metadata = dict(buy["payload"].get("intent_metadata") or {})
                    candidate = metadata.get("crypto_repo_context")
                    if isinstance(candidate, dict):
                        context = candidate
                closed.append({
                    "market": event["market"],
                    "buy_event_id": buy.get("event_id") if buy else None,
                    "sell_event_id": event["event_id"],
                    "realized_pnl_eur": event["realized_pnl_eur"],
                    "entry_price": buy.get("price") if buy else None,
                    "exit_price": event["price"],
                    "context": context,
                    "attributed": context is not None,
                    "components": extract_components(context or {}),
                    "descriptors": extract_descriptors(context or {}),
                })
        attributed = [r for r in closed if r["attributed"]]
        return {
            "status": "READY",
            "events": len(rows),
            "closed_trades": len(closed),
            "attributed_closed_trades": len(attributed),
            "unattributed_closed_trades": len(closed) - len(attributed),
            "realized_pnl_eur": total_realized,
            "fees_eur": total_fees,
            "open_buy_events": sum(len(v) for v in open_buys.values()),
            "attribution_ready_for_future_intents": True,
        }

    def refresh(self) -> dict[str, Any]:
        rows = self.load_forward_rows()
        normal = [r["normal_net_bps"] for r in rows]
        stressed = [r["stressed_net_bps"] for r in rows]
        gross = [r["gross_return_bps"] for r in rows]
        mfe = _finite([r["mfe_bps"] for r in rows])
        mae = _finite([r["mae_bps"] for r in rows])
        mean_mfe = float(mfe.mean()) if len(mfe) else None
        mean_mae = float(mae.mean()) if len(mae) else None
        status = "READY" if len(rows) >= self.minimum_observations else "COLLECTING"
        mean_normal = float(np.mean(normal)) if len(normal) else None
        positive_fraction = float(np.mean(np.asarray(normal) > 0.0)) if len(normal) else None
        loss_asymmetry = (
            abs(float(np.mean([x for x in normal if x <= 0.0])))
            / max(1e-9, float(np.mean([x for x in normal if x > 0.0])))
            if any(x <= 0.0 for x in normal) and any(x > 0.0 for x in normal)
            else None
        )
        research_priorities = []
        if mean_normal is not None and mean_normal <= 0.0:
            research_priorities.append("NET_EDGE_NON_POSITIVE")
        if loss_asymmetry is not None and loss_asymmetry > 1.0:
            research_priorities.append("LOSS_SEVERITY_EXCEEDS_WINNER_MAGNITUDE")
        if mean_mfe is not None and mean_mae is not None and mean_mfe < abs(mean_mae):
            research_priorities.append("ADVERSE_EXCURSION_EXCEEDS_FAVORABLE_EXCURSION")
        if positive_fraction is not None and positive_fraction >= 0.5 and mean_normal is not None and mean_normal <= 0.0:
            research_priorities.append("WIN_RATE_MASKED_BY_LOSS_ASYMMETRY")
        payload = {
            "schema_version": self.SCHEMA,
            "generated_at": _now(),
            "status": status,
            "horizon_hours": self.horizon_hours,
            "observations": len(rows),
            "minimum_observations": self.minimum_observations,
            "overall": {
                "gross": _summary(gross),
                "normal_net": _summary(normal),
                "stressed_net": _summary(stressed),
                "bayesian_normal_net": bayesian_mean_posterior(
                    normal,
                    draws=max(1000, self.bootstrap_draws),
                    seed=2401,
                ),
                "mean_mfe_bps": mean_mfe,
                "mean_mae_bps": mean_mae,
                "mfe_to_abs_mae_ratio": (
                    mean_mfe / abs(mean_mae)
                    if mean_mfe is not None and mean_mae not in {None, 0.0}
                    else None
                ),
            },
            "by_market": self._category(rows, lambda r: r["market"]),
            "by_strategy_family": self._category(
                rows, lambda r: r["descriptors"].get("strategy_family")
            ),
            "by_breakout_state": self._category(
                rows, lambda r: r["descriptors"].get("breakout_state")
            ),
            "by_edge_source": self._category(rows, lambda r: r["edge_source"]),
            "components": self._component_attribution(rows),
            "diagnostics": {
                "mean_normal_net_bps": mean_normal,
                "positive_fraction_normal_net": positive_fraction,
                "loss_to_winner_magnitude_ratio": loss_asymmetry,
                "research_priorities": research_priorities,
            },
            "paper": self._paper_attribution(),
            "authority": "RESEARCH_ONLY",
            "shadow_decision_influence": False,
            "live_decision_influence": False,
            "automatic_live_promotion": False,
            "orders_submitted": 0,
        }
        self.latest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        return payload

    def status(self) -> dict[str, Any]:
        try:
            value = json.loads(self.latest_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {"status": "INVALID"}
        except Exception:
            return {"schema_version": self.SCHEMA, "status": "NOT_BUILT"}
