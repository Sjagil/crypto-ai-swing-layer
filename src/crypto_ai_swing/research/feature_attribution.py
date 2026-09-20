
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


class Round44FeatureAttribution:
    SCHEMA = "round44_feature_attribution_v2"
    DEFAULT_HORIZONS = (1, 4, 24, 72, 168)

    def __init__(self, settings) -> None:
        self.settings = settings
        self.root = settings.project_root / "output/crypto_ai_swing/round44"
        self.latest = self.root / "feature_attribution.json"

    @staticmethod
    def _vector(context_raw: str) -> dict[str, float]:
        try:
            context = json.loads(context_raw)
        except Exception:
            return {}
        raw = (context.get("round44") or {}).get("learning_vector") or {}
        result: dict[str, float] = {}
        for key, value in raw.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                result[str(key)] = number
        return result

    @classmethod
    def _load_horizon(
        cls, database_path: Path, horizon_hours: int
    ) -> list[dict[str, Any]]:
        conn = sqlite3.connect(database_path)
        try:
            rows = conn.execute(
                """
                SELECT s.observation_id,s.observed_at,s.market,s.side,s.blocked,
                       s.context,o.return_bps,o.mfe_bps,o.mae_bps
                FROM signal_observations s
                JOIN forward_outcomes_v2 o ON o.observation_id=s.observation_id
                WHERE o.horizon_hours=?
                ORDER BY s.observed_at,s.observation_id
                """,
                (int(horizon_hours),),
            ).fetchall()
        finally:
            conn.close()
        material: list[dict[str, Any]] = []
        for observation_id, observed_at, market, side, blocked, context, ret, mfe, mae in rows:
            vector = cls._vector(context)
            if not vector:
                continue
            material.append(
                {
                    "observation_id": str(observation_id),
                    "observed_at": str(observed_at),
                    "market": str(market),
                    "side": str(side),
                    "blocked": bool(blocked),
                    "return_bps": float(ret),
                    "mfe_bps": float(mfe),
                    "mae_bps": float(mae),
                    "vector": vector,
                }
            )
        return material

    @staticmethod
    def _one_horizon(
        material: list[dict[str, Any]],
        *,
        horizon_hours: int,
        minimum_observations: int,
        maximum_features: int,
    ) -> dict[str, Any]:
        if len(material) < minimum_observations:
            return {
                "status": "COLLECTING",
                "horizon_hours": horizon_hours,
                "observations": len(material),
                "minimum_observations": minimum_observations,
                "market_count": len({row["market"] for row in material}),
                "shortlist": [],
                "shortlist_features": [],
            }
        frame = pd.DataFrame(
            [
                {
                    **row["vector"],
                    "_return_bps": row["return_bps"],
                    "_market": row["market"],
                    "_side": row["side"],
                    "_blocked": row["blocked"],
                    "_observed_at": row["observed_at"],
                }
                for row in material
            ]
        )
        target = pd.to_numeric(frame["_return_bps"], errors="coerce")
        feature_names = [name for name in frame.columns if not name.startswith("_")]
        results: list[dict[str, Any]] = []
        for feature in feature_names:
            values = pd.to_numeric(frame[feature], errors="coerce")
            valid = values.notna() & target.notna()
            count = int(valid.sum())
            if count < max(30, minimum_observations // 2):
                continue
            x = values[valid]
            y = target[valid]
            if x.nunique() < 3:
                continue
            rank_corr_raw = x.rank().corr(y.rank())
            if pd.isna(rank_corr_raw):
                continue
            rank_corr = float(rank_corr_raw)
            q25, q75 = float(x.quantile(0.25)), float(x.quantile(0.75))
            low = y[x <= q25]
            high = y[x >= q75]
            spread = (
                float(high.mean() - low.mean())
                if len(low) >= 5 and len(high) >= 5
                else None
            )
            market_effects: list[float] = []
            for market in sorted(frame.loc[valid, "_market"].unique()):
                mask = valid & (frame["_market"] == market)
                xm, ym = values[mask], target[mask]
                if len(xm) < 8 or xm.nunique() < 3:
                    continue
                corr = xm.rank().corr(ym.rank())
                if pd.notna(corr):
                    market_effects.append(float(corr))
            sign_consistency = (
                float(np.mean(np.sign(market_effects) == np.sign(rank_corr)))
                if market_effects and rank_corr != 0
                else None
            )
            score = abs(rank_corr) * (0.5 + 0.5 * (sign_consistency or 0.0))
            results.append(
                {
                    "feature": feature,
                    "observations": count,
                    "rank_correlation": rank_corr,
                    "high_minus_low_return_bps": spread,
                    "market_effect_count": len(market_effects),
                    "market_sign_consistency": sign_consistency,
                    "score": score,
                }
            )
        results.sort(key=lambda row: row["score"], reverse=True)
        shortlist = [
            row
            for row in results
            if abs(float(row["rank_correlation"])) >= 0.03
            and (
                row["market_sign_consistency"] is None
                or float(row["market_sign_consistency"]) >= 0.50
            )
        ][:maximum_features]
        return {
            "status": "READY",
            "horizon_hours": horizon_hours,
            "observations": len(material),
            "market_count": len({row["market"] for row in material}),
            "buy_observations": sum(row["side"] == "BUY" for row in material),
            "no_trade_observations": sum(row["side"] == "NO_TRADE" for row in material),
            "unblocked_observations": sum(not row["blocked"] for row in material),
            "feature_candidates": len(results),
            "shortlist": shortlist,
            "shortlist_features": [row["feature"] for row in shortlist],
        }

    def evaluate(
        self,
        database_path: Path,
        *,
        horizons_hours: Iterable[int] = DEFAULT_HORIZONS,
        minimum_observations: int = 60,
        maximum_features: int = 80,
        minimum_horizon_support: int = 2,
    ) -> dict[str, Any]:
        horizons = tuple(sorted({int(h) for h in horizons_hours if int(h) > 0}))
        if not database_path.is_file():
            return self._write(
                {
                    "schema_version": self.SCHEMA,
                    "status": "NO_FORWARD_DATABASE",
                    "qualified": False,
                    "horizons_hours": list(horizons),
                }
            )
        by_horizon: dict[str, Any] = {}
        feature_rows: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for horizon in horizons:
            material = self._load_horizon(database_path, horizon)
            report = self._one_horizon(
                material,
                horizon_hours=horizon,
                minimum_observations=minimum_observations,
                maximum_features=maximum_features,
            )
            by_horizon[str(horizon)] = report
            for row in report.get("shortlist") or []:
                feature_rows.setdefault(str(row["feature"]), []).append((horizon, row))

        stable: list[dict[str, Any]] = []
        for feature, rows in feature_rows.items():
            signs = [np.sign(float(row["rank_correlation"])) for _, row in rows]
            nonzero = [sign for sign in signs if sign != 0]
            sign_consistency = (
                max(nonzero.count(1.0), nonzero.count(-1.0)) / len(nonzero)
                if nonzero
                else 0.0
            )
            mean_abs_corr = float(
                np.mean([abs(float(row["rank_correlation"])) for _, row in rows])
            )
            support = len(rows)
            if support < minimum_horizon_support or sign_consistency < 0.75:
                continue
            stable.append(
                {
                    "feature": feature,
                    "horizon_support": support,
                    "horizons": [h for h, _ in rows],
                    "mean_absolute_rank_correlation": mean_abs_corr,
                    "cross_horizon_sign_consistency": sign_consistency,
                    "score": mean_abs_corr * (1.0 + 0.15 * (support - 1)),
                }
            )
        stable.sort(key=lambda row: float(row["score"]), reverse=True)
        stable = stable[:maximum_features]
        primary = dict(by_horizon.get("4") or {})
        ready_horizons = [
            int(h) for h, row in by_horizon.items() if row.get("status") == "READY"
        ]
        long_ready = all(
            dict(by_horizon.get(str(h)) or {}).get("status") == "READY"
            for h in (24, 72, 168)
            if h in horizons
        )
        payload = {
            "schema_version": self.SCHEMA,
            "status": "READY" if ready_horizons else "COLLECTING",
            "qualified": bool(stable) and 4 in ready_horizons,
            "generated_at": datetime.now(UTC).isoformat(),
            "horizons_hours": list(horizons),
            "ready_horizons": ready_horizons,
            "long_horizons_ready": long_ready,
            "by_horizon": by_horizon,
            "primary_horizon_hours": 4,
            "primary": primary,
            "stable_shortlist": stable,
            "stable_shortlist_features": [row["feature"] for row in stable],
            "selection_is_research_only": True,
            "selection_uses_matured_outcomes_only": True,
            "selection_is_cross_horizon": True,
            "automatic_model_application": False,
            "automatic_live_authority": False,
        }
        return self._write(payload)

    def _write(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        selected = {"generated_at": datetime.now(UTC).isoformat(), **payload}
        self.latest.write_text(
            json.dumps(selected, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return selected
