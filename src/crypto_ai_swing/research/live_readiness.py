
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class Round44LiveReadiness:
    """Evidence ladder only. Canonical/manual execution authority stays external."""

    SCHEMA = "round44_live_readiness_v2"
    STAGES = (
        "RESEARCH",
        "SHADOW_QUALIFIED",
        "PAPER_QUALIFIED",
        "CANARY_CANDIDATE",
        "LIVE_CANDIDATE",
    )
    HORIZONS = (1, 4, 24, 72, 168)

    def __init__(self, settings) -> None:
        self.settings = settings
        self.root = settings.project_root / "output/crypto_ai_swing/round44"
        self.latest = self.root / "live_readiness.json"

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    @classmethod
    def _forward(cls, database_path: Path) -> dict[str, Any]:
        empty = {
            "by_horizon": {str(h): 0 for h in cls.HORIZONS},
            "markets_by_horizon": {str(h): 0 for h in cls.HORIZONS},
            "unblocked_buy_by_horizon": {str(h): 0 for h in cls.HORIZONS},
            "primary_span_hours": 0.0,
            "complete_all_horizons": 0,
            "complete_long_horizons": 0,
        }
        if not database_path.is_file():
            return empty
        conn = sqlite3.connect(database_path)
        try:
            by_horizon = {
                int(h): int(c)
                for h, c in conn.execute(
                    "SELECT horizon_hours,COUNT(*) FROM forward_outcomes_v2 "
                    "GROUP BY horizon_hours"
                ).fetchall()
            }
            markets = {
                int(h): int(c)
                for h, c in conn.execute(
                    """
                    SELECT o.horizon_hours,COUNT(DISTINCT s.market)
                    FROM forward_outcomes_v2 o
                    JOIN signal_observations s ON s.observation_id=o.observation_id
                    GROUP BY o.horizon_hours
                    """
                ).fetchall()
            }
            unblocked_buy = {
                int(h): int(c)
                for h, c in conn.execute(
                    """
                    SELECT o.horizon_hours,COUNT(*)
                    FROM forward_outcomes_v2 o
                    JOIN signal_observations s ON s.observation_id=o.observation_id
                    WHERE s.side='BUY' AND s.blocked=0
                    GROUP BY o.horizon_hours
                    """
                ).fetchall()
            }
            primary = conn.execute(
                """
                SELECT MIN(s.observed_at),MAX(s.observed_at)
                FROM signal_observations s
                JOIN forward_outcomes_v2 o ON o.observation_id=s.observation_id
                WHERE o.horizon_hours=4 AND s.side='BUY' AND s.blocked=0
                """
            ).fetchone()
            complete_all = conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT observation_id
                    FROM forward_outcomes_v2
                    WHERE horizon_hours IN (1,4,24,72,168)
                    GROUP BY observation_id
                    HAVING COUNT(DISTINCT horizon_hours)=5
                )
                """
            ).fetchone()[0]
            complete_long = conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT observation_id
                    FROM forward_outcomes_v2
                    WHERE horizon_hours IN (24,72,168)
                    GROUP BY observation_id
                    HAVING COUNT(DISTINCT horizon_hours)=3
                )
                """
            ).fetchone()[0]
        finally:
            conn.close()
        span = 0.0
        if primary and primary[0] and primary[1]:
            try:
                start = datetime.fromisoformat(str(primary[0]).replace("Z", "+00:00"))
                end = datetime.fromisoformat(str(primary[1]).replace("Z", "+00:00"))
                span = max(0.0, (end - start).total_seconds() / 3600.0)
            except Exception:
                pass
        return {
            "by_horizon": {str(h): int(by_horizon.get(h, 0)) for h in cls.HORIZONS},
            "markets_by_horizon": {str(h): int(markets.get(h, 0)) for h in cls.HORIZONS},
            "unblocked_buy_by_horizon": {
                str(h): int(unblocked_buy.get(h, 0)) for h in cls.HORIZONS
            },
            "primary_span_hours": span,
            "complete_all_horizons": int(complete_all or 0),
            "complete_long_horizons": int(complete_long or 0),
        }

    def evaluate(self, forward_database_path: Path) -> dict[str, Any]:
        proactive = self._read(
            self.settings.project_root
            / "output/crypto_ai_swing/modes/shadow/proactive/latest.json"
        )
        round44 = self._read(self.root / "latest.json")
        attribution = self._read(self.root / "feature_attribution.json")
        context_pointer = self._read(
            self.settings.project_root
            / "output/crypto_ai_swing/agents/round44_context/latest.pointer.json"
        )
        supervised_pointer = self._read(
            self.settings.project_root / "output/crypto_ai_swing/agents/latest.pointer.json"
        )
        rl_pointer = self._read(
            self.settings.project_root
            / "output/crypto_ai_swing/agents/rl/latest.pointer.json"
        )
        manager = self._read(
            self.settings.project_root
            / "output/crypto_ai_swing/agents/manager/state.json"
        )
        economics = self._read(
            self.settings.project_root
            / "output/crypto_ai_swing/autonomy/round43/paper_economics/latest.json"
        )
        if not economics:
            economics = self._read(
                self.settings.project_root
                / "output/crypto_ai_swing/autonomy/round43/paper_economics.json"
            )
        drift = self._read(
            self.settings.project_root
            / "output/crypto_ai_swing/autonomy/round43/drift/latest.json"
        )
        forward = self._forward(forward_database_path)
        summary = dict(round44.get("summary") or {})
        econ = dict(economics.get("overall") or economics)
        attribution_primary = dict(attribution.get("primary") or {})

        shadow_gates = {
            "EXACT_25_MARKETS": int(summary.get("requested_markets") or 0) == 25,
            "FULL_25_SIX_TIMEFRAME_TECHNICAL": summary.get(
                "all_25_six_timeframe_technical_ready"
            )
            is True,
            "FULL_25_L1_L2_SNAPSHOT": summary.get("all_25_l1_l2_snapshot_ready") is True,
            "L3_CONTRACT_EXPLICIT": summary.get("l3")
            == "L3_UNSUPPORTED_BY_EXECUTION_VENUE",
            "FORWARD_4H_UNBLOCKED_BUY_30": int(
                forward["unblocked_buy_by_horizon"].get("4", 0)
            )
            >= 30,
            "FORWARD_4H_MARKETS_5": int(forward["markets_by_horizon"].get("4", 0)) >= 5,
            "FORWARD_SPAN_72H": float(forward["primary_span_hours"]) >= 72.0,
            "ATTRIBUTION_4H_READY": attribution_primary.get("status") == "READY",
        }
        paper_gates = {
            **shadow_gates,
            "PAPER_CLOSED_TRADES_30": int(econ.get("closed_trades") or 0) >= 30,
            "PAPER_EXPECTANCY_POSITIVE": float(econ.get("expectancy_bps") or -1e9) > 0.0,
            "PAPER_PROFIT_FACTOR_1_15": float(econ.get("profit_factor") or 0.0) >= 1.15,
            "FORWARD_24H_30": int(forward["by_horizon"].get("24", 0)) >= 30,
            "FORWARD_72H_30": int(forward["by_horizon"].get("72", 0)) >= 30,
            "FORWARD_168H_30": int(forward["by_horizon"].get("168", 0)) >= 30,
            "COMPLETE_LONG_HORIZONS_30": int(forward["complete_long_horizons"]) >= 30,
            "ATTRIBUTION_LONG_HORIZONS_READY": attribution.get("long_horizons_ready") is True,
        }
        context_qualified = context_pointer.get("qualified") is True
        canary = dict(proactive.get("prospective_canary_readiness") or {})
        canary_gates = {
            **paper_gates,
            "MDPRO_FULL_25_SEQUENCE_VALID": summary.get("all_25_mdpro_stream_ready") is True,
            "PROSPECTIVE_CONTEXT_OOS_QUALIFIED": context_qualified,
            "SUPERVISED_ARTIFACT_PRESENT": bool(supervised_pointer),
            "PROSPECTIVE_CANARY_ELIGIBLE": canary.get("eligible") is True,
        }
        rl_enabled = bool(
            ((getattr(self.settings, "agents", {}) or {}).get("rl", {}) or {}).get(
                "enabled", False
            )
        )
        if rl_enabled:
            canary_gates["RL_ARTIFACT_PRESENT_WHEN_ENABLED"] = bool(rl_pointer)
        live_gates = {
            **canary_gates,
            "COMPLETE_ALL_1_4_24_72_168_30": int(forward["complete_all_horizons"]) >= 30,
            "AGENT_MANAGER_HEALTHY": str(manager.get("status") or "").upper()
            in {"HEALTHY", "READY"},
            "DRIFT_NOT_FAILED": str(drift.get("status") or "UNKNOWN").upper()
            not in {"FAILED", "BLOCKED"},
            "NO_AUTOMATIC_LIVE_PROMOTION": True,
            "CANONICAL_MANUAL_AUTHORITY_REQUIRED": True,
        }

        if all(live_gates.values()):
            stage = "LIVE_CANDIDATE"
        elif all(canary_gates.values()):
            stage = "CANARY_CANDIDATE"
        elif all(paper_gates.values()):
            stage = "PAPER_QUALIFIED"
        elif all(shadow_gates.values()):
            stage = "SHADOW_QUALIFIED"
        else:
            stage = "RESEARCH"

        gates = {
            "RESEARCH": {},
            "SHADOW_QUALIFIED": shadow_gates,
            "PAPER_QUALIFIED": paper_gates,
            "CANARY_CANDIDATE": canary_gates,
            "LIVE_CANDIDATE": live_gates,
        }
        index = self.STAGES.index(stage)
        next_stage = self.STAGES[min(index + 1, len(self.STAGES) - 1)]
        blockers = [key for key, passed in gates[next_stage].items() if not passed]
        payload = {
            "schema_version": self.SCHEMA,
            "generated_at": datetime.now(UTC).isoformat(),
            "stage": stage,
            "next_stage": next_stage,
            "next_stage_blockers": blockers,
            "gates": {
                "shadow": shadow_gates,
                "paper": paper_gates,
                "canary": canary_gates,
                "live_candidate": live_gates,
            },
            "forward": forward,
            "paper_economics": {
                "closed_trades": econ.get("closed_trades"),
                "expectancy_bps": econ.get("expectancy_bps"),
                "profit_factor": econ.get("profit_factor"),
            },
            "l3_status": "L3_UNSUPPORTED_BY_EXECUTION_VENUE",
            "execution_authority": "NONE",
            "candidate_status_is_not_execution_authority": True,
            "automatic_live_authority": False,
            "automatic_live_promotion": False,
            "manual_canonical_authority_required": True,
            "threshold_relaxation_allowed": False,
            "risk_widening_allowed": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return payload
