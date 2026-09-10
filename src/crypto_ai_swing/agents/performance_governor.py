from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from crypto_ai_swing.bridge.crypto_operations import NativeOperationsBridge


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


class PerformanceGovernor:
    """Evidence-first governor for continuous model/strategy improvement.

    It does not create financial authority. It turns observed failure modes into
    bounded research priorities and only recognizes improvement when hard OOS,
    cost, robustness, and causality gates remain satisfied.
    """

    SCHEMA = "crypto_ai_swing_performance_governor_v1"

    def __init__(self, settings) -> None:
        self.settings = settings
        self.cfg = dict(
            (getattr(settings, "autonomy", {}) or {}).get(
                "performance_governor", {}
            )
            or {}
        )
        self.root = (
            Path(settings.project_root)
            / "output/crypto_ai_swing/performance_governor"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.root / "latest.json"
        self.history_path = self.root / "history.jsonl"
        self.operations = NativeOperationsBridge(
            settings.crypto_repo_root,
            project_root=settings.project_root,
        )

    @staticmethod
    def _supervised(status: Mapping[str, Any]) -> dict[str, Any]:
        metrics = dict(status.get("metrics") or {})
        heads = dict(status.get("head_qualifications") or {})
        return {
            "status": str(status.get("status") or "UNKNOWN"),
            "alpha": bool(heads.get("alpha")),
            "return": bool(heads.get("return")),
            "regime": bool(heads.get("regime")),
            "risk": bool(heads.get("risk")),
            "execution": bool(heads.get("execution")),
            "probability_quality_pass": bool(
                metrics.get("probability_quality_pass", False)
            ),
            "positive_oos_net_proxy": bool(
                metrics.get("positive_oos_net_proxy", False)
            ),
            "model_selection_pass": bool(
                metrics.get("model_selection_pass", False)
            ),
            "stochastic_validation_pass": bool(
                metrics.get("stochastic_validation_pass", False)
            ),
            "point_in_time_universe_qualified": bool(
                metrics.get("point_in_time_universe_qualified", False)
            ),
            "alpha_auc": _finite(metrics.get("alpha_auc")),
            "validation_alpha_auc": _finite(
                metrics.get("validation_alpha_auc")
            ),
            "selected_mean_net": _finite(
                metrics.get("selected_mean_net_after_cost_floor")
            ),
            "selected_conservative_mean_net": _finite(
                metrics.get("selected_conservative_mean_net")
            ),
            "return_skill_vs_zero": _finite(
                (metrics.get("return_test") or {}).get("skill_vs_zero")
            ),
            "risk_skill": _finite(
                (metrics.get("risk_test") or {}).get(
                    "skill_vs_constant_quantile"
                )
            ),
        }

    @staticmethod
    def _rl(status: Mapping[str, Any]) -> dict[str, Any]:
        stochastic = dict(status.get("stochastic_validation") or {})
        test = dict(status.get("test_metrics") or {})
        return {
            "status": str(status.get("status") or "UNKNOWN"),
            "qualified": bool(status.get("qualified", False)),
            "stochastic_validation_pass": bool(
                stochastic.get("passed", False)
            ),
            "mean_return": _finite(test.get("mean_return")),
            "mean_excess_vs_buy_hold": _finite(
                test.get("mean_excess_vs_buy_hold")
            ),
            "positive_market_fraction": _finite(
                test.get("positive_market_fraction")
            ),
            "worst_maximum_drawdown": _finite(
                test.get("worst_maximum_drawdown")
            ),
            "bayesian_positive_probability": _finite(
                (status.get("bayesian") or {}).get(
                    "probability_cross_market_mean_positive"
                )
            ),
        }

    def evaluate(
        self,
        *,
        agent_status: Mapping[str, Any],
        attribution: Mapping[str, Any] | None = None,
        strategy: Mapping[str, Any] | None = None,
        native_research: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        supervised = self._supervised(
            dict(agent_status.get("supervised") or {})
        )
        rl = self._rl(dict(agent_status.get("rl") or {}))
        attribution = dict(attribution or {})
        strategy = dict(strategy or {})
        native_research = dict(native_research or {})

        priorities: list[dict[str, Any]] = []

        def add(code: str, severity: int, reason: str) -> None:
            priorities.append(
                {"code": code, "severity": severity, "reason": reason}
            )

        if not supervised["alpha"]:
            add(
                "IMPROVE_ALPHA_OOS_EDGE",
                100,
                "alpha head is not qualified",
            )
        if not supervised["positive_oos_net_proxy"]:
            add(
                "IMPROVE_NET_EXPECTANCY_AFTER_COSTS",
                98,
                "selected OOS net proxy is not positive",
            )
        if not supervised["return"]:
            add(
                "IMPROVE_RETURN_HEAD",
                92,
                "return head is not qualified",
            )
        if (
            supervised["return_skill_vs_zero"] is not None
            and supervised["return_skill_vs_zero"] <= 0
        ):
            add(
                "BEAT_ZERO_RETURN_BASELINE",
                95,
                "return model does not beat the zero-return baseline",
            )
        if not supervised["point_in_time_universe_qualified"]:
            add(
                "POINT_IN_TIME_UNIVERSE",
                96,
                "point-in-time universe qualification is incomplete",
            )
        if not supervised["model_selection_pass"]:
            add(
                "REDUCE_MODEL_SELECTION_OVERFIT",
                90,
                "multiple-testing/model-selection gate failed",
            )
        if not supervised["stochastic_validation_pass"]:
            add(
                "IMPROVE_STOCHASTIC_ROBUSTNESS",
                94,
                "supervised stochastic validation failed",
            )
        if not rl["qualified"]:
            add(
                "IMPROVE_RL_ROBUSTNESS",
                88,
                "RL challenger is not qualified",
            )
        if (
            rl["mean_excess_vs_buy_hold"] is not None
            and rl["mean_excess_vs_buy_hold"] <= 0
        ):
            add(
                "IMPROVE_RL_BENCHMARK_RELATIVE_RETURN",
                86,
                "RL underperforms buy-and-hold on average",
            )
        if (
            rl["worst_maximum_drawdown"] is not None
            and rl["worst_maximum_drawdown"]
            > float(self.cfg.get("maximum_research_drawdown", 0.25))
        ):
            add(
                "REDUCE_RL_TAIL_DRAWDOWN",
                89,
                "RL worst-market drawdown exceeds research target",
            )
        if not strategy.get("champion"):
            add(
                "DISCOVER_STRATEGY_CHAMPION",
                84,
                "no prospective strategy champion is active",
            )

        overall = dict(attribution.get("overall") or {})
        normal_net = dict(overall.get("normal_net") or {})
        mean_bps = _finite(normal_net.get("mean_bps"))
        if mean_bps is not None and mean_bps <= 0:
            add(
                "REDUCE_TURNOVER_AND_COST_DRAG",
                91,
                "prospective net attribution is not positive",
            )

        priorities.sort(
            key=lambda row: (-int(row["severity"]), str(row["code"]))
        )
        hard_gates = {
            "alpha_qualified": supervised["alpha"],
            "positive_oos_net": supervised["positive_oos_net_proxy"],
            "model_selection": supervised["model_selection_pass"],
            "stochastic_validation": supervised[
                "stochastic_validation_pass"
            ],
            "point_in_time_universe": supervised[
                "point_in_time_universe_qualified"
            ],
        }
        ready_for_more_model_influence = all(hard_gates.values())

        payload = {
            "schema_version": self.SCHEMA,
            "generated_at": _now(),
            "status": "IMPROVEMENT_REQUIRED"
            if priorities
            else "ROBUSTNESS_GATES_SATISFIED",
            "supervised": supervised,
            "rl": rl,
            "strategy_status": str(
                strategy.get("status") or "NOT_AVAILABLE"
            ),
            "native_research_status": str(
                native_research.get("status") or "NOT_AVAILABLE"
            ),
            "hard_gates": hard_gates,
            "ready_for_more_model_influence": (
                ready_for_more_model_influence
            ),
            "priorities": priorities,
            "optimization_policy": {
                "objective": (
                    "increase prospective net performance after costs while "
                    "preserving or improving tail risk"
                ),
                "must_improve_out_of_sample": True,
                "must_include_full_costs": True,
                "must_preserve_point_in_time_causality": True,
                "must_pass_multiple_testing_controls": True,
                "must_pass_stochastic_robustness": True,
                "must_not_worsen_canonical_risk": True,
                "threshold_relaxation_is_not_improvement": True,
                "live_pnl_is_not_a_training_target_without_maturation": True,
            },
            "automatic_live_authority": False,
            "automatic_live_promotion": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        self.operations.atomic_write_json(self.latest_path, payload)
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(payload, sort_keys=True, default=str) + "\n"
            )
        return payload

    def status(self) -> dict[str, Any]:
        try:
            payload = json.loads(
                self.latest_path.read_text(encoding="utf-8")
            )
            return dict(payload) if isinstance(payload, dict) else {}
        except (OSError, TypeError, ValueError):
            return {
                "schema_version": self.SCHEMA,
                "status": "NOT_BUILT",
                "priorities": [],
                "automatic_live_authority": False,
                "automatic_live_promotion": False,
            }
