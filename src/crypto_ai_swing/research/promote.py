from __future__ import annotations


def promotion_decision(metrics: dict, config: dict) -> tuple[bool, list[str]]:
    gate = config.get("promotion", {})
    blockers: list[str] = []

    if int(metrics.get("trades", 0)) < int(gate.get("minimum_trades", 30)):
        blockers.append("MIN_TRADES")
    if float(metrics.get("profit_factor", 0.0)) < float(gate.get("minimum_profit_factor", 1.05)):
        blockers.append("PROFIT_FACTOR")
    if float(metrics.get("expectancy_bps", metrics.get("expectancy", 0.0) * 10000.0)) < float(
        gate.get("minimum_expectancy_bps", 0.0)
    ):
        blockers.append("EXPECTANCY")
    if float(metrics.get("psr", 0.0)) < float(gate.get("minimum_psr", 0.90)):
        blockers.append("PSR")
    if float(metrics.get("pbo", 1.0)) > float(gate.get("maximum_pbo", 0.50)):
        blockers.append("PBO")
    if float(metrics.get("max_drawdown", 1.0)) > float(gate.get("maximum_drawdown", 0.30)):
        blockers.append("DRAWDOWN")
    if gate.get("require_positive_stressed_oos", True) and not metrics.get("stressed_oos_positive", False):
        blockers.append("STRESSED_OOS")
    if gate.get("require_cross_engine_validation", True) and not metrics.get("cross_engine_validated", False):
        blockers.append("CROSS_ENGINE")
    if gate.get("require_forward_shadow", True) and not metrics.get("forward_shadow_pass", False):
        blockers.append("FORWARD_SHADOW")

    return not blockers, blockers
