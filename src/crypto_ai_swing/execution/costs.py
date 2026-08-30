from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CostEstimate:
    fee_bps: float
    spread_bps: float
    slippage_bps: float
    round_trip_bps: float
    net_edge_bps: float
    edge_to_cost_ratio: float
    approved: bool
    blockers: tuple[str, ...]


def estimate_cost(
    expected_edge_bps: float,
    spread_bps: float,
    volatility_pct: float,
    participation_rate: float,
    config: dict,
) -> CostEstimate:
    costs = config.get("costs", {})
    gate = config.get("edge_gate", {})
    liquidity = config.get("liquidity", {})
    blockers: list[str] = []

    fee = float(costs.get("fee_bps_per_side", 25.0))
    base_slip = float(costs.get("base_slippage_bps", 2.0))
    vol_coeff = float(costs.get("volatility_slippage_coefficient", 0.08))
    part_coeff = float(costs.get("participation_coefficient", 12.0))
    max_slip = float(costs.get("maximum_slippage_bps", 150.0))

    slippage = base_slip + vol_coeff * max(0.0, volatility_pct * 10000.0)
    slippage += part_coeff * math.sqrt(max(0.0, participation_rate))
    slippage = min(max_slip, slippage)

    one_way = fee + max(0.0, spread_bps) / 2.0 + slippage
    round_trip = one_way * 2.0 if gate.get("round_trip", True) else one_way
    net = expected_edge_bps - round_trip
    ratio = expected_edge_bps / round_trip if round_trip > 0 else float("inf")

    if spread_bps > float(liquidity.get("maximum_spread_bps", 80.0)):
        blockers.append("SPREAD")
    if participation_rate > float(liquidity.get("maximum_participation_rate", 0.05)):
        blockers.append("PARTICIPATION")
    if net < float(gate.get("minimum_net_edge_bps", 8.0)):
        blockers.append("NET_EDGE")
    if ratio < float(gate.get("minimum_edge_to_cost_ratio", 1.5)):
        blockers.append("EDGE_COST_RATIO")

    return CostEstimate(
        fee_bps=fee,
        spread_bps=spread_bps,
        slippage_bps=slippage,
        round_trip_bps=round_trip,
        net_edge_bps=net,
        edge_to_cost_ratio=ratio,
        approved=not blockers,
        blockers=tuple(blockers),
    )
