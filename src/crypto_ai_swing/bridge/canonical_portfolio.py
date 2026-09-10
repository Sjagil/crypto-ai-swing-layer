from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.contracts import Authority, RiskPlan
from crypto_ai_swing.decision_packet import DecisionPacket


@dataclass(frozen=True)
class CanonicalPortfolioDecision:
    risk_plan: RiskPlan
    net_edge_bps: float
    round_trip_cost_bps: float
    canonical_cost_model_version: str
    kelly: dict[str, Any]
    canonical_risk: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_plan": {
                "market": self.risk_plan.market,
                "equity_eur": str(self.risk_plan.equity_eur),
                "available_cash_eur": str(
                    self.risk_plan.available_cash_eur
                ),
                "order_notional_eur": str(
                    self.risk_plan.order_notional_eur
                ),
                "risk_eur": str(self.risk_plan.risk_eur),
                "stop_pct": self.risk_plan.stop_pct,
                "portfolio_heat_after": self.risk_plan.portfolio_heat_after,
                "approved": self.risk_plan.approved,
                "blockers": list(self.risk_plan.blockers),
            },
            "net_edge_bps": self.net_edge_bps,
            "round_trip_cost_bps": self.round_trip_cost_bps,
            "canonical_cost_model_version": self.canonical_cost_model_version,
            "kelly": self.kelly,
            "canonical_risk": self.canonical_risk,
        }


class CanonicalPortfolioRiskBridge:
    """Thin adapter; risk, Kelly and cost authority stay in Sjagil/crypto."""

    def __init__(
        self,
        crypto_repo_root: Path,
        *,
        project_root: Path,
        risk_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.crypto = CryptoLibraryBridge(crypto_repo_root)
        self.project_root = Path(project_root).expanduser().resolve()
        self.risk_config = dict(risk_config or {})

    def _native(self):
        settings = self.crypto.settings()
        risk_module = self.crypto.import_module("risk.risk_manager")
        math_module = self.crypto.import_module("research.trading_math")
        cost_module = self.crypto.import_module("core.economics")
        return settings, risk_module, math_module, cost_module

    @staticmethod
    def returns_from_frames(
        frames: Mapping[str, pd.DataFrame] | None,
    ) -> dict[str, pd.Series]:
        output: dict[str, pd.Series] = {}
        for market, frame in (frames or {}).items():
            if frame is None or frame.empty or "close" not in frame.columns:
                continue
            values = pd.to_numeric(
                frame["close"], errors="coerce"
            ).pct_change()
            values = values.replace(
                [float("inf"), float("-inf")], float("nan")
            ).dropna()
            if not values.empty:
                output[str(market).upper()] = values
        return output

    @staticmethod
    def _position_rows(
        positions: Iterable[Mapping[str, Any]] | None,
        *,
        exposure_eur: Decimal,
        open_risk_eur: Decimal,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        represented_exposure = 0.0
        represented_risk = 0.0
        for source in positions or ():
            row = dict(source)
            try:
                market = str(row["market"]).upper()
                quantity = float(
                    row.get("quantity", row.get("amount", 0.0))
                )
                mark_price = float(
                    row.get("mark_price", row.get("entry_price", 0.0))
                )
                risk = float(row.get("open_risk_eur", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            if quantity <= 0 or mark_price <= 0:
                continue
            rows.append(
                {
                    "market": market,
                    "quantity": quantity,
                    "mark_price": mark_price,
                    "open_risk_eur": max(0.0, risk),
                }
            )
            represented_exposure += quantity * mark_price
            represented_risk += max(0.0, risk)

        residual_exposure = max(
            0.0, float(exposure_eur) - represented_exposure
        )
        residual_risk = max(0.0, float(open_risk_eur) - represented_risk)
        if residual_exposure > 1e-9:
            rows.append(
                {
                    "market": "UNTRACKED-EUR",
                    "quantity": residual_exposure,
                    "mark_price": 1.0,
                    "open_risk_eur": residual_risk,
                }
            )
        return rows

    def canonical_cost(self) -> dict[str, Any]:
        settings, _risk, _math, cost_module = self._native()
        model = cost_module.CanonicalCostModel.from_settings(settings)
        return {
            "cost_model_version": str(model.cost_model_version),
            "round_trip_bps": (
                float(model.conservative_roundtrip_fraction) * 10_000.0
            ),
            "taker_fee_bps_per_side": (
                float(model.taker_fee_fraction) * 10_000.0
            ),
            "spread_bps": float(model.spread_bps),
            "slippage_bps_per_side": float(model.slippage_bps),
            "calibration_status": str(model.calibration_status),
        }

    def assess(
        self,
        packet: DecisionPacket,
        *,
        equity_eur: Decimal,
        cash_eur: Decimal,
        exposure_eur: Decimal,
        open_risk_eur: Decimal,
        positions: Iterable[Mapping[str, Any]] | None = None,
        frames: Mapping[str, pd.DataFrame] | None = None,
        day_start_equity_eur: Decimal | None = None,
        peak_equity_eur: Decimal | None = None,
        trades_today: int = 0,
        reconciled: bool = True,
        data_healthy: bool = True,
        risk_manager_healthy: bool = True,
        intelligence_timing_healthy: bool = True,
        drawdown_state: str = "NORMAL",
        execution_validation_canary: bool = False,
        maximum_canary_notional_eur: Decimal | None = None,
    ) -> CanonicalPortfolioDecision:
        settings, risk_module, math_module, cost_module = self._native()
        blockers: list[str] = []

        model = cost_module.CanonicalCostModel.from_settings(settings)
        baseline_round_trip_bps = (
            float(model.conservative_roundtrip_fraction) * 10_000.0
        )
        try:
            observed_spread_bps = float(
                packet.evidence.get("observed_spread_bps")
            )
        except (TypeError, ValueError):
            observed_spread_bps = float(model.spread_bps)
        if observed_spread_bps < 0.0 or not math.isfinite(observed_spread_bps):
            observed_spread_bps = float(model.spread_bps)
        dynamic_round_trip_bps = (
            2.0 * float(model.taker_fee_fraction) * 10_000.0
            + max(float(model.spread_bps), observed_spread_bps)
            + 2.0 * float(model.slippage_bps)
            + float(model.failed_execution_allowance_bps)
            + float(model.partial_fill_impact_bps)
        )
        round_trip_cost_bps = max(
            baseline_round_trip_bps,
            dynamic_round_trip_bps,
        )
        net_edge_bps = float(packet.expected_edge_bps) - round_trip_cost_bps
        minimum_net_edge_bps = float(
            (self.risk_config.get("kelly", {}) or {}).get(
                "minimum_net_edge_bps", 0.0
            )
        )

        if packet.side.value != "BUY":
            blockers.append("NO_ENTRY_SIGNAL")
        if equity_eur <= 0 or cash_eur < 0:
            blockers.append("INVALID_ACCOUNT_STATE")
        if packet.entry_price <= 0:
            blockers.append("INVALID_ENTRY_PRICE")
        if net_edge_bps <= minimum_net_edge_bps:
            blockers.append("CANONICAL_NET_EDGE_NON_POSITIVE")

        native_positions = []
        for row in self._position_rows(
            positions,
            exposure_eur=exposure_eur,
            open_risk_eur=open_risk_eur,
        ):
            native_positions.append(
                risk_module.PositionExposure(
                    market=str(row["market"]),
                    quantity=float(row["quantity"]),
                    mark_price=float(row["mark_price"]),
                    open_risk_eur=float(row["open_risk_eur"]),
                )
            )

        snapshot = risk_module.PortfolioSnapshot(
            equity_eur=float(equity_eur),
            cash_eur=float(cash_eur),
            day_start_equity_eur=float(
                day_start_equity_eur or equity_eur
            ),
            peak_equity_eur=float(peak_equity_eur or equity_eur),
            trades_today=max(0, int(trades_today)),
            positions=tuple(native_positions),
            reconciled=bool(reconciled),
            risk_manager_healthy=bool(risk_manager_healthy),
            data_healthy=bool(data_healthy),
            intelligence_timing_healthy=bool(
                intelligence_timing_healthy
            ),
            returns_by_market=self.returns_from_frames(frames),
            drawdown_state=str(drawdown_state),
        )

        risk_manager = risk_module.RiskManager.from_settings(
            settings,
            kill_switch_path=(
                settings.paths.checkpoints_dir / "kill_switch.json"
            ),
        )
        stop_price = packet.entry_price * (1.0 - packet.stop_pct)
        reliability = packet.evidence.get("reliability_evidence")
        if not isinstance(reliability, dict) or not reliability:
            reliability = None

        canonical_risk_decision = risk_manager.assess_entry(
            market=packet.market,
            entry_price=float(packet.entry_price),
            stop_price=float(stop_price),
            snapshot=snapshot,
            size_multiplier=1.0,
            reliability_evidence=reliability,
            live_mode=packet.authority is Authority.LIVE,
        )
        canonical_approved = bool(
            getattr(canonical_risk_decision, "approved", False)
        )
        canonical_quantity = float(
            getattr(canonical_risk_decision, "approved_quantity", 0.0)
            or 0.0
        )
        canonical_risk_eur = float(
            getattr(canonical_risk_decision, "risk_eur", 0.0)
            or 0.0
        )
        reason_codes = [
            str(value)
            for value in (
                getattr(canonical_risk_decision, "reason_codes", ()) or ()
            )
        ]
        if not canonical_approved:
            blockers.extend(reason_codes or ["CANONICAL_RISK_REJECTED"])

        kcfg = dict(self.risk_config.get("kelly", {}) or {})
        trade_cfg = dict(self.risk_config.get("trade", {}) or {})
        probability = packet.calibrated_win_probability
        effective_probability = None
        full_kelly = None
        fractional_kelly_value = None
        requested_risk_fraction = None
        kelly_size = None
        tier = "BASE"

        if probability is not None:
            uncertainty_shrink = max(
                0.0,
                min(
                    1.0,
                    float(kcfg.get("uncertainty_probability_shrink", 0.75)),
                ),
            )
            shrink = max(
                0.0,
                min(1.0, packet.uncertainty * uncertainty_shrink),
            )
            effective_probability = 0.5 + (
                float(probability) - 0.5
            ) * (1.0 - shrink)
            reward_risk = max(
                0.01,
                float(packet.take_profit_pct) / float(packet.stop_pct),
            )
            full_kelly = float(
                math_module.kelly_fraction(
                    effective_probability,
                    reward_risk,
                    1.0,
                )
            )
            fractional_kelly_value = float(
                math_module.fractional_kelly(
                    effective_probability,
                    reward_risk,
                    1.0,
                    fraction=float(
                        kcfg.get("fractional_kelly_fraction", 0.25)
                    ),
                )
            )

            if (
                packet.score
                >= float(kcfg.get("exceptional_score", 0.82))
                and packet.confidence
                >= float(kcfg.get("exceptional_confidence", 0.75))
                and packet.prospective_qualified
            ):
                tier = "EXCEPTIONAL"
                tier_ceiling = float(
                    trade_cfg.get("exceptional_risk_fraction", 0.011)
                )
            elif (
                packet.score >= float(kcfg.get("strong_score", 0.72))
                and packet.confidence
                >= float(kcfg.get("strong_confidence", 0.65))
            ):
                tier = "STRONG"
                tier_ceiling = float(
                    trade_cfg.get("strong_risk_fraction", 0.009)
                )
            else:
                tier_ceiling = float(
                    trade_cfg.get("risk_per_trade_fraction", 0.0065)
                )

            canonical_mode_cap = float(
                settings.risk.maximum_live_risk_per_trade
                if packet.authority is Authority.LIVE
                else settings.risk.maximum_research_risk_per_trade
            )
            requested_risk_fraction = max(
                0.0,
                min(
                    fractional_kelly_value,
                    tier_ceiling,
                    canonical_mode_cap,
                ),
            )
            if requested_risk_fraction <= 0.0:
                blockers.append("KELLY_NON_POSITIVE")
            else:
                kelly_size = math_module.calculate_position_size(
                    float(equity_eur),
                    requested_risk_fraction,
                    float(packet.entry_price),
                    float(stop_price),
                    fee_fraction_per_side=float(
                        settings.costs.default_fee
                    ),
                    slippage_fraction_per_side=(
                        float(settings.costs.slippage_bps)
                        + float(settings.costs.spread_bps) / 2.0
                    )
                    / 10_000.0,
                    max_position_fraction=float(
                        settings.risk.maximum_position_fraction
                    ),
                    allow_fractional_units=True,
                )
        elif (
            packet.authority is Authority.LIVE
            and not execution_validation_canary
            and bool(
                kcfg.get(
                    "require_calibrated_probability_for_live", True
                )
            )
        ):
            blockers.append("CALIBRATED_PROBABILITY_REQUIRED_FOR_LIVE")

        final_quantity = canonical_quantity
        if kelly_size is not None:
            # AI/Kelly can only reduce the canonical RiskManager quantity.
            final_quantity = min(
                final_quantity,
                float(kelly_size.units),
            )
        notional = max(0.0, final_quantity * packet.entry_price)
        if maximum_canary_notional_eur is not None:
            notional = min(
                notional,
                float(maximum_canary_notional_eur),
            )
        notional = min(notional, max(0.0, float(cash_eur)))
        if notional <= 0.0:
            blockers.append("ZERO_CANONICAL_POSITION")

        # Risk is a conservative estimate after any Kelly/canary reduction.
        fraction_of_canonical = (
            min(1.0, final_quantity / canonical_quantity)
            if canonical_quantity > 0.0
            else 0.0
        )
        final_risk_eur = canonical_risk_eur * fraction_of_canonical
        heat_after = (
            (float(open_risk_eur) + final_risk_eur) / float(equity_eur)
            if equity_eur > 0
            else 1.0
        )

        approved = not blockers
        risk_plan = RiskPlan(
            market=packet.market,
            equity_eur=equity_eur,
            available_cash_eur=cash_eur,
            order_notional_eur=Decimal(str(round(notional, 8))),
            risk_eur=Decimal(str(round(final_risk_eur, 8))),
            stop_pct=float(packet.stop_pct),
            portfolio_heat_after=float(heat_after),
            approved=approved,
            blockers=tuple(dict.fromkeys(blockers)),
        )
        return CanonicalPortfolioDecision(
            risk_plan=risk_plan,
            net_edge_bps=net_edge_bps,
            round_trip_cost_bps=round_trip_cost_bps,
            canonical_cost_model_version=str(model.cost_model_version),
            kelly={
                "applied": probability is not None,
                "probability_source": packet.probability_source,
                "raw_probability": probability,
                "effective_probability": effective_probability,
                "uncertainty": packet.uncertainty,
                "fractional_kelly_fraction": float(
                    kcfg.get("fractional_kelly_fraction", 0.25)
                ),
                "full_kelly": full_kelly,
                "fractional_kelly": fractional_kelly_value,
                "requested_risk_fraction": requested_risk_fraction,
                "risk_tier": tier,
                "canonical_risk_never_widened": True,
            },
            canonical_risk={
                "approved": canonical_approved,
                "reason_codes": reason_codes,
                "approved_quantity": canonical_quantity,
                "risk_eur": canonical_risk_eur,
                "snapshot_exposure_eur": float(snapshot.exposure_eur),
                "snapshot_open_risk_eur": float(snapshot.open_risk_eur),
                "backend": "Sjagil/crypto:risk.risk_manager.RiskManager",
            },
        )
