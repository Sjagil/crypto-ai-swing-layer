from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd

from crypto_ai_swing.bridge.canonical_portfolio import (
    CanonicalPortfolioRiskBridge,
)
from crypto_ai_swing.contracts import Authority, TradeIntent
from crypto_ai_swing.data.features import build_features
from crypto_ai_swing.decision_packet import DecisionPacket
from crypto_ai_swing.execution.active_swing_canary import (
    evaluate_execution_validation_canary,
)
from crypto_ai_swing.execution.shadow import ShadowLedger
from crypto_ai_swing.strategies.swing import build_signal


@dataclass
class PipelineResult:
    signals: list
    intents: list[TradeIntent]
    blocked: list[dict]
    decision_packets: list[DecisionPacket] = field(default_factory=list)


class SwingPipeline:
    """AI decision generation -> canonical Sjagil/crypto portfolio authority."""

    def __init__(self, settings):
        self.settings = settings
        self.canonical = CanonicalPortfolioRiskBridge(
            settings.crypto_repo_root,
            project_root=settings.project_root,
            risk_config=settings.risk,
        )

    @staticmethod
    def _portfolio_rows(
        positions: list[Mapping[str, Any]] | None,
        frames: Mapping[str, pd.DataFrame],
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for row in positions or []:
            item = dict(row)
            market = str(item.get("market") or "").upper()
            frame = frames.get(market)
            mark = item.get("mark_price")
            if (
                frame is not None
                and not frame.empty
                and "close" in frame.columns
            ):
                try:
                    mark = float(frame["close"].iloc[-1])
                except (TypeError, ValueError):
                    pass
            try:
                output.append(
                    {
                        "market": market,
                        "quantity": float(
                            item.get("quantity", item.get("amount", 0.0))
                        ),
                        "mark_price": float(
                            mark
                            if mark is not None
                            else item.get("entry_price", 0.0)
                        ),
                        "open_risk_eur": float(
                            item.get("open_risk_eur", 0.0)
                        ),
                    }
                )
            except (TypeError, ValueError):
                continue
        return output

    def run(
        self,
        frames: dict[str, pd.DataFrame],
        equity_eur: Decimal = Decimal(1000),
        cash_eur: Decimal = Decimal(1000),
        exposure_eur: Decimal = Decimal(0),
        open_risk_eur: Decimal = Decimal(0),
        spread_bps: dict[str, float] | None = None,
        market_context: dict[str, dict] | None = None,
        authority: Authority = Authority.SHADOW,
        portfolio_positions: list[Mapping[str, Any]] | None = None,
        account_state: Mapping[str, Any] | None = None,
    ) -> PipelineResult:
        spread_bps = spread_bps or {}
        market_context = market_context or {}
        account_state = dict(account_state or {})
        signals_cfg = dict(self.settings.swing.get("signals", {}) or {})
        minimum = float(signals_cfg.get("minimum_entry_score", 0.62))
        minimum_model_probability = float(
            signals_cfg.get("minimum_model_probability", 0.55)
        )
        ensemble_weights = dict(signals_cfg.get("ensemble", {}) or {})
        signals = []

        for market, frame in frames.items():
            feat = build_features(frame).dropna()
            if feat.empty:
                continue
            row = feat.iloc[-1].copy()
            if "quote_volume_24h" in frame.columns:
                try:
                    row["quote_volume_24h"] = float(
                        frame["quote_volume_24h"].dropna().iloc[-1]
                    )
                except (IndexError, TypeError, ValueError):
                    pass
            ts = feat.index[-1].to_pydatetime()
            context = market_context.get(market, {})
            signal = build_signal(
                market,
                ts,
                row,
                ml_probability=context.get("ml_probability"),
                forecast_score=context.get("forecast_score"),
                predicted_return=context.get("predicted_return"),
                predicted_mae=context.get("predicted_mae"),
                rl_score=context.get("rl_score"),
                research_meta_score=context.get("research_meta_score"),
                research_strategy_hint=context.get("research_strategy_hint"),
                nlp_score=context.get("nlp_score"),
                nlp_confidence=context.get("nlp_confidence"),
                nlp_severe_negative=bool(
                    context.get("nlp_severe_negative", False)
                ),
                mtf_score=context.get("mtf_score"),
                orderflow_score=context.get("orderflow_score"),
                context_entry_blocked=bool(
                    context.get("entry_blocked", False)
                    or context.get("agent_entry_blocked", False)
                ),
                minimum_entry_score=minimum,
                minimum_model_probability=minimum_model_probability,
                ensemble_weights=ensemble_weights,
            )
            signals.append(signal)

        maximum_positions = int(
            (self.settings.risk.get("portfolio", {}) or {}).get(
                "max_positions", 5
            )
        )
        current_markets = {
            str(row.get("market") or "").upper()
            for row in (portfolio_positions or [])
        }
        remaining_slots = max(0, maximum_positions - len(current_markets))
        candidates = sorted(
            (signal for signal in signals if signal.side.value == "BUY"),
            key=lambda signal: (signal.score, signal.expected_edge_bps),
            reverse=True,
        )
        # Existing markets may still be resized/reassessed; new markets are
        # bounded by the conservative swing capacity before canonical risk.
        selected = []
        new_slots_used = 0
        for signal in candidates:
            if signal.market in current_markets:
                selected.append(signal)
            elif new_slots_used < remaining_slots:
                selected.append(signal)
                new_slots_used += 1

        intents: list[TradeIntent] = []
        blocked: list[dict] = []
        packets: list[DecisionPacket] = []
        ttl = int(
            self.settings.execution.get("execution", {}).get(
                "intent_ttl_seconds", 120
            )
        )

        running_cash = Decimal(cash_eur)
        running_exposure = Decimal(exposure_eur)
        running_risk = Decimal(open_risk_eur)
        running_positions = self._portfolio_rows(
            portfolio_positions,
            frames,
        )
        maximum_spread = float(
            (self.settings.execution.get("liquidity", {}) or {}).get(
                "maximum_spread_bps", 35.0
            )
        )

        for signal in selected:
            context = market_context.get(signal.market, {})
            observed_spread = float(
                spread_bps.get(
                    signal.market,
                    context.get("spread_bps", 0.0) or 0.0,
                )
            )
            if observed_spread > maximum_spread:
                blocked.append(
                    {"market": signal.market, "blockers": ["SPREAD"]}
                )
                continue

            validation = evaluate_execution_validation_canary(
                authority=authority,
                signal=signal,
                context=context,
                proactive=self.settings.proactive,
            )
            canary_allowed = bool(validation.get("allowed"))
            uncalibrated_edge = (
                signal.edge_source
                == "HEURISTIC_SCORE_PROXY_RESEARCH_ONLY"
            )
            if (
                authority is Authority.LIVE
                and uncalibrated_edge
                and not canary_allowed
            ):
                reasons = ["UNCALIBRATED_EXPECTED_EDGE_SOURCE"]
                reasons.extend(
                    str(value)
                    for value in validation.get("blockers", [])
                )
                blocked.append(
                    {
                        "market": signal.market,
                        "blockers": sorted(set(reasons)),
                    }
                )
                continue

            packet = DecisionPacket.from_signal(
                signal,
                context=context,
                authority=authority,
            )
            packets.append(packet)
            maximum_canary_notional = (
                Decimal(str(validation.get("maximum_order_eur", "10")))
                if canary_allowed
                else None
            )

            canonical = self.canonical.assess(
                packet,
                equity_eur=equity_eur,
                cash_eur=running_cash,
                exposure_eur=running_exposure,
                open_risk_eur=running_risk,
                positions=running_positions,
                frames=frames,
                day_start_equity_eur=Decimal(
                    str(
                        account_state.get(
                            "day_start_equity_eur", equity_eur
                        )
                    )
                ),
                peak_equity_eur=Decimal(
                    str(
                        account_state.get(
                            "peak_equity_eur", equity_eur
                        )
                    )
                ),
                trades_today=int(account_state.get("trades_today", 0)),
                reconciled=bool(
                    account_state.get("reconciled", True)
                ),
                data_healthy=bool(
                    account_state.get("data_healthy", True)
                ),
                risk_manager_healthy=bool(
                    account_state.get("risk_manager_healthy", True)
                ),
                intelligence_timing_healthy=bool(
                    account_state.get(
                        "intelligence_timing_healthy", True
                    )
                ),
                drawdown_state=str(
                    account_state.get("drawdown_state", "NORMAL")
                ),
                execution_validation_canary=canary_allowed,
                maximum_canary_notional_eur=maximum_canary_notional,
            )
            risk = canonical.risk_plan
            if not risk.approved:
                blocked.append(
                    {
                        "market": signal.market,
                        "blockers": list(risk.blockers),
                        "decision_packet_hash": packet.canonical_hash(),
                        "canonical_risk": canonical.to_dict(),
                    }
                )
                continue

            created = datetime.now(UTC)
            intent = TradeIntent.new(
                created_at=created,
                market=signal.market,
                side=signal.side,
                notional_eur=risk.order_notional_eur,
                expected_edge_bps=signal.expected_edge_bps,
                estimated_round_trip_cost_bps=canonical.round_trip_cost_bps,
                net_edge_bps=canonical.net_edge_bps,
                stop_pct=signal.stop_pct,
                take_profit_pct=signal.take_profit_pct,
                trailing_stop_pct=signal.trailing_stop_pct,
                strategy=signal.strategy,
                authority=authority,
                expires_at=created + timedelta(seconds=ttl),
                metadata={
                    "decision_packet": packet.to_dict(),
                    "decision_packet_hash": packet.canonical_hash(),
                    "signal_score": signal.score,
                    "signal_confidence": signal.confidence,
                    "edge_source": signal.edge_source,
                    "canonical_portfolio_risk": canonical.to_dict(),
                    "canonical_cost_model_version": (
                        canonical.canonical_cost_model_version
                    ),
                    "portfolio_heat_after": risk.portfolio_heat_after,
                    "crypto_repo_context": context,
                    "execution_validation_canary": canary_allowed,
                    "execution_validation_canary_policy": (
                        validation if canary_allowed else None
                    ),
                    "economic_edge_unproven": bool(canary_allowed),
                    "alpha_evidence_authorized": False,
                    "automatic_live_promotion": False,
                    "autoscale_authorized": False,
                    "risk_authority": (
                        "Sjagil/crypto:risk.risk_manager.RiskManager"
                    ),
                    "kelly_authority": (
                        "Sjagil/crypto:research.trading_math"
                    ),
                    "cost_authority": (
                        "Sjagil/crypto:core.economics.CanonicalCostModel"
                    ),
                },
            )
            intents.append(intent)
            running_cash -= risk.order_notional_eur
            running_exposure += risk.order_notional_eur
            running_risk += risk.risk_eur
            running_positions.append(
                {
                    "market": signal.market,
                    "quantity": (
                        float(risk.order_notional_eur)
                        / max(packet.entry_price, 1e-12)
                    ),
                    "mark_price": packet.entry_price,
                    "open_risk_eur": float(risk.risk_eur),
                }
            )

        ledger_rel = self.settings.swing.get("paths", {}).get(
            "shadow_ledger",
            "output/crypto_ai_swing/shadow/shadow.sqlite",
        )
        ledger = ShadowLedger(self.settings.project_root / ledger_rel)
        try:
            for packet in packets:
                ledger.append(
                    packet.packet_id,
                    "DECISION_PACKET",
                    packet.to_dict(),
                )
            for intent in intents:
                ledger.append(
                    intent.intent_id,
                    "TRADE_INTENT",
                    intent.to_dict(),
                )
        finally:
            ledger.close()

        return PipelineResult(
            signals=signals,
            intents=intents,
            blocked=blocked,
            decision_packets=packets,
        )
