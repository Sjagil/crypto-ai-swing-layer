from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import pandas as pd

from crypto_ai_swing.contracts import Authority, TradeIntent
from crypto_ai_swing.data.features import build_features
from crypto_ai_swing.strategies.swing import build_signal
from crypto_ai_swing.portfolio.allocator import allocate
from crypto_ai_swing.execution.costs import estimate_cost
from crypto_ai_swing.execution.shadow import ShadowLedger
from crypto_ai_swing.bridge.crypto_operations import NativeOperationsBridge


@dataclass
class PipelineResult:
    signals: list
    intents: list[TradeIntent]
    blocked: list[dict]


class SwingPipeline:
    def __init__(self, settings):
        self.settings = settings
        self.native = NativeOperationsBridge(settings.crypto_repo_root, project_root=settings.project_root)

    def run(
        self,
        frames: dict[str, pd.DataFrame],
        equity_eur: Decimal = Decimal("1000"),
        cash_eur: Decimal = Decimal("1000"),
        exposure_eur: Decimal = Decimal("0"),
        open_risk_eur: Decimal = Decimal("0"),
        spread_bps: dict[str, float] | None = None,
        market_context: dict[str, dict] | None = None,
        authority: Authority = Authority.SHADOW,
    ) -> PipelineResult:
        spread_bps = spread_bps or {}
        market_context = market_context or {}
        minimum = float(
            self.settings.swing.get("signals", {}).get(
                "minimum_entry_score", 0.62
            )
        )
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
                except Exception:
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
            )
            signals.append(signal)

        allocations = allocate(
            signals,
            equity_eur,
            cash_eur,
            exposure_eur,
            open_risk_eur,
            self.settings.risk,
        )
        intents: list[TradeIntent] = []
        blocked: list[dict] = []
        exec_cfg = self.settings.execution
        ttl = int(
            exec_cfg.get("execution", {}).get("intent_ttl_seconds", 120)
        )

        for allocation in allocations:
            signal, risk = allocation.signal, allocation.risk
            if signal.side.value != "BUY":
                context = market_context.get(signal.market, {})
                blockers = []
                if context.get("entry_blocked"):
                    blockers.append("CRYPTO_REPO_CONTEXT_GATE")
                if context.get("nlp_severe_negative"):
                    blockers.append("NLP_SEVERE_NEGATIVE")
                if blockers:
                    blocked.append(
                        {"market": signal.market, "blockers": blockers}
                    )
                continue
            if authority is Authority.LIVE and signal.edge_source == "HEURISTIC_SCORE_PROXY_RESEARCH_ONLY":
                blocked.append({"market": signal.market, "blockers": ["UNCALIBRATED_EXPECTED_EDGE_SOURCE"]})
                continue
            if not risk.approved:
                blocked.append(
                    {
                        "market": signal.market,
                        "blockers": list(risk.blockers),
                    }
                )
                continue
            order_notional = risk.order_notional_eur
            native_sizing = None
            price = float(signal.features.get("price", 0.0) or 0.0)
            if price > 0 and equity_eur > 0:
                try:
                    fn = self.native.native_interface("research.trading_math", "calculate_position_size_from_stop_fraction")
                    trade_cfg = self.settings.risk.get("trade", {})
                    portfolio_cfg = self.settings.risk.get("portfolio", {})
                    cost_cfg = self.settings.execution.get("costs", {})
                    native_sizing = fn(
                        float(equity_eur), float(trade_cfg.get("risk_per_trade_fraction", 0.0065)), price, float(signal.stop_pct),
                        fee_fraction_per_side=float(cost_cfg.get("fee_bps_per_side", 25.0))/10_000.0,
                        slippage_fraction_per_side=float(cost_cfg.get("base_slippage_bps", 2.0))/10_000.0,
                        max_position_fraction=float(portfolio_cfg.get("max_single_position_fraction", 0.25)), allow_fractional_units=True,
                    )
                    order_notional = min(order_notional, Decimal(str(native_sizing.position_notional)))
                except Exception as exc:
                    if authority is Authority.LIVE:
                        blocked.append({"market": signal.market, "blockers": ["NATIVE_POSITION_SIZING_UNAVAILABLE", type(exc).__name__]})
                        continue
            quote_volume = float(
                signal.features.get("quote_volume_24h", 1_000_000.0)
            )
            participation = float(order_notional) / max(1.0, quote_volume)
            cost = estimate_cost(
                signal.expected_edge_bps,
                spread_bps.get(signal.market, 10.0),
                signal.features.get("atr_pct", 0.02),
                participation,
                exec_cfg,
                quote_volume_eur=quote_volume,
            )
            if not cost.approved:
                blocked.append(
                    {
                        "market": signal.market,
                        "blockers": list(cost.blockers),
                    }
                )
                continue

            created = datetime.now(timezone.utc)
            context = market_context.get(signal.market, {})
            intent = TradeIntent.new(
                created_at=created,
                market=signal.market,
                side=signal.side,
                notional_eur=order_notional,
                expected_edge_bps=signal.expected_edge_bps,
                estimated_round_trip_cost_bps=cost.round_trip_bps,
                net_edge_bps=cost.net_edge_bps,
                stop_pct=signal.stop_pct,
                take_profit_pct=signal.take_profit_pct,
                trailing_stop_pct=signal.trailing_stop_pct,
                strategy=signal.strategy,
                authority=authority,
                expires_at=created + timedelta(seconds=ttl),
                metadata={
                    "signal_score": signal.score,
                    "signal_confidence": signal.confidence,
                    "edge_source": signal.edge_source,
                    "native_position_sizing": (native_sizing.to_dict() if native_sizing is not None else None),
                    "portfolio_heat_after": risk.portfolio_heat_after,
                    "crypto_repo_context": context,
                },
            )
            intents.append(intent)

        ledger_rel = self.settings.swing.get("paths", {}).get(
            "shadow_ledger",
            "output/crypto_ai_swing/shadow/shadow.sqlite",
        )
        ledger = ShadowLedger(self.settings.project_root / ledger_rel)
        try:
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
        )
