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


@dataclass
class PipelineResult:
    signals: list
    intents: list[TradeIntent]
    blocked: list[dict]


class SwingPipeline:
    def __init__(self, settings):
        self.settings = settings

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
                nlp_score=context.get("nlp_score"),
                nlp_confidence=context.get("nlp_confidence"),
                nlp_severe_negative=bool(
                    context.get("nlp_severe_negative", False)
                ),
                mtf_score=context.get("mtf_score"),
                orderflow_score=context.get("orderflow_score"),
                context_entry_blocked=bool(
                    context.get("entry_blocked", False)
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
            if not risk.approved:
                blocked.append(
                    {
                        "market": signal.market,
                        "blockers": list(risk.blockers),
                    }
                )
                continue
            quote_volume = float(
                signal.features.get("quote_volume_24h", 1_000_000.0)
            )
            participation = float(risk.order_notional_eur) / max(
                1.0, quote_volume
            )
            cost = estimate_cost(
                signal.expected_edge_bps,
                spread_bps.get(signal.market, 10.0),
                signal.features.get("atr_pct", 0.02),
                participation,
                exec_cfg,
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
                notional_eur=risk.order_notional_eur,
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
