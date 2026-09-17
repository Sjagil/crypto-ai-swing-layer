# Round 41 — Unified AI Decision + Canonical Portfolio/Risk/Kelly

Round 41 makes the two repositories one production chain without copying
canonical infrastructure into the AI repository.

`Sjagil/crypto` remains authoritative for market/provider data, canonical
candles, account truth, reconciliation, transaction costs, Kelly mathematics,
position sizing, portfolio/correlation risk, Shariah eligibility, kill switch,
and live execution.

`Sjagil/crypto-ai-swing-layer` remains authoritative for feature engineering,
AI/agent development, calibrated inference, RL challengers, research,
prospective validation, decision generation, agent management, attribution,
and retraining.

## One runtime chain

canonical crypto data
→ swing/MTF/orderflow/NLP/CMC context
→ supervised alpha/regime/return/risk heads + RL challenger + research manager
→ single `DecisionPacket`
→ canonical `core.economics.CanonicalCostModel`
→ canonical `research.trading_math` Kelly/sizing
→ canonical `risk.risk_manager.RiskManager`
→ `TradeIntent`
→ Round-40 canonical execution authority
→ Bitvavo

The AI layer can only reduce canonical risk. It cannot widen a quantity that
the canonical RiskManager approved. The configured 1.10% exceptional swing
tier is therefore still capped by the canonical crypto live-risk ceiling.

## Agent-of-agents

`AgentManager` checks agent state every supervisor cycle. Missing, expired or
errored supervised/RL artifacts are retrained immediately. Healthy artifacts
are retrained on bounded schedules (supervised 4h, RL 24h by default). It also
refreshes the existing edge manager and promotion registry.

This avoids retraining on every market tick, which would create training
thrash/leakage. The manager is continuous, while training is evidence- and
schedule-driven.

The manager cannot arm live authority and cannot auto-promote a model live.

## Real data-flow verification

Read-only public flow:

    python scripts/round41_dataflow_audit.py --network

Strict paid/deep prospective context:

    python scripts/round41_dataflow_audit.py --network --deep

After training, require a trained supervised artifact:

    python scripts/round41_dataflow_audit.py --network --require-trained-agents

Force one agent-manager training cycle:

    python scripts/round41_agent_manager_once.py --force-train

The audit verifies real OHLCV on 15m/1h/4h for BTC/ETH/SOL, a canonical
market bundle (ticker/orderbook/trades/microstructure), feature engineering,
agent/RL runtime paths, DecisionPacket construction, and canonical
cost/RiskManager/Kelly plumbing. It is read-only and submits zero orders.

A scheduled/manual GitHub workflow runs the real public-data smoke every six
hours without trading secrets.
