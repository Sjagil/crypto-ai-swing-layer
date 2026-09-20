# Round 42 — Autonomous Performance, Strategy Research, Browser Intelligence and Live Lifecycle

Round 42 turns the Round-41 unified decision/risk path into a continuously
evaluated research and performance system.

## Ownership

`Sjagil/crypto` remains authoritative for:

- provider/market data and point-in-time data semantics
- canonical transaction costs
- trading mathematics, Kelly and position sizing
- portfolio/correlation risk and Shariah eligibility
- exact backtesting, walk-forward and stochastic validation
- strategy DNA generation and research promotion states
- browser/RSS acquisition primitives
- Bitvavo account truth, reconciliation and real exchange execution
- native protective stop orders

`crypto-ai-swing-layer` owns orchestration:

- ChiefAgent / agent-of-agents
- performance diagnosis
- adaptive research hypotheses
- strategy/agent challenger scheduling
- structured external evidence use
- DecisionPacket generation
- prospective attribution
- production lifecycle coordination

## Continuous improvement

"Continuous improvement" means continuous measurement and experimentation,
not guaranteed profit and not automatic threshold relaxation.

The ChiefAgent checks every supervisor cycle. Work is cadence-bound:

- performance/forward attribution: 15 minutes
- browser/RSS intelligence: 15 minutes
- strategy challenger lab: 1 hour
- canonical strategy DNA / native tournaments / Stage-0 research: 6 hours
- supervised agent retraining: 4 hours (Round 41 manager)
- exact canonical research: 24 hours
- RL retraining: 24 hours (Round 41 manager)

A failing model/strategy generates explicit research priorities. Those become
preregistered experiments and challenger searches. Only evidence-improving
challengers may gain influence. Live promotion remains fail-closed.

## Strategy creation and use

The strategy director reuses:

- canonical deterministic strategy DNA generation
- canonical research factory and exact backtester
- canonical normal/stressed transaction costs
- canonical walk-forward/leakage checks
- canonical stochastic validation
- existing native alpha tournaments
- existing prospective StrategyChallengerLab

The current prospective strategy champion is attached to the edge manager used
by the agents/proactive runtime.

## Internet/browser

Browser and RSS acquisition reuse `Sjagil/crypto:scrapers.intelligence`.
External pages are `UNTRUSTED_EXTERNAL` evidence. Text from a webpage never has
instruction or execution authority.

## Live execution

The live path remains:

DecisionPacket
→ canonical RiskManager/Kelly
→ LiveExecutionGuard
→ `Sjagil/crypto:core.swing_layer_live`
→ Bitvavo

A filled BUY requires the canonical native protective STOP_LOSS. TAKE_PROFIT
and TRAILING_STOP use the canonical risk-reducing SELL path. The swing layer
does not implement Bitvavo REST/HMAC transport.

For live positions, the local lifecycle cache is written from a confirmed buy
response so take-profit/trailing management has entry geometry. Canonical
reconciliation remains final account truth.

A hard-stop price observation does not submit a second market SELL; it monitors
the native exchange stop and reconciles, avoiding duplicate stop-vs-market-exit
races.

## Commands

Read-only ChiefAgent cycle:

    python scripts/round42_chief_cycle.py --mode shadow --browser

Force strategy research and exact canonical validation:

    python scripts/round42_strategy_cycle.py --force --exact

Browser/RSS research:

    python scripts/round42_browser_research.py

Read-only live authority/reconciliation audit:

    python scripts/round42_live_audit.py

One full shadow runtime cycle:

    python scripts/round42_runtime.py --mode shadow

Continuous paper runtime:

    python scripts/round42_runtime.py --mode paper --forever

Live mode uses the existing canonical authority. Round 42 never activates that
authority automatically.
