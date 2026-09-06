# v0.23.1 Forward Evidence + Registry Hotfix

This hotfix addresses two correctness defects found in the first v0.23 run:

1. UnifiedAutonomyRuntime independently matured the same forward ledger at 1h
   while ProactiveTrader already matured it at the configured 15m execution
   timeframe. Because `forward_outcomes_v2` is keyed only by observation and
   horizon, this could mix execution-reference clocks in one evidence table.

2. ResearchPromotionRegistry searched old edge-artifact paths while v0.22
   ResearchEdgeManager writes to `output/crypto_ai_swing/agents/meta/latest.json`.

The runtime now delegates forward maturation to ProactiveTrader when
`mature_on_cycle=true`, and otherwise uses the configured execution timeframe.
The registry now discovers the canonical meta-edge artifact.

A separate repair script backs up and rebuilds the derived v2 forward outcomes
at the configured execution timeframe. It restores the backup automatically if
the rebuilt count is smaller than the previous count.
