# Round 5: autonomous agents and native live canary

This round adds causal supervised agents, an optional long-only PPO challenger,
continuous research/retraining supervision, canonical economics bootstrap, and
a native swing canary bridge into Sjagil/crypto.

The live bridge is hard capped at EUR 10 total exposure and one position. It
uses the existing canonical portfolio target -> risk approval -> execution
intent chain, shared live-capital reservation, durable ledger, idempotency and
reconciliation. A filled entry must receive an exchange-native stop. If native
protection fails, the bridge attempts an immediate risk-reducing flatten.

Applying this round does not arm live trading. Agent bundles are SHADOW and
have live_decision_influence=false. There is no automatic live promotion.
