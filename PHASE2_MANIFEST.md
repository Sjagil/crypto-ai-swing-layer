# Phase 2 Round 1

Adds the first proactive/NLP/live-capable build round to `crypto-ai-swing-layer`.

Included:
- FinBERT finance NLP with deterministic offline fallback
- asset/entity attribution and event tags
- severe negative NLP entry veto
- bounded NLP contribution to strategy scoring
- proactive continuous runtime
- restart-safe SQLite decisions/positions/orders state
- Bitvavo public candles/ticker/account bridge
- fail-closed live preflight gates
- live-capable market buy/sell backend with no withdrawal method
- stop loss, take profit, trailing stop lifecycle
- shadow/paper/live authorities
- v0.2 package metadata/dependency sync
- tests for NLP, state persistence, Bitvavo signatures and live gates

Live remains disabled in `config/execution.yaml` by default. It becomes live-capable only after explicit configuration and environment gates pass.

This is Round 1. Round 2 should replace the direct Bitvavo live backend as the preferred path with the existing `Sjagil/crypto` execution authority and reconciliation bridge once its concrete intent-consumer command/interface is mapped.
