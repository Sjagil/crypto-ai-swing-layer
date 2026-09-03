# v0.13 Production Data + Native Alpha Foundation

`Sjagil/crypto` is the canonical foundation for data health, point-in-time
research inputs, alpha campaigns, portfolio/risk primitives and live execution
authority.

## Added

- deep native foundation status
- canonical OHLCV persistence through the Sjagil/crypto data loader
- native candle-health plus enhanced non-mutating data quality audits
- strict point-in-time daily feature-store certification
- causal structural pattern scans
- direct research-only native alpha campaigns
- persistent fail-closed SHADOW/PAPER/CANARY mode controller
- isolated proactive position/decision state per mode
- one-shot supervisor completion semantics
- deterministic offline paper fee/slippage/PnL certification
- consolidated runtime/research/alpha/portfolio/health CLI commands
- `crypto-ai-swing` CLI alias

## Safety boundary

The swing layer still cannot independently grant live execution authority.
Canary remains blocked until prospective readiness, native canary authority and
the explicit execution environment all pass. Native alpha campaigns remain
research-only and do not automatically influence live decisions.

## State isolation

Proactive state is separated under:

- `output/crypto_ai_swing/modes/shadow/`
- `output/crypto_ai_swing/modes/paper/`
- `output/crypto_ai_swing/modes/canary/`

Only a SHADOW artifact may be consumed by canary preflight.

## Point-in-time scope

`pit-data-certify` proves that the strict native daily feature-store path can
produce a causal immutable tensor. It does not make the current dynamic
25-market agent dataset point-in-time qualified. That existing gate remains
fail-closed until its actual research universe is historically point-in-time.
