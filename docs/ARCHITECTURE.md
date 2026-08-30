# Architecture

## Boundary rule

The existing crypto repository remains the authority for:

- Bitvavo REST and WebSocket connectivity
- exchange metadata and precision
- balances, orders, fills and reconciliation
- CoinMarketCap access
- historical acquisition
- live feeds
- scrapers and RSS
- secret handling
- final exchange execution

This layer owns:

- canonical feature preparation
- multi-timeframe alignment
- deterministic swing strategies
- supervised ML
- forecast challengers
- RL challengers
- leakage-safe research
- portfolio allocation proposals
- execution-cost estimation
- shadow lifecycle
- `TradeIntent` creation

The bridge between both sides is deliberately narrow.

## No exchange calls from strategies

```text
Strategy
  -> Signal
  -> Ensemble
  -> Cost gate
  -> RiskPlan
  -> TradeIntent
  -> External execution adapter
  -> Existing safety gate
  -> Existing Bitvavo execution
```

A strategy module has no Bitvavo client import.

## Data source authority

For a Bitvavo EUR execution route:

1. Bitvavo is native execution-market truth.
2. CoinMarketCap is universe, ranking, metadata and point-in-time context.
3. Other historical providers can be research proxies but must retain provenance.
4. Scrapers supply causal context only when publication timing is known.
5. Reference repositories are challengers, not data truth and not exchange authority.
