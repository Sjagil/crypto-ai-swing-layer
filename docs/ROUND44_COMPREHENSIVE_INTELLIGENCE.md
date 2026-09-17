
# Round 44: Comprehensive Intelligence + Live Readiness

Round44 is an evidence, feature and learning expansion. It does not grant exchange execution authority.

## Ownership and reuse

`Sjagil/crypto` remains canonical for provider access, Bitvavo L1/L2, order-book reconstruction, durable orderflow, OFI/CVD/liquidity facts, indicators, market structure, macro, CMC-adjacent provider context, costs, risk, execution and live authority. `crypto-ai-swing-layer` orchestrates the 25-market active-swing workflow, stores 15-minute PIT intelligence, trains bounded challengers and evaluates readiness. The purged split policy is ported from `Sjagil/stocks-quant-agent-final-2` without adding a runtime dependency on that checkout.

## Real L1/L2, no synthetic L3

Bitvavo Market Data Pro is consumed as non-conflated aggregated price-level L2 with snapshot synchronization and `startMdSeqNo/endMdSeqNo` range validation. Sequence gaps fail closed and resynchronize. Book depth is capped at the venue-supported 1000 levels. L3 is always `L3_UNSUPPORTED_BY_EXECUTION_VENUE`; no market-by-order identities are invented.

The existing canonical `ProspectiveOrderflowRecorder` remains authoritative for CVD, OFI windows, weighted depth/imbalance, liquidity changes, gaps, slippage/impact, absorption and book-quality facts. Round44 surfaces and hashes those facts instead of rebuilding a second orderflow engine.

## Technical intelligence

Every market is evaluated on exactly 15m/1h/2h/4h/1d/1w using `Sjagil/crypto:research.features.FeaturePipeline` and the canonical indicator registry. This includes candlestick families, trend/momentum, volatility, HH/HL/LH/LL structure, BOS/CHoCH, confirmed fractals, breakouts, sweeps, volume/flow and BTC-relative features when the causal benchmark is available.

Supervised and PPO training are upgraded from the local compact feature builder to canonical technical features with train-only feature selection and redundancy pruning. Historical L1/L2, macro and news are deliberately not backfilled because that would fabricate point-in-time evidence.

## Point-in-time context and learning

Each 15-minute snapshot stores source/version hashes for technical features, L1, L2, macro, CMC, causal news/NLP, regime, supervised outputs, RL output and the final learning vector. Missing provider context is reported as `UNAVAILABLE/DATA_PROVIDER_REQUIRED` or an explicit stale/unavailable state. No synthetic substitute is allowed.

Matured 1h/4h/24h/72h/168h outcomes drive cross-horizon feature attribution. The prospective context challenger trains only when the dataset hash changes, uses training-only feature selection, purged/embargoed chronological splits, multi-seed validation, an untouched OOS test and normal/stressed cost economics. It remains advisory and cannot promote itself live.

## Readiness ladder

`RESEARCH -> SHADOW_QUALIFIED -> PAPER_QUALIFIED -> CANARY_CANDIDATE -> LIVE_CANDIDATE`

Long-horizon evidence, positive paper economics, full 25-market technical completeness and sequence-valid Market Data Pro are gates, not suggestions. `LIVE_CANDIDATE` still means candidate-grade evidence only. Canonical execution authority plus manual approval remain mandatory; automatic live promotion, threshold relaxation and risk widening stay disabled.
