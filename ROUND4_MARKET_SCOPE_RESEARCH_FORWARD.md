# Round 4: market scope, native research execution and forward maturation

Version: 0.5.0

## Correctness fixes

1. Bitvavo trade `quantity` is now consumed by microstructure CVD. Both base-volume CVD and quote-notional CVD are exposed.
2. Native RSS records are enriched with `Sjagil/crypto:scrapers.intelligence.classify_text` when available.
3. NLP aggregation is market-scoped. Direct asset news has full weight. BTC news can inform alt markets at a bounded cross-asset weight, but cannot transfer a severe veto. Generic security and regulatory headlines are context, not a global severe veto.
4. Per-market NLP diagnostics record scope counts and top contributors.

## Research execution

`strategy-factory-plan` remains a preregistration surface. Round 4 adds `research-run`, which invokes `research.research_factory.build_research_factory_artifact` in the existing crypto repository.

Default:

```bash
python -m crypto_ai_swing.cli research-run --stage0-only --maximum-rows 20000
```

Exact validation is manual:

```bash
python -m crypto_ai_swing.cli research-run --exact --maximum-rows 20000
```

The native research factory remains authoritative for Stage 0, exact backtests, leakage checks, cost stress, walk-forward and stochastic validation. No automatic paper or live promotion is added.

## Forward evidence

The existing append-only forward ledger now has an outcome table. Signals can mature causally at 1h, 4h and 24h after those horizons have actually elapsed.

```bash
python -m crypto_ai_swing.cli forward-mature --horizons 1,4,24
python -m crypto_ai_swing.cli forward-report
```

The proactive runtime also matures old observations from the already-loaded 1h frames, avoiding additional market-data calls during a normal cycle.

## Execution authority

Direct Bitvavo order submission remains disabled. Round 4 does not grant live authority.
