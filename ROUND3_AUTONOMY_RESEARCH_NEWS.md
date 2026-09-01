# Round 3: autonomy research news evidence

This round connects the active swing loop to canonical research and causal news surfaces in `Sjagil/crypto` without creating a second exchange authority.

## Added

- canonical `scrapers.rss.collect_registered_feeds` news collection with causal timestamps
- FinBERT input from the canonical RSS collector plus persisted artifacts
- robust 24h quote-volume normalization and a market-data audit command
- explicit higher-timeframe causal alignment to the primary candle close
- append-only forward evidence ledger for shadow/paper/live observations
- native research bridge to `research.research_factory`, `research.backtest`, `research.optimization`, `research.stochastic_validation`, `research.classical_strategy_factory`, `research.autonomous_rd`, `research.combinatorial_lab`, and `research.alpha_discovery`
- deterministic strategy factory plan command backed by `Sjagil/crypto`
- direct Bitvavo execution disabled in the generic execution router
- live config fail-closed until the existing crypto runtime submission boundary is mapped exactly

## Commands

```bash
python -m crypto_ai_swing.cli news-scan
python -m crypto_ai_swing.cli market-data-audit BTC-EUR
python -m crypto_ai_swing.cli research-doctor
python -m crypto_ai_swing.cli strategy-factory-plan --trials 2000
python -m crypto_ai_swing.cli forward-status
python -m crypto_ai_swing.cli proactive --mode shadow --once
```

## Deliberately not enabled

- automatic paper promotion
- automatic live promotion
- direct live exchange submission

The next round should map exact candidate evaluation/backtest callables from the native research doctor, add candidate lifecycle orchestration, outcome maturation for forward evidence, portfolio correlation and daily-loss guards, and finally map the existing `Sjagil/crypto` execution/governance submission path.
