# Research gates

A candidate is not promoted because one backtest looks good.

Required evidence chain:

1. Fast discovery
2. Parameter search on development data
3. Purged walk-forward
4. Cost and slippage stress
5. Statistical validation
6. Cross-engine validation
7. Forward shadow

Important invariants:

- no random train/test shuffle for time series
- purge covers label horizon
- higher-timeframe data is lagged
- current CMC rankings are not projected backward
- no forward-filled missing market bars
- costs are applied before promotion
- multiple-testing risk is measured
- untouched test data is not used for tuning
- shadow success does not automatically enable live trading
