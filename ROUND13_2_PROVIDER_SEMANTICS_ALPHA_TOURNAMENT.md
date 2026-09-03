# v0.13.2 Provider-Semantic Data Quality + Native Alpha Tournament

Bitvavo may legitimately omit a candle when an interval has no trades. A
missing timestamp is therefore not automatically data corruption.

This release separates:

- local candles recoverable from a fresh provider refetch
- provider-absent intervals with no finer-timeframe candles
- provider cross-timeframe inconsistencies
- raw grid sparsity

Confirmed provider-absent / no-finer-candle intervals are treated as
provider-semantic zero-trade intervals for data integrity. They remain visible
as a liquidity-density diagnostic and are never forward-filled.

No raw candle is synthesized and no cross-provider candle is used to fill a
Bitvavo gap.

The release also expands the directly reusable native alpha catalog and adds a
research-only tournament report. It does not create an artificial combined
score and does not promote any strategy to live.
