# v0.15.0 Reference Provisioning + Cross-Engine Mechanics Parity

This round moves reference frameworks from passive configuration into isolated,
invokable research workers.

Added:

- repeatable isolated reference provisioning
- source checkout provisioning for enabled reference repositories
- canonical cost-model reuse from `Sjagil/crypto`
- one canonical OHLCV input and signal schedule
- native fixed-size mechanics replay
- VectorBT isolated replay worker
- NautilusTrader isolated bar-backtest worker
- strict input and signal hashes
- parity reporting with explicit tolerances
- no automatic alpha promotion
- no live decision authority
- no exchange-order authority

The initial EMA state machine validates mechanics only. It does not claim alpha.
