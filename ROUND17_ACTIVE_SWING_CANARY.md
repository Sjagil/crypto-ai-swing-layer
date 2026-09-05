# v0.17 Active Swing Live Canary

This round prioritizes real proactive swing execution rather than adding more model
complexity first.

It is not high-frequency trading. The runtime keeps closed 1h setup bars, 15m
execution timing, 2h/4h trend context and 1d/1w regime context. A 60-second
supervisor loop only monitors for new closed-bar decisions and manages open
positions.

The execution-validation canary can let a naturally generated strong swing BUY
reach the canonical Sjagil/crypto live preflight even before prospective alpha
evidence is mature. It is explicitly not alpha evidence and cannot authorize
autoscaling or automatic model promotion.

The existing native canary limits stay at EUR 10 maximum order, EUR 10 total
exposure, one position, one new order per day and EUR 1 maximum risk per trade.
Canonical Sjagil/crypto remains the sole authority that can accept or reject and
submit a real Bitvavo order.

Existing live exits remain canonical and proactive: stop loss, take profit and
trailing stop are evaluated every runtime cycle and route through
submit_swing_layer_exit.
