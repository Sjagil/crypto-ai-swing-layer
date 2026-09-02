# Round 11 Quant, Probability and MTF hardening

This round makes the swing layer use explicit quantitative evidence rather than
silently treating heuristic scores as probabilities or expected returns.

Key changes: richer causal distribution/risk features; Brier skill, log loss,
average precision and calibration error; native Sjagil/crypto White Reality
Check, Hansen SPA, PBO and DSR evidence for the classifier/threshold tournament;
native HAC effective sample size; hierarchical 1W/1D -> 4H/2H -> 1H -> 15M ->
orderflow decision roles; actual observation-time closed-bar clipping; and
forward maturation aligned to the configured 15m execution timeframe.

The previous forward SQLite file is retained as historical evidence. Round 11
starts `forward_v3.sqlite` so 1h-reference outcomes are never mixed with new
15m-execution-reference outcomes.

No live authority is enabled and automatic promotion remains disabled.
