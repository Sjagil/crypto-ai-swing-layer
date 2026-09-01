# Round 6: 25-market universe, agent qualification and cold-start research

Round 6 expands the runtime from three hard-coded assets to a dynamic 25-market
Bitvavo EUR spot universe. The universe is ranked from public 24h liquidity,
filtered through the canonical Sjagil/crypto Shariah eligibility layer, excludes
stablecoins and leveraged-token patterns, and is refreshed every six hours.

To keep a 60-second swing cycle practical, all 25 markets receive a cheap closed
1h screen and only the strongest eight plus any open positions receive full
15m/1h/2h/4h/1d/1w, orderbook, trades, CVD, NLP and agent enrichment.

The first three-market model had weak OOS evidence. Round 6 therefore prevents
an unqualified SHADOW model from vetoing or influencing shadow decisions. It
still records predictions. Training now uses the full runtime universe and a
validation-only model/threshold tournament before the untouched test partition
is evaluated. No model is promoted automatically.

Canonical economics currently has zero closed episodes, which means the native
P0.5 reset campaign is not semantically applicable. The native bridge now
reports a cold-start state instead of throwing. A separate bounded cold-start
research runner tests five causal long-only strategy families across the
25-market universe with next-bar execution and stressed transaction costs. Its
results remain RESEARCH_ONLY because current-liquidity universe selection is
not a point-in-time historical universe.
