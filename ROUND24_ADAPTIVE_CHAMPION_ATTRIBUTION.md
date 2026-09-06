# v0.24 Adaptive Champion/Challenger + Performance Attribution

## Purpose

v0.24 converts the research stack from a set of advisory scores into a
stateful, cost-aware learning control plane. It does not claim that a positive
edge exists. It measures where return is coming from, suppresses unavailable or
known-unqualified agent heads, freezes strategy challengers before prospective
testing, and only permits paper/shadow meta influence after independent gates
pass.

## Data flow

15m causal signal observation -> 1h/4h/24h forward maturation -> 4h cost-aware
attribution -> stateful strategy challenger lab -> adaptive agent-of-agents ->
research promotion registry.

## Attribution

The engine reports gross, normal-cost and stressed-cost results; tail loss,
payoff ratio and profit factor; MFE/MAE asymmetry; market, strategy-family and
breakout-state cohorts; component Spearman IC; high-vs-low bootstrap lift; and
Bayesian posterior probabilities. Future paper BUY events retain the exact
TradeIntent context so closed paper trades can be attributed without heuristic
nearest-signal joins.

## Agent manager

Known-unqualified supervised heads are excluded rather than assigned a neutral
0.5 score. Unqualified RL receives zero availability. Component weights are
learned from chronological prospective outcomes with coverage checks,
non-negative trust, Bayesian and stochastic holdout gates, a 35% component cap,
and no live authority.

## Champion/challenger

The candidate catalog is fixed and registered as known trials. Discovery and
validation occur before a freeze timestamp. Once frozen, only later
observations count toward the prospective test. Tests require normal and
stressed positive net means, a minimum positive fraction, Bonferroni-adjusted
Bayesian probability, positive posterior p05, positive uplift versus baseline,
72h prospective span, and canonical stationary-bootstrap + Dirichlet passes.
Failed challengers are retired. Champions are research-only and immutable while
a new challenger is collected.

## Safety invariants

- spot/long-only constraints unchanged
- no leverage/shorting change
- no withdrawal change
- no direct exchange authority
- no automatic live promotion
- no live decision influence from v0.24
- canonical Sjagil/crypto stochastic validator reused
- canonical 15m forward evidence remains the source of outcome truth
