# v0.25.0 Cost-Aware Edge Calibration + Exit Efficiency

Purpose: turn raw signal scores into a prospective, transaction-cost-aware
research gate without pretending that current heuristic scores are calibrated
profitability estimates.

## Core additions

- Hierarchical empirical-Bayes shrinkage across deterministic context segments.
- Segments include breakout state, strategy family, MTF bucket, technical bucket,
  spread bucket and overextension state.
- Current estimates use only matured causal forward outcomes.
- Qualification uses expanding sequential validation: each historical OOS row is
  gated using only rows that matured before it. The row can never select itself.
- Normal and stressed net returns are both required to be positive.
- Bayesian posterior and canonical stationary-bootstrap Monte Carlo + Dirichlet
  concentration stress remain fail-closed.
- Exit-efficiency diagnostics separate entry edge from path opportunity by using
  MFE, realized return and round-trip cost. This does not invent an executable
  exit rule.
- The strategy challenger catalog is expanded with registered Donchian-55,
  stronger MTF, spread and no-overextension hypotheses. Every new hypothesis
  increases the known-trial count and still requires a future frozen prospective
  test before becoming a research champion.
- ResearchEdgeManager v3 requires both its own holdout evidence and qualified
  net-edge calibration before any paper/shadow meta score can influence a
  signal. Live influence remains disabled.

## Safety and authority

Spot-only/long-only constraints are untouched. No leverage, shorts, withdrawals,
live authority, direct exchange authority, automatic model promotion or order
submission is added by this round.

## Interpretation

A positive MFE distribution is not the same as a realizable trading edge. The
exit-efficiency report is diagnostic. An exit rule must itself be frozen and
validated prospectively before it can influence paper/shadow decisions.

The design follows two principles from the supplied finance ML reference:
model selection should be assessed out of sample, and portfolio/trading rewards
must be risk- and cost-adjusted rather than based on raw prediction quality.
