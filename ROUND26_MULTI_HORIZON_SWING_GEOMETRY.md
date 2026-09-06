# v0.26 Multi-Horizon Swing Geometry + Advanced Linear Algebra

v0.26 changes economic qualification from a short 4h-only view to a causal
active-swing view while retaining 4h attribution as an entry/exit diagnostic.
It does not discount or ignore transaction costs. Instead it measures full
fees, spread and slippage relative to the magnitude of a multi-day swing.

## New research horizons

The forward ledger now requests 1h, 4h, 24h, 72h and 168h outcomes. Existing
causal semantics remain unchanged: the first closed 15m execution bar at or
after the actual observation is the reference. 72h and 168h evidence will
initially be sparse or empty and is allowed to mature naturally.

The 24h horizon becomes the default NetEdge/agent-manager anchor. 4h remains
valuable for MFE/MAE and entry timing diagnostics. SwingGeometry evaluates
24h/72h/168h and selects a historical horizon only from labels that had matured
before the historical decision being evaluated.

## Cost policy

Costs are never relaxed. A 60 bps round-trip still subtracts 60 bps from the
observed gross return. What changes is the diagnostic interpretation. A 60 bps
cost is 30% of a 2% move, 20% of a 3% move and 12% of a 5% move. Qualification
requires positive normal and stressed net returns after full costs, plus a
bounded cost fraction of mean winner magnitude.

## Swing asymmetry

The old 55% winning-trade requirement was scalp-like and can reject valid
trend-following distributions. v0.26 permits a lower positive-rate floor but
adds stronger distributional gates: positive net mean, positive stressed mean,
profit factor, payoff ratio, Bayesian P(mean>0), positive posterior lower bound,
and canonical stationary-bootstrap/Dirichlet validation. A 40% win rate can
only survive if winners materially dominate losers and the OOS evidence is
robust.

## Advanced linear algebra

SwingGeometry builds a causal feature matrix from the multi-timeframe state and
available qualified/advisory components. It uses robust centering/scaling,
Ledoit-Wolf covariance shrinkage, symmetric eigendecomposition with eigenvalue
flooring, effective-rank diagnostics, whitening, a regularized Fisher direction
solved with linear systems rather than explicit matrix inversion, bootstrap
cosine stability, principal-factor loadings and Mahalanobis out-of-distribution
distance.

The geometry layer is not a new live authority. It remains research/paper/shadow
only and must itself qualify before it can contribute to future shadow influence.

## v0.25 correctness fixes included

This round also removes the unused `math` import and restores backward
compatibility for integrations/tests that construct `ResearchEdgeManager` with
`__new__` and therefore do not yet have a NetEdgeCalibrator or SwingGeometry
collaborator.

## Authority

No leverage, shorts, withdrawals, automatic live promotion, live risk caps or
canonical Bitvavo execution authority are changed. No order is submitted by
this upgrade.
