# v0.27.0 Prospective Swing Entry Selector

This round turns abstention into a first-class research action.

Core contract:

1. Historical decision `t` may use only labels with `matured_at <= t`.
2. Full normal and stressed transaction costs remain deducted.
3. 24h/72h/168h are evaluated as swing horizons. No 72h/168h evidence is
   manufactured when no unblocked BUY observations have matured.
4. High-dimensional MTF context is regularized with Ledoit-Wolf covariance,
   symmetric eigendecomposition and a factor-space multi-response ridge solved
   with linear systems rather than an explicit inverse.
5. The selector predicts normal-net return, stressed-net return, MFE and
   absolute MAE, while Mahalanobis distance rejects out-of-distribution states.
6. A trade candidate must pass expected net, stressed net, MFE/cost,
   MFE/MAE, geometry, OOD and stability gates.
7. When evidence is weak, the action is ABSTAIN. Trade frequency is not an
   optimization target.
8. Qualification is based only on delayed-label sequential OOS selected
   outcomes, Bayesian evidence and the canonical stationary-bootstrap /
   Dirichlet stochastic validator.
9. Paper/shadow influence is possible only after qualification.
10. Live influence and automatic live promotion remain false.
