# v0.22 Probabilistic Edge Manager

Goals:
- preserve the existing 15m -> 1h -> 2h/4h -> 1d/1w causal pipeline;
- create richer MTF strategy-family challengers;
- create an agent-of-agents that learns non-negative weights only from matured
  prospective outcomes;
- keep RL at zero influence unless it passes a train/validation/final-test
  tournament plus Bayesian, Monte Carlo and Dirichlet robustness;
- never optimize on the final test set;
- never grant live authority from research output.

A positive return is not guaranteed. The contract is instead: no research
agent is allowed to influence even shadow scoring unless its later holdout
evidence is net-positive and probabilistically robust.
