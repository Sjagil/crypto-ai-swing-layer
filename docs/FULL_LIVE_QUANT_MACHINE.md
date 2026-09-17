# Full-Live Quant / AI Machine v1

This bundle removes the **tiny execution-validation capital clamp** while keeping
canonical fail-closed execution, account reconciliation, native protective stops,
kill switches, cost gates and portfolio risk controls.

It does **not** mean “buy everything” and it does not bypass the canonical risk
manager. A trade must still have a positive after-cost edge and pass the full
portfolio/risk/execution chain.

## Runtime architecture

### Raspberry Pi: execution/inference node

Every 60 seconds:

1. refresh the executable universe and closed candles;
2. evaluate 1w/1d macro, 4h/2h trend, 1h setup and 15m execution context;
3. calculate technical, volume, order-flow, spread, CMC/news and regime context;
4. run the existing supervised agent, evidence-qualified TCN+GRU champion and RL;
5. combine qualified probability models with evidence-weighted model averaging;
6. build a trade signal and estimate expected edge;
7. subtract canonical fees, spread, slippage and execution allowances;
8. calculate canonical portfolio risk, fractional Kelly and correlation sizing;
9. submit the real Bitvavo spot order through `Sjagil/crypto`;
10. place/verify native protection and reconcile the account after submission.

No leverage, margin, shorting, derivatives or withdrawals are enabled by this
patch.

### Mac: research/training node

The Mac polls every five minutes but only launches expensive work when due:

- supervised + HPO retraining: existing AgentManager cadence (4h by default);
- TCN+GRU retraining: 6h;
- canonical strategy factory / indicator combinations: 6h;
- RL retraining/HPO: existing 24h cadence;
- qualified challengers are automatically compared with the current champion;
- only a challenger that improves the evidence gate receives live model influence.

This makes learning continuous without allowing a newly-fit model to mutate live
capital blindly.

## Quant sizing

The actual size is governed by the canonical risk engine. The conceptual chain is:

```text
calibrated probability
-> uncertainty shrinkage
-> reward/risk ratio
-> fractional Kelly
-> risk-tier ceiling
-> canonical live risk ceiling
-> stop-distance position sizing
-> single-position/exposure/cash constraints
-> correlation multiplier
-> cost/net-edge gate
-> final notional
```

For a calibrated win probability `p` and reward/risk ratio `b`, full Kelly is:

```text
f* = (b p - (1-p)) / b
```

The stack uses a fraction of Kelly and shrinks uncertain probabilities toward 0.5
before sizing. A simplified risk-based notional is:

```text
risk_eur = equity_eur * approved_risk_fraction
notional_eur ~= risk_eur / stop_distance
```

The actual implementation takes the minimum of the canonical risk size, Kelly
size, correlation-adjusted size, available cash and portfolio limits.

Expected edge is also net of real execution costs:

```text
net_edge = expected_edge
           - 2*fee
           - spread
           - 2*slippage
           - failed_execution_allowance
           - partial_fill_impact
```

## Model averaging

The book *Machine Learning in Finance* motivates Bayesian model averaging. This
bundle uses the same core idea but labels the implementation accurately as
**evidence-weighted model averaging**, because the weights are based on OOS
metrics rather than exact marginal model likelihoods:

```text
p_ensemble = sum_i w_i * p_i
w_i = softmax(evidence_score_i / temperature)
```

Evidence includes OOS AUC, Brier score, selected after-cost expectancy, market
breadth and sample size. An unqualified model receives no live weight.

## TCN + GRU

`src/crypto_ai_swing/models/tcn_gru.py` is based on the final causal implementation:

- left-padded causal TCN;
- dilations for multiple temporal scales;
- unidirectional GRU;
- market-isolated sequences;
- train-only imputation/scaling;
- purged chronological train/validation/test;
- dedicated probability calibration;
- separate threshold selection;
- untouched final test;
- historical persisted OHLCV for training;
- hot-reloaded live champion pointer.

The governor promotes a TCN+GRU candidate only when the final OOS test has enough
observations and market breadth, positive conservative after-cost expectancy and
acceptable classification/calibration statistics.

## Continuous strategy research

The canonical `Sjagil/crypto` research stack remains the strategy factory. It
already contains technical feature/indicator infrastructure and strategy research;
the Mac research process now invokes it repeatedly with exact evaluation enabled.
This avoids creating a second inconsistent backtester in the swing layer.

Research may explore many parameterizations and models. Live capital does not see
a strategy until it has passed chronological/OOS/cost/evidence checks.

## Install order

The easiest path is the single self-contained installer
`apply_full_live_quant_machine_standalone.py`. It writes the model/runtime/research
modules and deployment helpers into the swing repo, and patches both repositories.
No archive is required.

1. Start from clean working trees.
2. Run the standalone installer against both repos.
3. Commit the **canonical crypto repo first**.
4. Run the generated `scripts/pin_canonical_head.py` so the swing layer locks to
   that exact new canonical commit.
5. Run both test suites and `scripts/verify_full_live_quant_machine.py`.
6. Commit/push the swing branch.
7. Pull the exact commits on the Pi.
8. Activate full-live authority explicitly once with `scripts/activate_full_live.py`.
9. Install/start `deploy/crypto-swing-full-live.service` on the Pi.
10. Run `scripts/run_mac_research_machine.sh` on the Mac.

## Important operational property

The emergency full-live EUR values in the service file are intentionally high
absolute circuit breakers so they do not become another €10/€25 canary. They are
**not targets**. Portfolio-scale sizing is still produced by RiskManager + Kelly +
correlation + cash/exposure + after-cost edge logic.
