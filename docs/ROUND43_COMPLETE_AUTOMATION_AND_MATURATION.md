# Round 43 — Complete automation, evidence maturation and bounded optimization

Round 43 turns the existing Round 42 control plane into a persistent research
and shadow/paper learning service. It deliberately does not add a second
exchange, risk, sizing, cost or execution implementation.

## Canonical ownership

`Sjagil/crypto` remains authoritative for market data, account state,
reconciliation, risk, Kelly/sizing, costs, execution, protective orders,
kill-switch behavior, live authority and Shariah eligibility.

`crypto-ai-swing-layer` remains responsible for AI, research, prospective
evidence, attribution, strategy/model challengers, browser/RSS intelligence and
orchestration.

## Automated loop

The persistent Round 43 controller runs the existing `UnifiedAutonomyRuntime`
and adds four bounded control functions around it:

1. Mature causal forward outcomes from canonical crypto market data.
2. Diagnose evidence, data, agent and research drift.
3. Create evidence-triggered optimization directives.
4. Trigger bounded ChiefAgent research/training and selector Optuna only when
   the evidence delta and cooldown contracts permit it.

The underlying Round 42 runtime continues to automate market cycles,
attribution, strategy lab refreshes, swing geometry, entry selection, edge
management, supervised training, RL training, strategy DNA generation,
canonical research campaigns and browser/RSS research.

## Evidence stages

Round 43 exposes a research maturation stage:

- `NO_EVIDENCE`
- `COLLECTING`
- `DISCOVERY_READY`
- `OPTIMIZATION_READY`
- `RESEARCH_CHAMPION`
- `OPTUNA_QUALIFIED_RESEARCH`

These labels do not grant paper or live authority. Existing prospective,
promotion and canonical live gates remain authoritative.

## Optimization policy

Heavy work is condition-driven rather than run continuously:

- strategy research requires enough discovery evidence plus new matured
  primary-horizon outcomes;
- exact research requires enough complete 24/72/168h observations and a new
  evidence delta;
- selector Optuna requires a complete-horizon calibration population and a new
  evidence delta;
- retraining can be forced when an existing agent is missing, expired or in
  error;
- every heavy action has a cooldown.

Search spaces may tighten or select research parameters, but Round 43 is not
allowed to lower qualification thresholds, widen risk, grant live authority or
promote automatically to live.

## Persistent macOS service

After tests pass, install one LaunchAgent in shadow mode:

`python scripts/round43_launchd.py install --mode shadow`

Inspect it with:

`python scripts/round43_launchd.py status`

Inspect the research/evidence state with:

`python scripts/round43_status.py --compact`

Remove the service with:

`python scripts/round43_launchd.py uninstall`

Only `shadow` and `paper` are accepted by the Round 43 service. `live` is
intentionally rejected.

## GitHub scheduled research

`round43-public-research-audit` remains a stateless CI research audit. GitHub
scheduled runners must not be counted as persistent forward evidence. The local
persistent service owns the accumulating prospective evidence database.

## Performance statement

Round 43 optimizes for robust prospective net performance after costs and
benchmark-relative evidence. It does not guarantee positive returns.

## Runtime cache hygiene

The persistent macOS runtime overrides accidental Windows-style Hugging Face
cache paths and uses the normal macOS user cache under `~/.cache/huggingface`.
This prevents Windows drive paths from being created as literal repository
directories. These caches are runtime artifacts and are never research evidence
or Git inputs.

## Unattended active-swing operation

The persistent local service uses a five-minute control loop. Forward evidence
is still deduplicated into closed 15-minute decision buckets, while expensive
research and training remain due-driven. The macOS LaunchAgent wraps the
runtime in `caffeinate -s`, so system sleep is suppressed while AC power is
available.

Expected recurring work:

- market/universe and prospective signal collection: every control cycle;
- edge/evidence/attribution refresh: approximately every 15 minutes;
- strategy challenger lab: hourly;
- supervised agent retraining: every 4 hours or immediately when missing/error;
- strategy DNA, native tournament and stage-0 research: every 6 hours;
- PPO RL retraining and exact canonical research/walk-forward: every 24 hours;
- browser/RSS research: approximately every 15 minutes;
- selector Optuna: evidence-triggered after complete delayed horizons mature.

Gradient backpropagation is used where the underlying learner uses it, for
example the PPO MLP policy. Tree/boosting models use their native fitting
algorithms rather than artificial backpropagation.

No task in this service grants live authority. Transition to real Bitvavo
execution remains a separate canonical canary/live-authority decision.

## Realized active-swing economic feedback

Round43 pairs paper BUY and SELL events into closed trades and
attributes cost-adjusted PnL by strategy. It measures net return,
expectancy, profit factor, payoff ratio, R-multiple, holding time,
MFE/MAE, realized drawdown, and compounded trade-path return.

The evaluator also joins delayed 4h/24h/72h/168h prospective
evidence. The 4h horizon is treated mainly as an early entry-quality
diagnostic; 24h, 72h, and 168h carry most of the active-swing
follow-through weight.

Weak realized economics may trigger bounded retraining and strategy
research. Negative strategy cohorts are only research quarantine
candidates. This layer cannot submit exchange orders, relax
thresholds, widen risk, arm live authority, or directly promote a
model.
