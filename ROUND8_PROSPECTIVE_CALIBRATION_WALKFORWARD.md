# Round 8: prospective edge calibration and temporal research robustness

Version: 0.9.0

This round targets the two largest evidence gaps exposed after Round 7.

## Prospective canary gate

The live canary approval path now requires real causal 4h forward BUY evidence.
The default minimum is 30 unblocked BUY outcomes across at least 5 markets and
at least 72 hours of observation span. Mean realized return must be positive
and the positive-return rate must be at least 50 percent.

Passing this gate only makes the fixed EUR 10 canary eligible. It never enables
autoscaling, leverage, shorting, withdrawals, or automatic live promotion.

`edge-calibration-status` exposes the exact sample, breadth, time-span,
predicted-versus-realized calibration error, and blockers.

## Research walk-forward hardening

Cold-start research now uses three sequential development-only temporal windows
before any final holdout is evaluated. The candidate grid is expanded from 20
to 38 pre-registered classical variants, but the final holdout remains limited
to a bounded shortlist.

No strategy family can consume more than two finalist slots by default. This
prevents one parameter neighborhood from using the entire holdout budget.

Final test data remains excluded from candidate selection. Zero survivors is a
valid result and never relaxes the promotion gates.

## Canonical economics

The native Sjagil/crypto research factory still requires immutable canonical
paper/live execution episodes. Round 8 does not fabricate P0.5 family episodes
from backtests or shadow outcomes.

## Operational policy

Automatic live promotion remains disabled. The fixed live canary remains
fail-closed until prospective evidence and the existing crypto execution
authority preflight both pass.
