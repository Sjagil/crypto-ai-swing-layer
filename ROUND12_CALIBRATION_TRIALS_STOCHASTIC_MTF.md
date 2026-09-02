# Round 12: calibration, cumulative trials, stochastic evidence and MTF conflicts

Round 12 addresses the failures exposed by Round 11 instead of loosening gates.

Probability calibration now uses a chronological calibration-fit segment and a
separate purged model/threshold-selection segment. Raw, sigmoid and isotonic
calibrations are evaluated before the untouched final test.

The alpha search now maintains a persistent cumulative trial ledger. Identical
reruns are deduplicated. Materially new datasets or search spaces increase the
known-trial denominator supplied to native Deflated Sharpe calculations.

Selected OOS net-return paths receive native Sjagil/crypto HAC effective sample
size, stationary bootstrap Monte Carlo, drawdown/terminal-loss tail checks and
Dirichlet chronological concentration stress.

The MTF gate now has an explicit weekly/daily conflict policy. A strongly
bearish weekly state cannot be neutralized by a weakly positive daily score.

Current-universe training is still explicitly not point-in-time. Formal alpha
qualification therefore remains fail-closed while that requirement is enabled.
Shadow evidence collection continues normally.

No live authority is changed and automatic promotion remains disabled.
