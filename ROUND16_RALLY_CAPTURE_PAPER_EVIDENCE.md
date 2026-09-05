# v0.16.0 Rally Capture + Paper Evidence Fidelity

v0.16.0 converts the observed September 2026 false-negative lessons into a
bounded research/paper architecture without weakening live authority.

## Included

- Independent continuation/acceleration deep-scan lane.
- SHADOW/PAPER-only soft macro-lag exception.
- Heuristic edge proxy separated from calibrated expected return.
- Paper evidence can bypass only edge-ratio/net-edge gates, never liquidity,
  participation, spread, risk, account, Shariah or live authority gates.
- Versioned canonical fee/slippage baseline reused from `Sjagil/crypto`.
- Forward evidence stores edge source and excludes heuristic proxy edges from
  predicted-vs-realized calibration.
- >35% 24h assets are observed and flagged instead of being made invisible by
  default.
- NLP confidence is capped when no direct-asset evidence exists.
- VectorBT verifies its actual order timestamps and independently recomputes
  frame/signal hashes.
- Nautilus distinguishes fill events from logical orders, independently
  recomputes hashes and exposes position-record/open-position semantics.
- Supervisor unexpected runtime exceptions are persisted.
- macOS launchd scripts install/uninstall a persistent PAPER supervisor.

## Live safety

This round does not enable live submission, automatic promotion, autoscaling,
leverage, shorting, derivatives or withdrawals. A heuristic edge source remains
blocked in LIVE, and canary preflight requires a calibrated/qualified current
BUY intent.
