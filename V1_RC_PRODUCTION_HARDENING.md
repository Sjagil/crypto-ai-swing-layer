# v1.0.0rc1 Production Hardening

This release candidate hardens the boundary between research decisions and
real exchange mutations.

- Sjagil/crypto remains the only live exchange authority.
- Spot-only, long-only, no leverage/derivatives, no withdrawals.
- Existing EUR 10 canary caps remain unchanged.
- Production preflight checks authority, account truth, reconciliation and
  portfolio visibility without submitting orders.
- A durable journal is written before every live mutation.
- Crash/unknown submission outcomes never trigger automatic resubmission.
- Unresolved BUY operations block new entries.
- BUY and EXIT submissions require post-submit reconciliation.
- EXIT readiness is separate from entry-canary readiness.
- Secrets are redacted from RC artifacts.
- Execution certification requires multiple reconciled real roundtrips.
- Execution certification does not certify profitability.
- Scaling remains disabled until both strategy evidence and execution
  certification qualify independently.
