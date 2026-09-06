# v0.24.1 Paper Reconciliation Hotfix

Fixes duplicate simulated BUY inventory caused by migration bootstrapping after
a normal paper BUY had already been persisted. New bootstraps are created only
for positions with zero event-ledger inventory. Partial mismatches remain
fail-closed.

Includes a backup-first repair utility that may delete only bootstrap BUY
events when their quantity exactly explains the excess event inventory. It
never deletes ordinary paper BUY or SELL events and never submits an order.

The unified autonomy runtime is also hardened: if the proactive paper ledger
reports RECONCILIATION_REQUIRED, the top-level runtime now reports DEGRADED
instead of silently returning ONESHOT_COMPLETE with an empty errors list.
