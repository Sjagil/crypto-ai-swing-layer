# Round 38: Prospective Selector Runtime Ledger

Round 38 makes the existing v0.27 prospective swing-entry selector auditable at
runtime without granting it execution authority.

## What changes

The selector already participates in the research edge-manager gate. Round 38
adds a quality-gated runtime wrapper around that existing selector instead of
building a second selector.

Before a paper/shadow selector decision can pass, the wrapper verifies:

- the cached selector policy exists and is qualified;
- the policy is fresh enough for runtime use;
- selector feature coverage meets the configured threshold;
- the core MTF, challenger and universe-screen sections are present;
- the context came through the canonical `Sjagil/crypto` data path.

A failed check produces `ABSTAIN`. The wrapper cannot convert an abstention into
a buy.

## Decision ledger

Every runtime evaluation is written idempotently to:

`output/crypto_ai_swing/research/entry_selector/runtime_decisions.sqlite`

The ledger stores hashes and decision diagnostics, not raw context or provider
credentials. Re-evaluating the same market/decision/model/context produces the
same decision id and does not duplicate the row.

## Authority contract

Round 38 remains advisory only:

- `authority = ADVISORY_ONLY`
- `live_decision_influence = false`
- `automatic_live_promotion = false`
- `orders_generated = 0`
- `orders_submitted = 0`

The canonical `Sjagil/crypto` execution authority remains the only live-order
path.

## Install

After Round 37 has been applied:

```bash
unzip -o ~/Downloads/crypto-ai-swing-layer-round38-selector-runtime-ledger.zip \
  -d /Users/ayoubalhari/Downloads/crypto-ai-swing-layer
cd /Users/ayoubalhari/Downloads/crypto-ai-swing-layer
python scripts/install_round38.py
python -m compileall -q src scripts tests
ruff check --select F,E9 src scripts tests
python -m pytest -q tests/test_round38_selector_runtime.py
python -m pytest -q
python scripts/selector_runtime_status.py
git diff --check
```
