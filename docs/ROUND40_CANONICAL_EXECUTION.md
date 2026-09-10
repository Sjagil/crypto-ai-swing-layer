# Round 40: Canonical Exchange and Live Execution

Round 40 removes the remaining Bitvavo transport implementation from `crypto-ai-swing-layer`.

- No local Bitvavo REST/HMAC/order client remains.
- Live router submission can only use `Sjagil/crypto:core.swing_layer_live`.
- Live status is projected from canonical authority state.
- Proactive live account truth comes from the canonical account snapshot.
- Shadow/paper accounting stays simulated by default.
- Ownership requires zero legacy exchange migration exceptions.
- Spot-only, no-margin, no-leverage, no-short and no-withdrawal remain fail-closed.

Round 40 does not authorize live trading or bypass evidence, reconciliation, Shariah, risk, kill-switch, canary, or manual authority requirements.

Round 41 routes the AI decision packet through canonical portfolio/risk/Kelly.
