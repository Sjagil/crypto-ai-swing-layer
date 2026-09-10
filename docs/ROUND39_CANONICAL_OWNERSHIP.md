# Round 39: Canonical Ownership and Production Contract

Round 39 makes `Sjagil/crypto` the explicit canonical infrastructure, quant,
risk and execution authority used by `Sjagil/crypto-ai-swing-layer`.

## Canonical ownership

`Sjagil/crypto` owns exchange/public/private transport, canonical market data,
provider normalization, account truth, reconciliation, trading math, Kelly,
position sizing, cost truth, portfolio/correlation risk, Shariah eligibility,
live execution, protective exits and kill switches.

`crypto-ai-swing-layer` owns AI/ML features, model and agent development,
training, calibration, registries, champion/challenger management, strategy and
quant research, prospective validation, intelligence fusion, decision generation,
attribution and retraining.

## Round 39 invariants

- CI checks out the exact `crypto` commit pinned by `canonical_crypto_engine.json`.
- The native bridge exposes canonical expectancy/Kelly/sizing, portfolio risk and
  Bitvavo execution contracts instead of rebuilding them.
- Signal ensemble weights are read from `config/swing.yaml` rather than the old
  hard-coded 60/20/10/10 split.
- `minimum_model_probability` can veto an entry when qualified ML probability is present.
- A source audit blocks new local exchange/math/risk duplication.
- Existing local Bitvavo call sites remain temporary migration exceptions only.

## Round 40

Round 40 migrates router, control plane, proactive runtime and CLI away from the
local `execution/bitvavo.py` implementation onto canonical `Sjagil/crypto`
execution/account/data interfaces. The local Bitvavo implementation is then deleted.

Round 39 does not bypass existing live, evidence, risk, Shariah, reconciliation,
kill-switch or canary gates.
