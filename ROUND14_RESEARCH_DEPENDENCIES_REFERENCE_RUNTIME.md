# v0.14 Research Dependencies + Isolated Reference Runtime

This round removes two ambiguous states from the research stack.

## 1. Native campaign prerequisites are explicit

`multi-alpha-v2` depends on canonical Sjagil/crypto evidence artifacts and daily datasets. A missing component report is no longer labelled as generic infrastructure failure.

The dependency preflight checks:

- `output/lab/reports/absolute_momentum_campaign_v1.json`
- `output/lab/reports/residual_reversal_campaign_v1.json`
- canonical 1d Parquet for BTC-EUR, ETH-EUR, SOL-EUR and LINK-EUR

When evidence is missing, the tournament returns `BLOCKED_PREREQUISITE_EVIDENCE` and exposes the exact native repair command. It does not silently run research campaigns, generate fake reports or bypass historical trial accounting.

Current repair commands are native Sjagil/crypto commands:

```bash
cd "$CRYPTO_REPO_PATH"
python main.py lab campaign run --name absolute-momentum-v1 --yes
python main.py lab campaign run --name residual-reversal-v1 --yes
```

Then rerun:

```bash
cd /Users/ayoubalhari/Downloads/crypto-ai-swing-layer
source .venv/bin/activate
export CRYPTO_REPO_PATH="/Users/ayoubalhari/Downloads/crypto"
crypto-swing native-alpha-tournament
```

For a preflight without running the tournament:

```bash
python scripts/native_alpha_dependencies.py multi-alpha-v2
```

## 2. Reference environments are audited as isolated runtimes

`python scripts/reference_doctor.py` checks each configured reference repository against its dedicated `.venvs/...` interpreter. The probe runs in the reference interpreter through a subprocess. It never injects that environment into the swing layer process.

It also reports `.venvs` directories that are not bound to a configured reference. This intentionally surfaces local environments such as `stocks` without inventing a repository URL or silently enabling a donor stack.

Use:

```bash
python scripts/reference_doctor.py
python scripts/reference_doctor.py --enabled-only
```

## Safety boundary

- Sjagil/crypto remains the source of truth for provider data, research evidence, portfolio/risk primitives and execution authority.
- Reference engines are challengers and cross-checkers only.
- No reference process may submit a Bitvavo order.
- No missing research artifact is synthesized.
- No live authority, canary authority or automatic promotion is added in this round.
