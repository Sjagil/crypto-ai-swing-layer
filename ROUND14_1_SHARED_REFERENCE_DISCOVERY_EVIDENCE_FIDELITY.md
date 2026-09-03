# v0.14.1 Shared Reference Discovery + Native Evidence Fidelity

This patch hardens two diagnostics exposed after the v0.14 runtime was exercised
against the real workspace.

## Shared reference discovery

Reference environments are no longer assumed to live only below the swing-layer
checkout. The runtime now searches:

- explicit `--venv-root` values
- `CRYPTO_SWING_REFERENCE_VENV_ROOT`
- `<project>/.venvs`
- `<project-parent>/.venvs`

Repository source roots follow the same model through `--repo-root` and
`CRYPTO_SWING_REFERENCE_REPO_ROOT`.

A Python executable is not considered a healthy package runtime when its
declared probe module is missing. Source-backed challengers can still be marked
ready when an isolated interpreter and the configured repository source are both
present.

Reference processes remain isolated challenger workers. They have no exchange
order authority.

## Native evidence fidelity

The native campaign bridge now preserves the field names actually emitted by the
canonical `Sjagil/crypto` research campaigns, including:

- `primary_strategy_id`
- `primary_policy_name`
- `generated_trial_count`
- `registered_unique_trials`
- `total_known_trials`
- `pbo`
- `paper_candidates`
- `primary_positive_research_lead`

Forward observer summaries are compacted into an evidence digest without turning
diagnostic returns into promotion evidence. The digest explicitly reports closed
observations, rebalances, high-volatility regime coverage, formal-gate evaluation
state and best/worst diagnostic forward returns.

## Tournament classification

The tournament no longer collapses every non-promoted campaign into one generic
bucket. Historical gate failure while forward data is still accumulating is
reported separately from a campaign whose historical gates passed and is merely
waiting for prospective evidence.

No synthetic score is introduced. No research threshold is relaxed. No paper,
canary or live authority is granted.
