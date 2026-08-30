# Existing crypto repository integration

The actual repository can evolve, so the bridge uses capability discovery instead of hard imports.

Run:

```bash
python -m crypto_ai_swing.cli doctor
```

The generated `output/crypto_ai_swing/doctor.json` records which known paths currently exist.

The safest integration order is:

```text
existing data artifacts
  -> this layer reads
  -> this layer writes TradeIntent JSON
  -> existing repo consumes intent
  -> existing repo performs risk/reconciliation/readiness
  -> existing repo executes
```

Only configure `config/integrations.yaml -> execution_adapter.command` after the existing repo has an explicit command that accepts an intent file and still enforces its own safety gates.

Do not point the command directly at a raw Bitvavo order endpoint.
