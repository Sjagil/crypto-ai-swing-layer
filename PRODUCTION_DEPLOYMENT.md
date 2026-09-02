# Production runtime deployment

The project can be deployed as a persistent macOS runtime before real-money
authority is enabled.

## Modes

- `shadow`: full live market ingestion, research, agents, forward evidence and
  supervisor scheduling without exchange order authority.
- `paper`: paper execution mode.
- `live`: real exchange path, additionally blocked by prospective evidence,
  native canary preflight and `CRYPTO_SWING_CANARY_EXECUTE=YES`.

Production installation defaults to `shadow`.

## One-time setup

```bash
cd /path/to/crypto-ai-swing-layer
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[crypto-runtime,dev]'
export CRYPTO_REPO_PATH=/path/to/crypto
python -m pytest -q
```

## Start manually

```bash
export CRYPTO_REPO_PATH=/path/to/crypto
export CRYPTO_SWING_PRODUCTION_MODE=shadow
./scripts/run_production.sh
```

Only one supervisor is allowed by the runtime lock.

## Install as macOS LaunchAgent

```bash
export CRYPTO_REPO_PATH=/path/to/crypto
export CRYPTO_SWING_PRODUCTION_MODE=shadow
./scripts/install_launchd.sh
```

The LaunchAgent starts on login and restarts after abnormal process exits.

## Health

```bash
./scripts/healthcheck.sh
./scripts/runtime_status.sh
```

Default maximum heartbeat age is 900 seconds so scheduled training or research work does not create false stale-heartbeat failures.

## Logs

```text
output/crypto_ai_swing/logs/launchd.stdout.log
output/crypto_ai_swing/logs/launchd.stderr.log
output/crypto_ai_swing/supervisor/heartbeat.json
output/crypto_ai_swing/supervisor/state.json
```

## Live startup

Do not install the LaunchAgent directly in live mode.

A manual live startup uses all existing gates:

```bash
export CRYPTO_REPO_PATH=/path/to/crypto
export CRYPTO_SWING_PRODUCTION_MODE=live
export CRYPTO_SWING_CANARY_EXECUTE=YES
./scripts/run_production.sh
```

Before the supervisor starts, `live_startup_gate.py` requires:

1. prospective canary evidence `eligible=true`;
2. current `live-canary-preflight` `ready=true`;
3. explicit execution environment authorization.

Nothing in this deployment package approves the canary, enables autoscaling,
increases the EUR 10 cap, enables leverage or shorting, or enables withdrawals.
