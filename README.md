# Crypto AI Swing Layer

A crypto-only AI swing-trading research and shadow-execution layer designed to sit on top of an existing crypto repository that already owns Bitvavo, CoinMarketCap, feeds, scrapers, storage, account state, safety gates and exchange execution.

This project deliberately does not duplicate those integrations. It discovers and consumes them through a narrow bridge, normalizes their data, builds leakage-safe multi-timeframe features, runs deterministic strategies plus ML/RL/forecast challengers, applies portfolio and execution-cost gates, and emits `TradeIntent` artifacts. Live execution is disabled by default.

## Architecture

```text
Existing crypto repository
  Bitvavo REST / WS
  CoinMarketCap
  scrapers / RSS / intelligence
  historical data lake
  realtime market state
  account / reconciliation
          |
          v
crypto_ai_swing.bridge
          |
          v
canonical OHLCV + point-in-time universe
          |
          v
15m execution context
1h trigger
2h / 4h setup
1d / 1w regime
          |
          v
feature factory
  technical
  relative strength
  volatility
  liquidity
  orderflow context
  sentiment/context
          |
          +----------------------+
          |                      |
          v                      v
 deterministic alpha        AI challengers
 trend/pullback              supervised ML
 breakout/retest             Kronos forecast
 momentum                    RL / FinRL research
 volatility expansion
          |                      |
          +----------+-----------+
                     v
              ensemble score
                     |
                     v
        cost + liquidity + edge gate
                     |
                     v
          portfolio / risk governor
                     |
                     v
                TradeIntent
                     |
          +----------+-----------+
          |                      |
          v                      v
     shadow ledger       existing execution adapter
                         paper first, live explicit
```

## What this layer reuses

The discovery bridge knows about candidate paths already used in the crypto codebase, including:

- `src/data_sources/bitvavo_ws_v7.py`
- `src/data_sources/bitvavo_market_data_pro_ws_v7.py`
- `src/data/providers/coinmarketcap_client.py`
- `src/data/providers/historical_provider_capability_matrix.py`
- `scripts/data/run_market_data_ingestion_v7.py`
- `src/runtime/realtime_market_state_v1.py`
- `scripts/runtime/build_realtime_market_state_v1.py`
- `scripts/data_build/build_historical_data_lake_v7.py`
- `scripts/data_quality/validate_historical_data_lake_v7.py`
- `scripts/data_build/build_training_ready_dataset_v7.py`
- `scrapers/rss.py`
- `scrapers/intelligence.py`
- `worldmonitor_data_plane_launcher.py`

They are discovery hints, not hard imports. `doctor` verifies which paths exist on the actual machine.

## Reference repositories

The selected reference stack is intentionally crypto-relevant and role-separated.

Core references:

- NautilusTrader: event-driven replay and execution-model cross-checking
- vectorbt: fast research and signal cross-checking
- Optuna: parameter and model search
- skfolio: portfolio and risk challengers
- Stable-Baselines3 Contrib: RL challenger policies
- Kronos: probabilistic/time-series forecast challenger
- FinRL-Trading: RL research patterns and benchmark workflows

Optional references:

- LEAN: independent cross-engine backtest validation
- MoonDev Trading AI Agents: research-agent patterns only, never execution authority

Excluded by default because they add little to this crypto-specific layer or are broker/equity focused:

- PyBroker
- Qlib
- VeighNa
- vnpy_ib
- Stocks donor repo

References live in `references/` and should use isolated environments under `.venvs/`. Do not mix their `site-packages` into the main environment.

## Quick start

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -r requirements.txt

cp .env.example .env
```

Point the layer at the existing crypto repo:

```bash
export CRYPTO_REPO_PATH="/absolute/path/to/crypto"
```

Run discovery:

```bash
python -m crypto_ai_swing.cli doctor
```

Run a synthetic end-to-end smoke test:

```bash
python -m crypto_ai_swing.cli smoke
```

Clone selected reference repos:

```bash
python scripts/clone_references.py
```

Run a shadow pass from a folder containing canonical Parquet files:

```bash
python -m crypto_ai_swing.cli shadow \
  --data-dir /path/to/crypto/data/processed \
  --pattern "*_1h.parquet"
```

## Install into the existing crypto repository

If you want the package copied into the existing repository rather than kept as a sibling project:

```bash
python scripts/install_into_crypto.py /absolute/path/to/crypto
```

The installer refuses to overwrite existing files unless `--overwrite` is supplied. It copies only this layer's package, configs, scripts, docs and tests.

## Execution policy

Default mode is `shadow`.

The model stack cannot submit an exchange order directly. Strategies produce signals. The portfolio layer produces risk plans. The router produces `TradeIntent`. An existing execution adapter may consume those intents only after the configured safety boundary.

Live mode requires all of the following:

1. `runtime.mode: live` in config.
2. `execution.live_enabled: true`.
3. `CRYPTO_SWING_LIVE_ACK=I_UNDERSTAND`.
4. A configured external execution adapter.
5. Healthy market data and reconciliation inputs.
6. A non-stale intent.
7. Cost and risk gates passing.

The layer never enables withdrawals.

## Multi-timeframe contract

Default swing chain:

```text
15m = execution timing
1h  = primary trigger and model bar
2h  = setup confirmation
4h  = setup / trend confirmation
1d  = market regime
1w  = structural regime
```

Only closed bars are accepted. Higher-timeframe features are lagged before joining to lower-timeframe decisions.

## AI roles

The default authority order is:

```text
market/data truth
    >
deterministic risk and execution constraints
    >
validated strategy evidence
    >
supervised forecast / classifier
    >
Kronos challenger
    >
RL challenger
    >
LLM commentary
```

No LLM is required. An LLM can summarize evidence but cannot create execution authority.

## Research gate

Candidate strategies move through:

```text
FAST_DISCOVERY
PARAMETER_SEARCH
PURGED_WALK_FORWARD
COST_STRESS
STATISTICAL_VALIDATION
CROSS_ENGINE_VALIDATION
FORWARD_SHADOW
```

Promotion is evidence-based and does not automatically enable live execution.

## Directory overview

Run:

```bash
python scripts/print_tree.py
```

for the generated structure.
