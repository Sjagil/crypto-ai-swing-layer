# Round 37 Canonical Data Source Fabric

`crypto-ai-swing-layer` remains an orchestration/product repository.
`Sjagil/crypto` remains the canonical runtime and data source of truth. Existing
compatibility/enrichment modules in the swing repository are not promoted to
authoritative providers or execution authority by Round 37.

## What changed

The swing layer now has a secret-safe `CanonicalDataSourceFabric` which verifies
and orchestrates these canonical contracts:

- `data.data_loader.DataLoader`
- `data.multi_source_runtime.run_multi_source_collector`
- `data.collector_health.collector_health_report`
- `data.prospective_context.ProspectiveContextCollector`
- `core.market_intelligence.build_coin_ranking`
- `scrapers.rss.collect_registered_feeds`
- `scrapers.intelligence.run_intelligence_pipeline`

The canonical multi-source collector owns public Bitvavo, Kraken and MEXC streams,
low-frequency context collection, immutable source ledgers, RSS collection and
collector health. The canonical intelligence pipeline owns web scraping, robots
handling, RSS, relevance filtering, causal timestamp handling, deduplication and
Parquet persistence.

The swing `CryptoNewsCollector` now consumes the full canonical web + RSS
intelligence pipeline. It falls back to canonical RSS only when the broader
pipeline is unavailable. Round 37 adds no new network client or execution path.

## Credentials and environment ownership

Provider and exchange credentials belong in the local `Sjagil/crypto/.env`, not
in `crypto-ai-swing-layer/.env`. The swing `.env` loader imports only `CRYPTO_REPO_PATH` and `CRYPTO_SWING_*`
variables. On canonical bridge activation, the canonical `.env` becomes
authoritative for provider, exchange, risk and live-authority keys. This prevents
a duplicated swing flag such as `LIVE_TRADING_ALLOWED` from silently overriding
the canonical engine.

Set this only when an explicitly exported deployment/test process must override
the canonical environment:

```bash
export CRYPTO_SWING_CANONICAL_ENV_PRECEDENCE=process
```

No audit payload contains secret values. Audits report key names and configured
state only.

## Bitvavo while moving networks

The data fabric does not require a Bitvavo IP whitelist for public market data.
Private account authentication is separately diagnosed by
`scripts/diagnose_private_account.py`. The swing layer does not weaken the
canonical live-execution policy. If the canonical engine requires an IP whitelist
for live trading, live remains blocked even when a non-whitelisted trade key can
authenticate successfully.

If the local key is intentionally not IP-restricted, do not leave stale assertions
such as `BITVAVO_IP_WHITELIST_CONFIRMED=true` in the effective environment. The
data-source audit reports duplicate/conflicting environment keys and whitelist
assertion state without exposing values.

## Audit without network collection

```bash
python scripts/canonical_data_source_audit.py
```

This verifies the canonical module contracts, provider configuration coverage,
RSS/web source catalog, current collector artifacts and environment conflicts.
It submits no orders and performs no private exchange requests.

## Bounded real public collection

```bash
python scripts/canonical_data_source_audit.py --collect-seconds 30
```

This runs the canonical public/read-only multi-source collector for 30 seconds,
then runs the canonical web + RSS intelligence pipeline and re-audits the result.
Use `--skip-intelligence` to run only the market/context collector.

To require all keyed canonical provider settings to be configured as part of the
audit, add:

```bash
python scripts/canonical_data_source_audit.py --strict-config
```

Missing optional provider keys never cause the swing layer to invent a fallback
client. They remain explicit data-coverage gaps until configured in
`Sjagil/crypto/.env`.

## Acceptance

Run the normal repository gates plus the Round 37 tests:

```bash
python -m compileall -q src scripts tests
ruff check --select F,E9 src scripts tests
python -m pytest -q
python scripts/lean_integration_audit.py
python scripts/canonical_engine_contract_audit.py
python scripts/canonical_data_source_audit.py
git diff --check
```

For a real external-data smoke test, additionally run the bounded collection
command above. Real private Bitvavo readiness remains a separate explicit check.
