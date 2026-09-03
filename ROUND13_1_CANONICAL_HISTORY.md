# v0.13.1 Canonical History Contract Repair

This patch repairs the storage contract exposed by v0.13.

`data-sync` fetched valid recent frames through `Sjagil/crypto`, but a nonempty
frame was incorrectly labelled `PERSISTED` even when the flat canonical file
consumed by native health checks and research campaigns did not exist.

v0.13.1 delegates provider pagination, resume, normalized provider storage,
validation and manifest generation to `Sjagil/crypto:data.data_loader.DataLoader`.

Canonical flow:

1. `sync_canonical_ohlcv_compact`
2. native provider-normalized Parquet
3. `materialize_provider_ohlcv_compact`
4. flat compatibility Parquet plus manifest for native research campaigns

No live authority is added and no canary is armed.
