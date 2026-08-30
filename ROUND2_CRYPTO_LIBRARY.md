# Round 2: Sjagil/crypto as the runtime library

This round changes the active data path from a duplicate direct REST market-data
client to the existing `Sjagil/crypto` repository.

The swing layer now imports and uses the crypto repository for:

- `config.settings.get_settings`
- `data.data_loader.DataLoader`
- Bitvavo OHLCV, ticker, trades and order-book snapshots
- 15m / 1h / 2h / 4h / 1d / 1w market context
- `data.prospective_context.ProspectiveContextCollector`
- CMC/prospective context already owned by the crypto repository
- microstructure readiness from `data.orderflow_recorder`
- JSON/JSONL and Parquet RSS/news/intelligence artifacts
- discovery/import validation of WebSocket, L2, feature store, governance,
  account inventory and execution authority modules

Live execution deliberately remains fail-closed. The previous direct Bitvavo order
call is no longer the preferred live path. Run `crypto-library-doctor` to expose
the exact public interface of the installed `core.execution_authority`; map that
exact callable in the next round instead of guessing a private execution API.
