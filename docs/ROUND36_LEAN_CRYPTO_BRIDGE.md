# Round 36 Lean Crypto Bridge

`crypto-ai-swing-layer` is the product/orchestration repository.

`Sjagil/crypto` remains the canonical engine checkout and the single source of
truth for candles, market data, research primitives, risk, portfolio authority,
Bitvavo execution, stop-loss/take-profit and reconciliation.

The two repositories are integrated in-process through `CryptoLibraryBridge`.
No crypto source tree, data store, output tree, reference repositories or Rust
build artifacts are vendored into `crypto-ai-swing-layer`.

Recommended local layout:

```text
/Users/ayoubalhari/Downloads/
  crypto/
  crypto-ai-swing-layer/
```

Set `CRYPTO_REPO_PATH=/Users/ayoubalhari/Downloads/crypto`.
