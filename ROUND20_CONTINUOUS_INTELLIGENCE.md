# v0.20.0 Continuous Intelligence

The existing supervisor is now tuned as the always-on improvement loop.

Every 60 seconds it continues to:
- screen all 25 runtime markets on closed 1h bars;
- manage forward evidence and prospective outcomes;
- deep-scan the strongest candidates across 15m/1h/2h/4h/1d/1w;
- generate signals/holds without forcing a trade.

This round adds:
- MACD, ADX/+DI/-DI, Donchian 55, ATR expansion, Bollinger squeeze,
  volume ratio, OBV slope, CMF and EMA slope features;
- explicit technical/breakout state on the 25-market screen;
- multi-timeframe technical intelligence on deep-scan markets;
- dashboard signal/watch rows from the swing-layer proactive output;
- research every 3h, agent retraining every 4h, economics every 6h.

Research/model outputs remain advisory/shadow and are never auto-promoted.
