# v0.19.0 Realtime Terminal Operations

`crypto-swing terminal-dashboard` is a read-only Rich TUI.

- 25-market Bitvavo ticker WebSocket
- 4 Hz terminal refresh with event-driven ingestion
- spread and message-age visibility
- public/private/orderflow WebSocket health
- service authority and execution health
- orders, fills and positions
- signals and decisions
- agents, models and training state

The price feed reuses the canonical `Sjagil/crypto` WebSocketManager. It has
no order method and does not grant live authority. The 15m/1h swing decision
cadence remains independent from the fast market-data and monitoring path.
