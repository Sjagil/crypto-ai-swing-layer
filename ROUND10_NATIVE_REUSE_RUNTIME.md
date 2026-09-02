# Round 10: native crypto reuse and production runtime

The swing layer stays responsible for swing orchestration, agents and strategy
logic. `Sjagil/crypto` stays authoritative for runtime primitives that already
exist.

Round 10 reuses the following directly instead of creating parallel systems:

- `utils.common.atomic_write_json` and append-only persistence utilities;
- `notifications.telegram.TelegramNotifier`;
- `risk.risk_manager.OperationalDegradation` and `RiskManager`;
- `risk.correlation_analyzer.CorrelationAnalyzer`;
- `research.trading_math.calculate_position_size*`;
- `core.live_asset_preflight.live_account_health`;
- the existing native live execution adapter remains unchanged.

The supervisor now writes a progress heartbeat throughout long research and
training tasks. `runtime_health.py` can report `HEALTHY_BUSY` and also verifies
the heartbeat PID is alive.

Operational degradation and recovery notifications are routed through the
existing native Telegram implementation. Telegram failures remain isolated and
cannot alter execution authority.

No live authority, canary cap, leverage, shorting, withdrawal or automatic
promotion setting is changed by this round.
