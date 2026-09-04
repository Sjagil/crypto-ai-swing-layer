from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, StrategyConfig

try:
    from nautilus_trader.config import LoggingConfig
except ImportError:
    from nautilus_trader.config import LoggerConfig as LoggingConfig

from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy


class ParityScheduleConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    entry_timestamps: tuple[int, ...]
    exit_timestamps: tuple[int, ...]


class ParityScheduleStrategy(Strategy):
    def __init__(self, config: ParityScheduleConfig):
        super().__init__(config)
        self._entry_timestamps = frozenset(config.entry_timestamps)
        self._exit_timestamps = frozenset(config.exit_timestamps)

    def on_start(self):
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar):
        ts = int(bar.ts_event)
        if ts in self._entry_timestamps:
            if self.portfolio.is_flat(self.config.instrument_id):
                self._buy()
            return
        if ts in self._exit_timestamps and self.portfolio.is_net_long(
            self.config.instrument_id
        ):
            self.close_all_positions(self.config.instrument_id)

    def _buy(self):
        instrument = self.cache.instrument(self.config.instrument_id)
        order = self.order_factory.market(
            self.config.instrument_id,
            OrderSide.BUY,
            instrument.make_qty(self.config.trade_size),
        )
        self.submit_order(order)


def _timeframe_bar_string(timeframe: str) -> str:
    mapping = {
        "5m": "5-MINUTE",
        "15m": "15-MINUTE",
        "30m": "30-MINUTE",
        "1h": "1-HOUR",
        "2h": "2-HOUR",
        "4h": "4-HOUR",
        "1d": "1-DAY",
        "1W": "1-WEEK",
        "1w": "1-WEEK",
    }
    if timeframe not in mapping:
        raise ValueError(f"Unsupported Nautilus timeframe: {timeframe}")
    return mapping[timeframe]


def _currency(code: str) -> Currency:
    try:
        return Currency.from_str(code)
    except ValueError:
        creator = getattr(Currency, "get_or_create_crypto", None)
        if callable(creator):
            return creator(code)
        raise


def _logging_config():
    try:
        return LoggingConfig(log_level="ERROR")
    except TypeError:
        return LoggingConfig()


def _to_ns(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(pd.Timestamp(value).value)
    except (TypeError, ValueError):
        return None


def _fills_report(engine: BacktestEngine):
    generator = getattr(engine, "generate_fills_report", None)
    if callable(generator):
        return generator()
    trader_generator = getattr(engine.trader, "generate_fills_report", None)
    if callable(trader_generator):
        return trader_generator()
    raise RuntimeError("Nautilus fills report API unavailable")


def _canonical_fill_ledger(
    fills: pd.DataFrame,
    *,
    initial_cash: float,
    final_mark: float,
    fee_fraction: float,
    execution_drag_fraction: float,
) -> dict[str, Any]:
    cash = float(initial_cash)
    base_quantity = 0.0
    observed_timestamps: list[int] = []

    if fills is not None and len(fills) > 0:
        working = fills.copy()
        if "ts_event" in working.columns:
            working = working.sort_values("ts_event")

        for _, row in working.iterrows():
            side = str(row.get("order_side") or row.get("side") or "").upper()
            px = float(row.get("last_px"))
            qty = float(row.get("last_qty"))
            observed = _to_ns(row.get("ts_event"))
            if observed is not None:
                observed_timestamps.append(observed)

            if "BUY" in side:
                adjusted_px = px * (1.0 + execution_drag_fraction)
                gross = qty * adjusted_px
                cash -= gross + gross * fee_fraction
                base_quantity += qty
            elif "SELL" in side:
                adjusted_px = px * (1.0 - execution_drag_fraction)
                gross = qty * adjusted_px
                cash += gross - gross * fee_fraction
                base_quantity -= qty

    final_value = cash + base_quantity * float(final_mark)
    return {
        "cash": cash,
        "base_quantity": base_quantity,
        "final_value": final_value,
        "total_return": final_value / initial_cash - 1.0 if initial_cash > 0 else 0.0,
        "observed_fill_timestamps": observed_timestamps,
    }


def main() -> int:
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    spec = payload["spec"]
    frame = pd.read_parquet(payload["data_path"]).sort_index()

    base_code, quote_code = str(spec["market"]).split("-", 1)
    venue = Venue("SIM")
    instrument_id = InstrumentId.from_str(f"{base_code}/{quote_code}.SIM")
    base = _currency(base_code)
    quote = _currency(quote_code)

    first_close = float(frame["close"].iloc[0])
    fixed_quantity = float(spec["trade_notional_eur"]) / first_close

    pair = CurrencyPair(
        instrument_id=instrument_id,
        raw_symbol=Symbol(f"{base_code}/{quote_code}"),
        base_currency=base,
        quote_currency=quote,
        price_precision=8,
        size_precision=8,
        price_increment=Price.from_str("0.00000001"),
        size_increment=Quantity.from_str("0.00000001"),
        ts_event=0,
        ts_init=0,
        maker_fee=Decimal(str(payload["costs"]["maker_fee_fraction"])),
        taker_fee=Decimal(str(payload["costs"]["taker_fee_fraction"])),
    )

    bar_type = BarType.from_str(
        f"{base_code}/{quote_code}.SIM-"
        f"{_timeframe_bar_string(spec['timeframe'])}-LAST-EXTERNAL"
    )

    bars = []
    for timestamp, row in frame.iterrows():
        ts = pd.Timestamp(timestamp)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        nanos = int(ts.value)
        volume = max(float(row.get("volume", 0.0)), 0.00000001)
        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{float(row['open']):.8f}"),
                high=Price.from_str(f"{float(row['high']):.8f}"),
                low=Price.from_str(f"{float(row['low']):.8f}"),
                close=Price.from_str(f"{float(row['close']):.8f}"),
                volume=Quantity.from_str(f"{volume:.8f}"),
                ts_event=nanos,
                ts_init=nanos,
            )
        )

    entry_timestamps = tuple(
        int(pd.Timestamp(ts).value)
        for ts in frame.index[frame["entry"].astype(bool)]
    )
    exit_timestamps = tuple(
        int(pd.Timestamp(ts).value)
        for ts in frame.index[frame["exit"].astype(bool)]
    )
    expected_fill_timestamps = sorted(entry_timestamps + exit_timestamps)

    engine = BacktestEngine(config=BacktestEngineConfig(logging=_logging_config()))
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        starting_balances=[Money(float(spec["initial_cash_eur"]), quote)],
        base_currency=None,
        bar_execution=True,
    )
    engine.add_instrument(pair)
    engine.add_data(bars)

    strategy = ParityScheduleStrategy(
        ParityScheduleConfig(
            instrument_id=pair.id,
            bar_type=bar_type,
            trade_size=Decimal(str(fixed_quantity)),
            entry_timestamps=entry_timestamps,
            exit_timestamps=exit_timestamps,
        )
    )
    engine.add_strategy(strategy)
    engine.run()

    fills = _fills_report(engine)
    account = engine.trader.generate_account_report(venue)
    positions = engine.trader.generate_positions_report()

    ledger = _canonical_fill_ledger(
        fills,
        initial_cash=float(spec["initial_cash_eur"]),
        final_mark=float(frame["close"].iloc[-1]),
        fee_fraction=float(payload["fee_fraction"]),
        execution_drag_fraction=float(payload["execution_drag_fraction"]),
    )
    observed_fill_timestamps = sorted(ledger["observed_fill_timestamps"])

    result = {
        "schema_version": "crypto_ai_swing_cross_engine_worker_v2",
        "engine": "nautilus",
        "status": "COMPLETED",
        "frame_hash": payload["frame_hash"],
        "signal_hash": payload["signal_hash"],
        "expected_orders": len(expected_fill_timestamps),
        "orders": len(fills) if fills is not None else 0,
        "fill_schedule_match": observed_fill_timestamps == expected_fill_timestamps,
        "expected_fill_timestamps": expected_fill_timestamps,
        "observed_fill_timestamps": observed_fill_timestamps,
        "final_value": ledger["final_value"],
        "total_return": ledger["total_return"],
        "canonical_cash": ledger["cash"],
        "canonical_base_quantity": ledger["base_quantity"],
        "closed_positions": len(positions) if positions is not None else 0,
        "account_columns": [str(x) for x in account.columns] if account is not None else [],
        "orders_submitted": 0,
        "authority": "RESEARCH_ONLY",
    }

    try:
        import nautilus_trader
    except ImportError:
        result["engine_version"] = None
    else:
        result["engine_version"] = getattr(nautilus_trader, "__version__", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
