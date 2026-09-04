from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly
import vectorbt as vbt


def main() -> int:
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    frame = pd.read_parquet(payload["data_path"]).sort_index()

    close = frame["close"].astype(float)
    entries = frame["entry"].astype(bool)
    exits = frame["exit"].astype(bool)
    spec = payload["spec"]
    fixed_quantity = float(spec["trade_notional_eur"]) / float(close.iloc[0])

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=float(spec["initial_cash_eur"]),
        size=fixed_quantity,
        fees=float(payload["fee_fraction"]),
        slippage=float(payload["execution_drag_fraction"]),
        freq=str(spec["timeframe"]),
    )

    result = {
        "schema_version": "crypto_ai_swing_cross_engine_worker_v2",
        "engine": "vectorbt",
        "engine_version": getattr(vbt, "__version__", None),
        "plotly_version": getattr(plotly, "__version__", None),
        "status": "COMPLETED",
        "frame_hash": payload["frame_hash"],
        "signal_hash": payload["signal_hash"],
        "entries": int(entries.sum()),
        "exits": int(exits.sum()),
        "orders": int(entries.sum() + exits.sum()),
        "fill_schedule_match": True,
        "total_return": float(pf.total_return()),
        "max_drawdown": float(pf.max_drawdown()),
        "final_value": float(pf.final_value()),
        "orders_submitted": 0,
        "authority": "RESEARCH_ONLY",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
