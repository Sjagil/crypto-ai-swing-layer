#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def contains(path: Path, value: str) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"MISSING:{path}"
    return (value in path.read_text(encoding="utf-8"), str(path))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--swing-root", default=".")
    p.add_argument("--crypto-root", required=True)
    args = p.parse_args()
    swing = Path(args.swing_root).expanduser().resolve()
    crypto = Path(args.crypto_root).expanduser().resolve()

    checks = [
        contains(swing / "src/crypto_ai_swing/models/tcn_gru.py", "historical_ohlcv_many"),
        contains(swing / "src/crypto_ai_swing/models/tcn_gru_runtime.py", "class TCNGRURuntime"),
        contains(swing / "src/crypto_ai_swing/models/adaptive_model_averaging.py", "blend_probabilities"),
        contains(swing / "src/crypto_ai_swing/cli.py", "FULL_LIVE_CANONICAL_GATE_NOT_READY"),
        contains(swing / "src/crypto_ai_swing/orchestration/proactive.py", "CRYPTO_SWING_FULL_LIVE"),
        contains(swing / "src/crypto_ai_swing/orchestration/proactive.py", 'context[market]["ml_ensemble"]'),
        contains(swing / "src/crypto_ai_swing/orchestration/pipeline.py", "and not full_live"),
        contains(swing / "src/crypto_ai_swing/agents/live_promotion.py", "LIVE_QUALIFIED"),
        contains(crypto / "core/live_capital.py", "CRYPTO_FULL_LIVE_MAXIMUM_ORDER_EUR"),
        contains(crypto / "core/swing_layer_live.py", "approve_swing_layer_full_live"),
    ]
    failed = [path for ok, path in checks if not ok]
    for ok, path in checks:
        print(f"{'PASS' if ok else 'FAIL'} {path}")
    print("FULL_LIVE_QUANT_MACHINE=PASS" if not failed else "FULL_LIVE_QUANT_MACHINE=FAIL")
    return 0 if not failed else 4


if __name__ == "__main__":
    raise SystemExit(main())
