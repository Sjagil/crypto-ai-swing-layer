from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from crypto_ai_swing.validation.cross_engine import run_cross_engine_validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", default="BTC-EUR")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--fast-ema", type=int, default=20)
    parser.add_argument("--slow-ema", type=int, default=60)
    parser.add_argument("--maximum-rows", type=int, default=5000)
    parser.add_argument("--venv-root", type=Path)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    crypto_root = Path(os.getenv("CRYPTO_REPO_PATH", root.parent / "crypto")).expanduser().resolve()
    payload = run_cross_engine_validation(
        root,
        crypto_root,
        market=args.market,
        timeframe=args.timeframe,
        fast_ema=args.fast_ema,
        slow_ema=args.slow_ema,
        maximum_rows=args.maximum_rows,
        venv_root=args.venv_root,
        repo_root=args.repo_root,
    )
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
