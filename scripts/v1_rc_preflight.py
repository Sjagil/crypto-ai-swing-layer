from __future__ import annotations
import argparse, json
from crypto_ai_swing.production.readiness import ProductionReadinessEngine
from crypto_ai_swing.settings import Settings

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", default="BTC-EUR,ETH-EUR,SOL-EUR,LINK-EUR")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    settings = Settings.load()
    markets = [x.strip().upper() for x in args.markets.split(",") if x.strip()]
    payload = ProductionReadinessEngine(settings).assess(
        markets, network=not args.offline, persist=True
    )
    print(json.dumps(payload, indent=2, default=str))
    return 0 if payload.get("live_canary_ready") else 2

if __name__ == "__main__":
    raise SystemExit(main())
