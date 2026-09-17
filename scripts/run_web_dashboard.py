from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_ai_swing.monitoring.web_dashboard import run_web_dashboard
from crypto_ai_swing.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only crypto AI swing web operations dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    settings = Settings.load(ROOT)
    run_web_dashboard(
        settings,
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
    )


if __name__ == "__main__":
    main()
