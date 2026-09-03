from __future__ import annotations

import argparse
import json
from pathlib import Path

from crypto_ai_swing.research.native_dependencies import (
    inspect_native_campaign_dependencies,
)
from crypto_ai_swing.settings import Settings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect native Sjagil/crypto campaign prerequisites."
    )
    parser.add_argument("campaign", nargs="?", default="multi-alpha-v2")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    settings = Settings.load(root)
    payload = inspect_native_campaign_dependencies(
        crypto_repo_root=settings.crypto_repo_root,
        campaign=args.campaign,
    )
    print(json.dumps(payload, indent=2, default=str))
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
