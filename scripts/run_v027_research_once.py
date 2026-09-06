from __future__ import annotations

import json

from crypto_ai_swing.research.entry_selector import ProspectiveSwingEntrySelector
from crypto_ai_swing.settings import Settings


def main() -> int:
    settings = Settings.load()
    selector = ProspectiveSwingEntrySelector(settings, mode="paper")
    payload = selector.refresh(force=True)
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
