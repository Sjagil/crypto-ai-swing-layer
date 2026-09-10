from __future__ import annotations

import json

from crypto_ai_swing.research.selector_runtime import ProspectiveSelectorRuntime
from crypto_ai_swing.settings import Settings


def main() -> int:
    settings = Settings.load()
    runtime = ProspectiveSelectorRuntime(settings, mode="paper")
    try:
        print(json.dumps(runtime.status(), indent=2, sort_keys=True, default=str))
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
