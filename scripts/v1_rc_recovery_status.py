from __future__ import annotations
import json
from pathlib import Path
from crypto_ai_swing.production.recovery import ExecutionRecoveryJournal
from crypto_ai_swing.settings import Settings

def main() -> int:
    settings = Settings.load()
    journal = ExecutionRecoveryJournal(
        Path(settings.project_root)
        / "output/crypto_ai_swing/production/execution_recovery.sqlite"
    )
    print(json.dumps(journal.status(), indent=2, default=str))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
