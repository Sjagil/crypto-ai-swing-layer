from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_ai_swing.research.entry_selector import ProspectiveSwingEntrySelector
from crypto_ai_swing.research.selector_optuna import SelectorOptunaTuner
from crypto_ai_swing.settings import Settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="paper", choices=("shadow", "paper"))
    parser.add_argument(
        "--trials",
        type=int,
        default=40,
        help="Target total finished trials in the persisted study, not trials per run.",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument(
        "--study-name",
        default="prospective-swing-entry-selector-v028",
    )
    args = parser.parse_args()

    settings = Settings.load(ROOT)
    selector = ProspectiveSwingEntrySelector(settings, mode=args.mode)
    tuner = SelectorOptunaTuner(selector)
    payload = tuner.run(
        trials=args.trials,
        study_name=args.study_name,
        seed=args.seed,
        fold_count=args.folds,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload.get("status") in {"COMPLETE", "COLLECTING"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
