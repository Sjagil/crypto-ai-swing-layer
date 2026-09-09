from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_ai_swing.research.optuna_runtime import OptunaStudyRuntime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    runtime = OptunaStudyRuntime(ROOT)
    payload = runtime.status()
    if args.smoke:
        search_space = {
            "x": {"kind": "float", "low": -2.0, "high": 2.0, "step": 0.25},
            "bucket": {"kind": "categorical", "choices": ["A", "B"]},
        }
        study_name = "round28-runtime-smoke"
        ensured = runtime.ensure_study(
            study_name=study_name,
            search_space=search_space,
            objective_version="runtime_smoke_v1",
            metadata={"research_only": True},
            seed=2801,
        )
        contract = str(ensured["contract_hash"])
        runtime.recover_running(study_name=study_name, contract_hash=contract)
        asked = runtime.ask(
            study_name=study_name,
            contract_hash=contract,
            search_space=search_space,
            seed=2801,
        )
        x = float(asked["params"]["x"])
        value = -(x - 0.5) ** 2
        runtime.tell(
            study_name=study_name,
            contract_hash=contract,
            trial_number=int(asked["trial_number"]),
            value=value,
        )
        payload["smoke"] = {
            "status": "PASSED",
            "study": runtime.summary(study_name=study_name, contract_hash=contract),
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
