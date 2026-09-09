#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_ai_swing.settings import Settings


def _safe(callable_, *args, **kwargs) -> dict[str, Any]:
    try:
        value = callable_(*args, **kwargs)
    except Exception as exc:
        return {
            "status": "ERROR",
            "error": f"{type(exc).__name__}:{str(exc)[:800]}",
        }
    return {"status": "READY", "value": value}


def main() -> int:
    settings = Settings.load(ROOT)
    crypto_root = settings.crypto_repo_root.resolve()
    if str(crypto_root) not in sys.path:
        sys.path.insert(0, str(crypto_root))

    from config.settings import Settings as CryptoSettings
    from core.economics import CanonicalCostModel
    from core.opportunity_intelligence import (
        build_canonical_ml_dataset,
        train_canonical_shadow_models,
    )
    from research.combinatorial_lab import (
        CombinationGenerator,
        LogicMode,
        signal_block_registry,
        validate_blocks,
    )

    crypto_settings = CryptoSettings.load(
        env_file=crypto_root / ".env",
        create_directories=True,
    )

    cost = CanonicalCostModel.from_settings(crypto_settings)
    registry = signal_block_registry()
    validation = validate_blocks(registry)

    block_id = next(iter(registry))
    generated = CombinationGenerator(registry).generate(
        sizes=(1,),
        logic_modes=(LogicMode.ALL,),
        block_ids=(block_id,),
    )

    dataset = build_canonical_ml_dataset(crypto_settings)
    training = train_canonical_shadow_models(crypto_settings)

    payload = {
        "schema_version": "crypto_ai_swing_crypto_capability_v1",
        "status": "READY",
        "crypto_repo_root": str(crypto_root),
        "canonical_cost_model": {
            "status": "READY",
            "cost_model_version": getattr(
                cost, "cost_model_version", None
            ),
        },
        "strategy_generator": {
            "status": (
                "READY"
                if validation.get("status") == "PASSED" and len(generated) >= 1
                else "BLOCKED"
            ),
            "registry_blocks": len(registry),
            "registry_validation": validation,
            "smoke_block_id": block_id,
            "smoke_generated": len(generated),
        },
        "canonical_ml_dataset": {
            "status": dataset.get("status"),
            "row_count": dataset.get(
                "canonical_point_in_time_rows",
                dataset.get("row_count", 0),
            ),
            "pipeline_smoke_rows": dataset.get("pipeline_smoke_rows"),
            "minimum_shadow_evaluation_rows": dataset.get(
                "minimum_shadow_evaluation_rows"
            ),
            "orders_generated": dataset.get("orders_generated", 0),
            "orders_submitted": dataset.get("orders_submitted", 0),
        },
        "canonical_ml_training": {
            "status": training.get("status"),
            "model_registered": training.get("model_registered", False),
            "row_count": training.get(
                "row_count",
                training.get("canonical_point_in_time_rows", 0),
            ),
            "positive_rows": training.get("positive_rows"),
            "negative_rows": training.get("negative_rows"),
            "reason": training.get("reason"),
            "live_decision_influence": training.get(
                "live_decision_influence", False
            ),
            "orders_generated": training.get("orders_generated", 0),
            "orders_submitted": training.get("orders_submitted", 0),
        },
        "execution_authority": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    if payload["strategy_generator"]["status"] != "READY":
        payload["status"] = "BLOCKED"
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
