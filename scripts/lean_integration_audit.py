#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_ai_swing.bridge.runtime import activate_canonical_crypto
from crypto_ai_swing.settings import Settings

CRYPTO_ONLY_TOP_LEVEL = (
    "core", "execution", "portfolio", "risk", "research", "reporting",
    "scrapers", "notifications", "ml", "ui", "utils", "data_store",
    "crypto-references",
)


def main() -> int:
    settings = Settings.load(ROOT)
    bridge = activate_canonical_crypto(settings)
    integration = bridge.integration_status()
    crypto_root = Path(settings.crypto_repo_root).resolve()

    physical_duplicates = [
        name for name in CRYPTO_ONLY_TOP_LEVEL if (ROOT / name).exists()
    ]
    required = (
        "core.autonomous_live",
        "core.event_driven_live",
        "data.data_loader",
        "execution.execution",
        "portfolio.active_allocator",
        "research.features",
        "scrapers.intelligence",
    )
    import_locations = {}
    wrong_root = []
    failures = {}
    for name in required:
        try:
            module = bridge.import_module(name)
            location = Path(str(module.__file__)).resolve()
            import_locations[name] = str(location)
            if crypto_root not in location.parents:
                wrong_root.append(name)
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {str(exc)[:300]}"

    checks = {
        "swing_is_orchestrator_only": not physical_duplicates,
        "canonical_crypto_checkout_exists": (crypto_root / ".git").is_dir(),
        "canonical_bridge_ready": bool(integration.get("ready")),
        "critical_modules_import_from_crypto_checkout": not wrong_root and not failures,
        "separate_execution_authority_only_in_crypto": not (ROOT / "execution").exists(),
        "no_duplicated_data_store": not (ROOT / "data_store").exists(),
        "no_duplicated_crypto_references": not (ROOT / "crypto-references").exists(),
    }
    payload = {
        "schema_version": "round36_lean_bridge_audit_v1",
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "swing_root": str(ROOT),
        "crypto_root": str(crypto_root),
        "checks": checks,
        "physical_duplicates": physical_duplicates,
        "integration": integration,
        "critical_import_locations": import_locations,
        "critical_import_failures": failures,
        "wrong_root_modules": wrong_root,
        "orders_submitted": 0,
        "private_exchange_requests": 0,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
