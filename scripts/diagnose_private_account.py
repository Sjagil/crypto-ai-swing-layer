#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_ai_swing.bridge.engine_contract import validate_engine_contract
from crypto_ai_swing.bridge.runtime import activate_canonical_crypto
from crypto_ai_swing.settings import Settings as SwingSettings


def _safe_error_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        allowed = {}
        for key in (
            "errorCode",
            "error",
            "message",
            "status",
            "reason",
            "code",
        ):
            if key in value:
                allowed[key] = value[key]
        return allowed
    return {"message": str(value)[:500]}


async def main_async(require_clean: bool) -> int:
    swing = SwingSettings.load(ROOT)
    try:
        contract = validate_engine_contract(
            ROOT,
            settings=swing,
            require_clean=require_clean,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema_version": "round36_1_private_probe_v1",
                    "status": "BLOCKED_ENGINE_CONTRACT",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "orders_generated": 0,
                    "orders_submitted": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    bridge = activate_canonical_crypto(swing)
    # Third-party dependencies come from the active environment.
    # CryptoLibraryBridge is intentionally reserved for modules that
    # must resolve from the pinned canonical Sjagil/crypto checkout.
    import aiohttp
    settings_module = bridge.import_module("config.settings")
    execution_module = bridge.import_module("execution.execution")
    live_module = bridge.import_module("core.autonomous_live")

    settings = settings_module.Settings.load(create_directories=True)
    markets = tuple(live_module.resolved_live_markets(settings))

    result: dict[str, Any] = {
        "schema_version": "round36_1_private_probe_v1",
        "checked_at": datetime.now(UTC).isoformat(),
        "engine": {
            "root": str(contract.root),
            "branch": contract.branch,
            "head": contract.actual_head,
            "clean": contract.clean,
            "matched_lock": contract.matched,
        },
        "markets": list(markets),
        "stages": {},
        "orders_generated": 0,
        "orders_submitted": 0,
        "withdrawal_requests": 0,
    }

    timeout = aiohttp.ClientTimeout(
        total=min(20.0, settings.market_data.request_timeout_seconds)
    )

    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            client = execution_module.build_live_client(
                settings,
                session=session,
                ledger_path=(
                    settings.paths.checkpoints_dir / "live_execution.jsonl"
                ),
            )
            result["stages"]["build_live_client"] = {"status": "READY"}
        except Exception as exc:
            result["stages"]["build_live_client"] = {
                "status": "BLOCKED",
                "exception_type": type(exc).__name__,
                "message": str(exc)[:500],
            }
            result["status"] = "BLOCKED"
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
            return 1

        try:
            venue_ms = await client.server_time_ms()
            local_ms = int(datetime.now(UTC).timestamp() * 1000)
            result["stages"]["server_time"] = {
                "status": "READY",
                "drift_ms_approx": venue_ms - local_ms,
            }
        except Exception as exc:
            result["stages"]["server_time"] = {
                "status": "BLOCKED",
                "exception_type": type(exc).__name__,
                "message": str(exc)[:500],
            }

        # Directly inspect the read-only balance endpoint once so an HTTP
        # status/errorCode is not collapsed into a generic preflight blocker.
        signed_path = "/v2/balance"
        headers = client._headers("GET", signed_path, "")
        try:
            async with session.get(
                f"https://api.bitvavo.com{signed_path}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                payload = await response.json(content_type=None)
                stage: dict[str, Any] = {
                    "status": "READY" if response.status < 400 else "BLOCKED",
                    "http_status": response.status,
                }
                if response.status >= 400:
                    stage["venue_error"] = _safe_error_payload(payload)
                elif isinstance(payload, list):
                    positives = []
                    eur_available = "0"
                    for row in payload:
                        if not isinstance(row, Mapping):
                            continue
                        symbol = str(row.get("symbol") or "").upper()
                        available = str(row.get("available") or "0")
                        in_order = str(row.get("inOrder") or "0")
                        try:
                            positive = float(available) > 0 or float(in_order) > 0
                        except (TypeError, ValueError):
                            positive = False
                        if symbol == "EUR":
                            eur_available = available
                        if positive:
                            positives.append(symbol)
                    stage.update(
                        {
                            "eur_available": eur_available,
                            "positive_symbols": sorted(set(positives)),
                            "row_count": len(payload),
                        }
                    )
                else:
                    stage["status"] = "BLOCKED"
                    stage["venue_error"] = {
                        "message": "balance payload is not a list"
                    }
                result["stages"]["balance_http"] = stage
        except Exception as exc:
            result["stages"]["balance_http"] = {
                "status": "BLOCKED",
                "exception_type": type(exc).__name__,
                "message": str(exc)[:500],
            }

        if result["stages"].get("balance_http", {}).get("status") == "READY":
            try:
                fees = await client.account_fees()
                result["stages"]["account_fees"] = {
                    "status": "READY",
                    "maker": str(fees["maker"]),
                    "taker": str(fees["taker"]),
                }
            except Exception as exc:
                result["stages"]["account_fees"] = {
                    "status": "BLOCKED",
                    "exception_type": type(exc).__name__,
                    "message": str(exc)[:500],
                }

            try:
                reconciliation = await client.reconcile(markets=markets)
                result["stages"]["reconcile"] = {
                    "status": "READY" if reconciliation.healthy else "BLOCKED",
                    "healthy": bool(reconciliation.healthy),
                    "reason_codes": list(reconciliation.reason_codes),
                    "local_open_orders": reconciliation.local_open_orders,
                    "remote_open_orders": reconciliation.remote_open_orders,
                }
            except Exception as exc:
                result["stages"]["reconcile"] = {
                    "status": "BLOCKED",
                    "exception_type": type(exc).__name__,
                    "message": str(exc)[:500],
                }

    blockers = [
        name
        for name, stage in result["stages"].items()
        if stage.get("status") == "BLOCKED"
    ]
    result["status"] = "READY" if not blockers else "BLOCKED"
    result["blocked_stages"] = blockers
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["status"] == "READY" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-dirty-engine",
        action="store_true",
        help="diagnostics only; never grants live authority",
    )
    args = parser.parse_args()
    return asyncio.run(
        main_async(require_clean=not args.allow_dirty_engine)
    )


if __name__ == "__main__":
    raise SystemExit(main())
