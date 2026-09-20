#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from crypto_ai_swing.orchestration.pi_learning_worker import (
    PiUnifiedLearningWorker,
)
from crypto_ai_swing.settings import Settings


PRINT_LOCK = threading.Lock()


def now() -> str:
    return datetime.now(UTC).astimezone().strftime(
        "%Y-%m-%d %H:%M:%S%z"
    )


def emit(message: str) -> None:
    with PRINT_LOCK:
        print(
            f"[{now()}] {message}",
            flush=True,
        )


def process_stats() -> str:
    try:
        result = subprocess.run(
            [
                "ps",
                "-p",
                str(os.getpid()),
                "-o",
                "%cpu=,rss=,state=",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )

        raw = result.stdout.strip().split()

        if len(raw) >= 3:
            cpu = raw[0]
            rss_kb = int(raw[1])
            state = raw[2]

            return (
                f"cpu={cpu}% "
                f"rss={rss_kb / 1024:.1f}MB "
                f"state={state}"
            )

    except Exception:
        pass

    return "process_stats=unavailable"


def short_summary(value: Any) -> str:
    result: dict[str, Any] = {}

    if isinstance(value, dict):
        candidates = value
    else:
        candidates = {}

        for key in (
            "status",
            "state",
            "row_count",
            "dataset_id",
            "markets",
            "metrics",
        ):
            try:
                candidates[key] = getattr(value, key)
            except Exception:
                pass

    for key in (
        "status",
        "state",
        "rows",
        "row_count",
        "market_count",
        "dataset_id",
        "selected_seed",
        "trial_count",
        "candidate_count",
        "qualified",
    ):
        if key in candidates:
            result[key] = candidates[key]

    markets = candidates.get("markets")

    if isinstance(markets, (list, tuple, set)):
        result["markets"] = len(markets)

    errors = candidates.get("errors")

    if isinstance(errors, list):
        result["errors"] = len(errors)

    if not result:
        return ""

    try:
        return json.dumps(
            result,
            default=str,
            separators=(",", ":"),
        )
    except Exception:
        return str(result)


def run_with_progress(
    name: str,
    function: Callable[[], Any],
    *,
    heartbeat_seconds: int,
) -> Any:
    started = time.monotonic()
    stop = threading.Event()

    emit(
        f"[START] task={name} "
        f"pid={os.getpid()} "
        f"{process_stats()}"
    )

    def heartbeat() -> None:
        counter = 0

        while not stop.wait(heartbeat_seconds):
            counter += 1
            elapsed = time.monotonic() - started

            emit(
                f"[RUNNING] task={name} "
                f"elapsed={elapsed:.0f}s "
                f"heartbeat={counter} "
                f"{process_stats()}"
            )

    thread = threading.Thread(
        target=heartbeat,
        name=f"progress-{name}",
        daemon=True,
    )
    thread.start()

    try:
        result = function()

        elapsed = time.monotonic() - started
        summary = short_summary(result)

        emit(
            f"[DONE] task={name} "
            f"elapsed={elapsed:.1f}s "
            f"{process_stats()}"
            + (
                f" result={summary}"
                if summary
                else ""
            )
        )

        return result

    except Exception as exc:
        elapsed = time.monotonic() - started

        emit(
            f"[ERROR] task={name} "
            f"elapsed={elapsed:.1f}s "
            f"type={type(exc).__name__} "
            f"error={str(exc)[:500]}"
        )

        raise

    finally:
        stop.set()
        thread.join(timeout=1)


def instrument(
    target: Any,
    attribute: str,
    label: str,
    heartbeat_seconds: int,
) -> None:
    original = getattr(target, attribute, None)

    if not callable(original):
        emit(
            f"[SKIP-INSTRUMENT] "
            f"{label} callable not found"
        )
        return

    def wrapped(*args, **kwargs):
        return run_with_progress(
            label,
            lambda: original(*args, **kwargs),
            heartbeat_seconds=heartbeat_seconds,
        )

    setattr(target, attribute, wrapped)


def git_head(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        return result.stdout.strip() or "UNKNOWN"

    except Exception:
        return "UNKNOWN"


def instrument_worker(
    worker: PiUnifiedLearningWorker,
    heartbeat_seconds: int,
) -> None:

    # ContinuousLearningWorker internals.
    instrument(
        worker.base.hpo,
        "run_supervised",
        "continuous_learning.supervised_hpo",
        heartbeat_seconds,
    )

    instrument(
        worker.base.hpo,
        "run_rl",
        "continuous_learning.rl_hpo",
        heartbeat_seconds,
    )

    instrument(
        worker.base.chief,
        "cycle",
        "continuous_learning.chief_agent",
        heartbeat_seconds,
    )

    instrument(
        worker.base.governor,
        "cycle",
        "continuous_learning.model_governor",
        heartbeat_seconds,
    )

    # Unified Pi worker components.
    instrument(
        worker.cmc,
        "status",
        "cmc.status",
        heartbeat_seconds,
    )

    instrument(
        worker,
        "_run_cmc_research",
        "cmc.feature_research",
        heartbeat_seconds,
    )

    instrument(
        worker,
        "_run_tcn",
        "tcn_gru.training",
        heartbeat_seconds,
    )

    instrument(
        worker.tcn_governor,
        "cycle",
        "tcn_gru.governor",
        heartbeat_seconds,
    )

    instrument(
        worker,
        "_run_factory",
        "strategy_factory",
        heartbeat_seconds,
    )

    # This prints total ContinuousLearningWorker duration too.
    instrument(
        worker.base,
        "run_once",
        "continuous_learning.total",
        heartbeat_seconds,
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=("shadow", "paper", "live"),
        default="shadow",
    )

    parser.add_argument(
        "--once",
        action="store_true",
    )

    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--factory-exact",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    emit("============================================================")
    emit("CRYPTO AI SWING VERBOSE LEARNING WORKER")
    emit("============================================================")
    emit(f"repo={root}")
    emit(f"git_head={git_head(root)}")
    emit(f"python_pid={os.getpid()}")
    emit(f"mode={args.mode}")
    emit(f"factory_exact={args.factory_exact}")
    emit(f"heartbeat_seconds={args.heartbeat_seconds}")

    settings = Settings.load(root)

    worker = PiUnifiedLearningWorker(
        settings,
        mode=args.mode,
        factory_exact=args.factory_exact,
    )

    instrument_worker(
        worker,
        max(5, args.heartbeat_seconds),
    )

    if args.once:
        payload = run_with_progress(
            "pi_unified_cycle",
            worker.run_once,
            heartbeat_seconds=max(
                5,
                args.heartbeat_seconds,
            ),
        )

        emit("============================================================")
        emit("FINAL RESULT")
        emit("============================================================")

        print(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                default=str,
            ),
            flush=True,
        )

        return (
            0
            if payload.get("status") == "HEALTHY"
            else 2
        )

    cycle = 0
    interval = max(60, args.poll_seconds)

    while True:
        cycle += 1

        emit(
            f"[CYCLE START] cycle={cycle}"
        )

        payload = run_with_progress(
            f"pi_unified_cycle_{cycle}",
            worker.run_once,
            heartbeat_seconds=max(
                5,
                args.heartbeat_seconds,
            ),
        )

        emit(
            f"[CYCLE RESULT] cycle={cycle} "
            f"status={payload.get('status')} "
            f"markets={payload.get('market_count')} "
            f"errors={len(payload.get('errors') or [])}"
        )

        remaining = interval

        while remaining > 0:
            wait = min(
                max(5, args.heartbeat_seconds),
                remaining,
            )

            time.sleep(wait)
            remaining -= wait

            emit(
                f"[IDLE] "
                f"next_cycle_in={remaining}s "
                f"{process_stats()}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
