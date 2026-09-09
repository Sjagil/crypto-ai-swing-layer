#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ADVISORY_PHASES = {
    "lab_trials_audit",
    "crypto_build_reference_integration_phase_a",
    "crypto_build_reference_master_map",
}
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_ai_swing.bridge.reference_stack import ReferenceStack
from crypto_ai_swing.settings import Settings


def _run(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    category: str,
) -> dict[str, Any]:
    """Run one phase while streaming child stderr progress to the operator."""

    from collections import deque
    import threading

    started = datetime.now(UTC)
    print(
        f"[operational-stack] START {name}: {' '.join(command)}",
        file=sys.stderr,
        flush=True,
    )
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout_lines: deque[str] = deque(maxlen=1000)
    stderr_lines: deque[str] = deque(maxlen=1000)

    def pump(stream: Any, sink: deque[str], *, echo: bool) -> None:
        if stream is None:
            return
        for line in iter(stream.readline, ""):
            sink.append(line)
            if echo:
                print(
                    f"[{name}] {line}",
                    file=sys.stderr,
                    end="",
                    flush=True,
                )
        stream.close()

    stdout_thread = threading.Thread(
        target=pump,
        args=(process.stdout, stdout_lines),
        kwargs={"echo": False},
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=pump,
        args=(process.stderr, stderr_lines),
        kwargs={"echo": True},
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    duration = (datetime.now(UTC) - started).total_seconds()
    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)

    if timed_out:
        print(
            f"[operational-stack] TIMEOUT {name} ({duration:.1f}s)",
            file=sys.stderr,
            flush=True,
        )
        return {
            "name": name,
            "category": category,
            "status": "TIMEOUT",
            "returncode": None,
            "duration_seconds": duration,
            "stdout_tail": stdout[-12000:],
            "stderr_tail": stderr[-12000:],
        }

    status = "PASSED" if process.returncode == 0 else "BLOCKED"
    print(
        f"[operational-stack] END {name}: {status} ({duration:.1f}s)",
        file=sys.stderr,
        flush=True,
    )
    return {
        "name": name,
        "category": category,
        "status": status,
        "returncode": process.returncode,
        "duration_seconds": duration,
        "stdout_tail": stdout[-12000:],
        "stderr_tail": stderr[-12000:],
    }




def _crypto_cmd(crypto: Path, *args: str) -> list[str]:
    python = (
        crypto / ".venv" / "bin" / "python"
        if (crypto / ".venv" / "bin" / "python").is_file()
        else Path(sys.executable)
    )
    return [str(python), str(crypto / "main.py"), *args]


def _script_cmd(*parts: str) -> list[str]:
    return [sys.executable, str(ROOT.joinpath(*parts))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("acceptance", "research", "full", "live-readiness"),
        default="acceptance",
    )
    parser.add_argument(
        "--markets",
        default="BTC-EUR,ETH-EUR,SOL-EUR,LINK-EUR",
    )
    parser.add_argument(
        "--timeframes",
        default="15m,1h,2h,4h,1d,1W",
    )
    parser.add_argument(
        "--capital",
        default="2000",
    )
    parser.add_argument(
        "--download-history",
        action="store_true",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
    )
    parser.add_argument(
        "--deep-research",
        action="store_true",
        help="Opt into long standard research; default research is bounded.",
    )
    args = parser.parse_args()

    settings = Settings.load(ROOT)
    crypto = settings.crypto_repo_root.resolve()
    results: list[dict[str, Any]] = []

    references = ReferenceStack(ROOT).status()
    results.append(
        {
            "name": "reference_stack",
            "category": "code",
            "status": "PASSED" if references["ready"] else "BLOCKED",
            "returncode": 0 if references["ready"] else 2,
            "payload": references,
        }
    )

    results.append(
        _run(
            "crypto_capabilities",
            _script_cmd(
                "scripts",
                "workers",
                "crypto_capability_worker.py",
            ),
            cwd=ROOT,
            timeout=300,
            category="code",
        )
    )

    base = [
        ("doctor", ("doctor",), "code"),
        ("config_validate", ("config", "validate"), "code"),
        ("system_audit", ("system", "audit"), "code"),
        ("data_providers", ("data", "providers"), "data"),
        ("data_status", ("data", "status", "--compact"), "data"),
        ("features_build", ("features", "build"), "data"),
        ("strategies_list", ("strategies", "list"), "research"),
        ("paper_status", ("paper", "status"), "paper"),
        ("lab_trials_audit", ("lab", "trials", "audit"), "research"),
    ]
    for name, command, category in base:
        results.append(
            _run(
                name,
                _crypto_cmd(crypto, *command),
                cwd=crypto,
                timeout=min(args.timeout, 900),
                category=category,
            )
        )

    reference_scripts = (
        "build_reference_integration_phase_a.py",
        "build_reference_master_map.py",
        "build_reference_integration_health.py",
    )
    for script in reference_scripts:
        path = crypto / "scripts" / script
        if path.is_file():
            results.append(
                _run(
                    f"crypto_{path.stem}",
                    [sys.executable, str(path)],
                    cwd=crypto,
                    timeout=min(args.timeout, 900),
                    category="reference",
                )
            )

    if args.profile in {"research", "full"}:
        results.append(
            _run(
                "history_sync",
                _crypto_cmd(
                    crypto,
                    "history",
                    "download",
                    "--min-years",
                    "7",
                    "--markets",
                    args.markets,
                    "--timeframes",
                    args.timeframes,
                    "--providers",
                    "bitvavo,kraken",
                    "--resume",
                ),
                cwd=crypto,
                timeout=args.timeout,
                category="data",
            )
        )
        results.append(
            _run(
                "history_audit",
                _crypto_cmd(
                    crypto,
                    "history",
                    "audit",
                    "--min-years",
                    "7",
                    "--markets",
                    args.markets,
                    "--timeframes",
                    args.timeframes,
                ),
                cwd=crypto,
                timeout=args.timeout,
                category="data",
            )
        )

        research_campaign = [
            sys.executable,
            str(ROOT / "scripts" / "run_canonical_research_campaign.py"),
            "--crypto-root",
            str(crypto),
            "--mode",
            "deep" if args.deep_research else "acceptance",
            "--markets",
            args.markets,
            "--timeframes",
            args.timeframes,
            "--capital",
            args.capital,
        ]
        results.append(
            _run(
                "canonical_research",
                research_campaign,
                cwd=ROOT,
                timeout=(
                    args.timeout
                    if args.deep_research
                    else min(args.timeout, 1800)
                ),
                category="research",
            )
        )

        results.append(
            _run(
                "lab_real_data_once",
                _crypto_cmd(
                    crypto,
                    "lab",
                    "run",
                    "--data-mode",
                    "real",
                    "--profile",
                    "hypotheses" if args.deep_research else "quick",
                    "--markets",
                    args.markets,
                    "--combination-sizes",
                    "1,2",
                    "--timeframes",
                    "1h,4h",
                    "--workers",
                    "2",
                    "--only-missing",
                    "--resume",
                    "--once",
                ),
                cwd=crypto,
                timeout=(args.timeout if args.deep_research else min(args.timeout, 1800)),
                category="strategy_generation",
            )
        )

    if args.profile == "full":
        for name, command in (
            ("backtest", ("backtest",)),
            ("optimize", ("optimize",)),
            ("walk_forward", ("walk-forward",)),
            ("monte_carlo", ("monte-carlo",)),
            (
                "validate_survivors",
                (
                    "research",
                    "validate-survivors",
                    "--min-years",
                    "7",
                    "--resume",
                ),
            ),
            (
                "leaderboard",
                ("leaderboard", "build", "--window", "seven-year"),
            ),
            (
                "report",
                ("report", "build", "--scope", "seven-year"),
            ),
            (
                "strategies_top",
                ("strategies", "top", "--limit", "20"),
            ),
        ):
            results.append(
                _run(
                    name,
                    _crypto_cmd(crypto, *command),
                    cwd=crypto,
                    timeout=args.timeout,
                    category="research",
                )
            )

    readiness_command = _script_cmd(
        "scripts",
        "full_stack_readiness.py",
    )
    if args.profile == "live-readiness":
        readiness_command.append("--live")
    results.append(
        _run(
            "full_stack_readiness",
            readiness_command,
            cwd=ROOT,
            timeout=min(args.timeout, 900),
            category=(
                "live_readiness"
                if args.profile == "live-readiness"
                else "readiness"
            ),
        )
    )

    if args.profile == "live-readiness":
        results.append(
            _run(
                "canonical_live_preflight",
                _crypto_cmd(crypto, "live", "preflight"),
                cwd=crypto,
                timeout=min(args.timeout, 900),
                category="live_readiness",
            )
        )

    hard_categories = {"code"}
    hard_failures = [
        row for row in results
        if row.get("category") in hard_categories
        and row.get("status") != "PASSED"
    ]
    blocked = [
        row for row in results
        if row.get("status") != "PASSED"
    ]
    advisory_blocked = [
        row for row in blocked
        if row.get("name") in ADVISORY_PHASES
    ]
    required_blocked = [
        row for row in blocked
        if row.get("name") not in ADVISORY_PHASES
    ]

    status = (
        "CODE_BLOCKED"
        if hard_failures
        else "OPERATIONAL_WITH_DATA_OR_EXTERNAL_BLOCKERS"
        if required_blocked
        else "READY_WITH_ADVISORIES"
        if advisory_blocked
        else "READY"
    )
    payload = {
        "schema_version": "crypto_ai_swing_operational_stack_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "profile": args.profile,
        "status": status,
        "code_ready": not hard_failures,
        "all_phases_ready": not required_blocked,
        "advisory_phases_ready": not advisory_blocked,
        "markets": args.markets.split(","),
        "timeframes": args.timeframes.split(","),
        "results": results,
        "blocked_phases": [
            {
                "name": row.get("name"),
                "category": row.get("category"),
                "status": row.get("status"),
                "returncode": row.get("returncode"),
            }
            for row in required_blocked
        ],
        "advisory_blocked_phases": [
            {
                "name": row.get("name"),
                "category": row.get("category"),
                "status": row.get("status"),
                "returncode": row.get("returncode"),
            }
            for row in advisory_blocked
        ],
        "safety": {
            "spot_only": True,
            "long_only": True,
            "automatic_live_promotion": False,
            "autoscale_authorized": False,
            "live_orders_submitted_by_runner": 0,
            "live_mutation_commands_executed": False,
        },
    }
    output = (
        ROOT
        / "output"
        / "crypto_ai_swing"
        / "operations"
        / "operational_stack_latest.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    payload["artifact"] = str(output)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if not hard_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
