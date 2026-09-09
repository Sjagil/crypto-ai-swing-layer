#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _crypto_python(root: Path) -> Path:
    candidate = root / ".venv" / "bin" / "python"
    if not candidate.is_file():
        raise FileNotFoundError(f"canonical crypto venv missing: {candidate}")
    return candidate


def _strategy_ids(root: Path, python: Path) -> list[str]:
    result = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import json; "
                "from research.strategies import strategy_registry; "
                "print(json.dumps(sorted(strategy_registry())))"
            ),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "strategy registry discovery failed:\n" + result.stderr[-4000:]
        )
    return list(json.loads(result.stdout.strip().splitlines()[-1]))


def _run_one(
    *,
    root: Path,
    python: Path,
    strategy: str,
    mode: str,
    markets: str,
    timeframes: str,
    capital: str,
    timeout: int,
) -> dict[str, object]:
    deep = mode == "deep"
    trials = 50 if deep else 2
    folds = 6 if deep else 2
    bootstrap = 1000 if deep else 100
    monte_carlo = 1000 if deep else 100
    profile = "standard" if deep else "smoke"
    method = "random"
    output_root = (
        root / "output" / "research" / "operational-stack" / mode / strategy
    )
    checkpoint = (
        root
        / "output"
        / "research"
        / "operational-stack"
        / "checkpoints"
        / f"{strategy}.jsonl"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    command = [
        str(python),
        "main.py",
        "research",
        "--providers",
        "bitvavo,kraken",
        "--skip-download",
        "--scrapers",
        "none",
        "--markets",
        markets,
        "--timeframes",
        timeframes,
        "--strategies",
        strategy,
        "--profile",
        profile,
        "--capital",
        capital,
        "--risk-per-trade",
        "0.0065",
        "--fee",
        "0.0025",
        "--slippage-bps",
        "8",
        "--method",
        method,
        "--trials",
        str(trials),
        "--walk-forward-folds",
        str(folds),
        "--bootstrap-samples",
        str(bootstrap),
        "--monte-carlo-runs",
        str(monte_carlo),
        "--output-dir",
        str(output_root),
    ]
    if deep:
        command.extend(["--checkpoint", str(checkpoint)])
    else:
        command.extend(["--max-rows", "5000"])
    started = time.monotonic()
    print(
        f"[canonical-campaign] START {strategy} mode={mode} trials={trials}",
        file=sys.stderr,
        flush=True,
    )
    process = subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    timed_out = False
    stdout = ""
    stderr = ""
    while True:
        try:
            stdout, stderr = process.communicate(timeout=30)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            print(
                f"[canonical-campaign] HEARTBEAT {strategy} elapsed={elapsed:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            if elapsed >= timeout:
                timed_out = True
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                break

    duration = time.monotonic() - started
    status = (
        "TIMEOUT"
        if timed_out
        else "PASSED"
        if process.returncode == 0
        else "BLOCKED"
    )
    print(
        f"[canonical-campaign] END {strategy} status={status} duration={duration:.1f}s",
        file=sys.stderr,
        flush=True,
    )
    return {
        "strategy": strategy,
        "status": status,
        "returncode": None if timed_out else process.returncode,
        "duration_seconds": duration,
        "trials": trials,
        "profile": profile,
        "max_rows": None if deep else 5000,
        "checkpoint": str(checkpoint) if deep else None,
        "output_dir": str(output_root),
        "stdout_tail": stdout[-12000:],
        "stderr_tail": stderr[-6000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crypto-root",
        default="/Users/ayoubalhari/Downloads/crypto",
    )
    parser.add_argument(
        "--mode",
        choices=("acceptance", "deep"),
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
    parser.add_argument("--capital", default="2000")
    parser.add_argument("--strategy-timeout", type=int)
    args = parser.parse_args()

    root = Path(args.crypto_root).expanduser().resolve()
    python = _crypto_python(root)
    strategies = _strategy_ids(root, python)
    timeout = (
        args.strategy_timeout
        if args.strategy_timeout is not None
        else 3600
        if args.mode == "deep"
        else 300
    )

    rows = [
        _run_one(
            root=root,
            python=python,
            strategy=strategy,
            mode=args.mode,
            markets=args.markets,
            timeframes=args.timeframes,
            capital=args.capital,
            timeout=timeout,
        )
        for strategy in strategies
    ]
    blocked = [row for row in rows if row["status"] != "PASSED"]
    payload = {
        "schema_version": "canonical_research_campaign_v1",
        "status": "PASSED" if not blocked else "BLOCKED",
        "mode": args.mode,
        "strategy_count": len(strategies),
        "strategies_passed": len(rows) - len(blocked),
        "strategies_blocked": len(blocked),
        "markets": args.markets.split(","),
        "timeframes": args.timeframes.split(","),
        "rows": rows,
        "safety": {
            "spot_only": True,
            "long_only": True,
            "automatic_live_promotion": False,
            "paper_promotion_requested": False,
            "orders_submitted_by_campaign": 0,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blocked else 2


if __name__ == "__main__":
    raise SystemExit(main())
