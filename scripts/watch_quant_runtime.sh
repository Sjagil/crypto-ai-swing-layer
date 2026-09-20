#!/bin/zsh

ROOT="/Users/ayoubalhari/Downloads/crypto-ai-swing-layer"
PY="$ROOT/.venv/bin/python"

cd "$ROOT" || exit 1

while true; do
    clear

    echo "============================================================"
    echo " CRYPTO AI / QUANT RUNTIME"
    echo " $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    echo
    echo "=== ACTIVE LEARNING PROCESSES ==="

    PIDS="$(pgrep -f \
      'pi_learning_worker.py|pi_learning_worker_verbose.py|learning-worker' \
      || true)"

    if [ -z "$PIDS" ]; then
        echo "NO LEARNING WORKER RUNNING"
    else
        for PID in $PIDS; do
            ps -p "$PID" \
              -o pid=,ppid=,etime=,%cpu=,%mem=,rss=,state=,command=
        done
    fi

    echo
    echo "=== RECENT ARTIFACT ACTIVITY ==="

    find output/crypto_ai_swing \
      -type f \
      -mmin -2 \
      -print 2>/dev/null \
      | head -20

    echo
    echo "=== LEARNING HEARTBEAT ==="

    "$PY" - <<'PY'
import json
from pathlib import Path

paths = [
    Path("output/crypto_ai_swing/pi_learning_worker/heartbeat.json"),
    Path("output/crypto_ai_swing/learning_worker/heartbeat.json"),
]

for path in paths:
    if not path.exists():
        continue

    try:
        x = json.loads(path.read_text())

        print(f"{path}:")
        print("  status       =", x.get("status"))
        print("  cycle_id     =", x.get("cycle_id"))
        print("  pid          =", x.get("pid"))
        print("  started_at   =", x.get("started_at"))
        print("  completed_at =", x.get("completed_at"))

        tasks = x.get("tasks") or {}

        for name, task in tasks.items():
            if isinstance(task, dict):
                print(
                    f"  task {name:<24} "
                    f"status={task.get('status') or task.get('state')}"
                )

        errors = x.get("errors") or []

        if errors:
            print("  ERRORS:")
            for error in errors[-5:]:
                print("   ", error)

    except Exception as exc:
        print(path, "READ_ERROR", exc)
PY

    echo
    echo "=== CURRENT SHADOW PRODUCT OUTPUT ==="

    "$PY" - <<'PY'
import json
from pathlib import Path

path = Path(
    "output/crypto_ai_swing/modes/shadow/proactive/latest.json"
)

if not path.exists():
    path = Path(
        "output/crypto_ai_swing/proactive/latest.json"
    )

if not path.exists():
    print("No proactive/latest.json yet")
    raise SystemExit

x = json.loads(path.read_text())

signals = x.get("signals") or []
buys = [
    s for s in signals
    if str(s.get("side")).upper() == "BUY"
]

executions = x.get("executions") or []
blocked = x.get("blocked") or []
positions = x.get("positions") or {}

print("generated_at     =", x.get("generated_at"))
print("screened_markets =", x.get("screened_markets"))
print("signals          =", len(signals))
print("BUY signals      =", len(buys))
print("executions       =", len(executions))
print("blocked          =", len(blocked))
print("positions        =", len(positions))

print()
print("BUY CANDIDATES")

for s in sorted(
    buys,
    key=lambda z: float(z.get("score") or 0),
    reverse=True,
):
    print(
        f"  {s.get('market', ''):<10} "
        f"score={float(s.get('score') or 0):.3f} "
        f"edge={float(s.get('edge_bps') or 0):8.2f}bps"
    )

print()
print("SIMULATED EXECUTIONS")

for row in executions:
    intent = row.get("intent") or {}
    meta = intent.get("metadata") or {}
    risk = meta.get("canonical_portfolio_risk") or {}
    plan = risk.get("risk_plan") or {}

    print(
        f"  {intent.get('market', ''):<10} "
        f"raw={float(intent.get('expected_edge_bps') or 0):8.2f}bps "
        f"net={float(risk.get('net_edge_bps') or 0):8.2f}bps "
        f"notional=€{plan.get('order_notional_eur')}"
    )

print()
print("BLOCKED")

for row in blocked:
    print(
        f"  {row.get('market', ''):<10} "
        f"{','.join(row.get('blockers') or [])}"
    )

print()
print("POSITIONS")

for market, pos in positions.items():
    print(
        f"  {market:<10} "
        f"entry={pos.get('entry_price')} "
        f"highest={pos.get('highest_price')}"
    )
PY

    echo
    echo "=== MODEL / DATA STATE ==="

    "$PY" - <<'PY'
import json
from pathlib import Path

agent = Path(
    "output/crypto_ai_swing/agents/latest.json"
)

hpo = Path(
    "output/crypto_ai_swing/hpo/best.json"
)

tcn = Path(
    "output/crypto_ai_swing/agents/tcn_gru/latest.pointer.json"
)

for title, path in [
    ("SUPERVISED", agent),
    ("HPO", hpo),
    ("TCN_GRU", tcn),
]:
    print(title)

    if not path.exists():
        print("  NOT_FOUND", path)
        continue

    try:
        x = json.loads(path.read_text())

        print("  status     =", x.get("status"))
        print("  generated  =", x.get("generated_at"))
        print("  dataset_id =", x.get("dataset_id"))
        print("  qualified  =", x.get("qualified"))

    except Exception as exc:
        print("  ERROR =", exc)
PY

    echo
    echo "=== CMC PIT RESEARCH ==="

    "$PY" - <<'PY'
import json
from pathlib import Path

path = Path(
    "output/crypto_ai_swing/research/cmc_pit_feature_research.json"
)

if not path.exists():
    print("NO CMC RESEARCH ARTIFACT")
else:
    x = json.loads(path.read_text())

    print("status        =", x.get("status"))
    print("markets       =", x.get("market_count"))
    print("features      =", x.get("feature_count"))
    print("pairs         =", x.get("pair_count"))
    print("timeframe     =", x.get("timeframe"))
    print("horizon_bars  =", x.get("horizon_bars"))

    for row in (x.get("aggregate") or [])[:5]:
        print(
            " ",
            row.get("feature"),
            "median_ic=",
            row.get("median_ic"),
            "markets=",
            row.get("market_count"),
        )
PY

    echo
    echo "Refreshing in 15 seconds..."
    sleep 15
done
