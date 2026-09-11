
from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
from pathlib import Path

LABEL = "com.sjagil.crypto.round44-mdpro"


def paths():
    swing = Path(__file__).resolve().parents[1]
    crypto = Path(os.getenv("CRYPTO_REPO_PATH", swing.parent / "crypto")).expanduser().resolve()
    python = swing / ".venv/bin/python"
    plist = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
    logdir = swing / "output/crypto_ai_swing/round44/logs"
    universe = (
        swing
        / "output/crypto_ai_swing/modes/shadow/proactive/latest.json"
    )
    return swing, crypto, python, plist, logdir, universe


def install():
    swing, crypto, python, plist, logdir, universe = paths()
    if not python.is_file():
        raise SystemExit("ROUND44_VENV_PYTHON_MISSING")
    runner = crypto / "scripts/run_bitvavo_mdpro_orderflow.py"
    if not runner.is_file():
        raise SystemExit("ROUND44_CANONICAL_MDPRO_RUNNER_MISSING")
    logdir.mkdir(parents=True, exist_ok=True)
    plist.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Label": LABEL,
        "ProgramArguments": [
            "/usr/bin/caffeinate",
            "-s",
            str(python),
            str(runner),
            "--universe-json",
            str(universe),
            "--depth",
            "1000",
        ],
        "WorkingDirectory": str(crypto),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(logdir / "mdpro_stdout.log"),
        "StandardErrorPath": str(logdir / "mdpro_stderr.log"),
        "EnvironmentVariables": {
            "PYTHONUNBUFFERED": "1",
            "CRYPTO_REPO_PATH": str(crypto),
        },
    }
    plist.write_bytes(plistlib.dumps(payload))
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}", str(plist)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{uid}", str(plist)],
        check=True,
    )
    print(
        json.dumps(
            {
                "status": "INSTALLED",
                "label": LABEL,
                "plist": str(plist),
                "automatic_live_authority": False,
            },
            indent=2,
        )
    )


def uninstall():
    _, _, _, plist, _, _ = paths()
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}", str(plist)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    plist.unlink(missing_ok=True)
    print(json.dumps({"status": "UNINSTALLED", "label": LABEL}, indent=2))


def status():
    _, _, _, plist, _, _ = paths()
    uid = os.getuid()
    result = subprocess.run(
        ["launchctl", "print", f"gui/{uid}/{LABEL}"],
        text=True,
        capture_output=True,
        check=False,
    )
    print(
        json.dumps(
            {
                "status": "RUNNING" if result.returncode == 0 else "NOT_RUNNING",
                "label": LABEL,
                "plist_exists": plist.is_file(),
                "launchctl_excerpt": result.stdout[:6000],
                "automatic_live_authority": False,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("install", "uninstall", "status"))
    args = parser.parse_args()
    {"install": install, "uninstall": uninstall, "status": status}[args.action]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
