#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--swing-root", default=".")
    p.add_argument("--crypto-root", required=True)
    args = p.parse_args()
    swing = Path(args.swing_root).expanduser().resolve()
    crypto = Path(args.crypto_root).expanduser().resolve()

    sha = git(crypto, "rev-parse", "HEAD")
    branch = git(crypto, "branch", "--show-current") or "full-live-quant-machine"
    if git(crypto, "status", "--porcelain"):
        raise SystemExit("Canonical crypto repo must be clean before pinning its HEAD")

    path = swing / "config/canonical_crypto_engine.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["commit"] = sha
    data["branch"] = branch
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PINNED_CANONICAL_COMMIT={sha}")
    print(f"PINNED_CANONICAL_BRANCH={branch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
