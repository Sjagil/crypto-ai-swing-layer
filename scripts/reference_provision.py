from __future__ import annotations

import argparse
import json
from pathlib import Path

from crypto_ai_swing.bridge.reference_provision import provision_references, write_provision_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    payload = provision_references(
        root,
        venv_root=args.venv_root,
        repo_root=args.repo_root,
        names=args.name or None,
        apply=args.apply,
        include_disabled=False,
    )
    out = root / "output/crypto_ai_swing/references/provision_latest.json"
    write_provision_report(payload, out)
    payload["output"] = str(out)
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
