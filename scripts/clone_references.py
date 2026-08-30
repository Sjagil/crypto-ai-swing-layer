from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Also clone disabled references")
    parser.add_argument("--depth", type=int, default=1)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config/references.yaml").read_text(encoding="utf-8"))
    refs = cfg["references"]

    failures = []
    for name, spec in refs.items():
        if not spec.get("enabled", False) and not args.all:
            print(f"SKIP {name}: disabled")
            continue

        target = root / spec["directory"]
        if target.exists():
            print(f"EXISTS {name}: {target}")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "git", "clone", "--depth", str(args.depth),
            spec["repository"], str(target),
        ]
        print("CLONE", name, spec["repository"])
        proc = subprocess.run(cmd, shell=False, check=False)
        if proc.returncode != 0:
            failures.append(name)

    if failures:
        print("FAILED:", ", ".join(failures))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
