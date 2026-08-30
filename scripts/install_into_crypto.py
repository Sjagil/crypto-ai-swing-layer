from __future__ import annotations

from pathlib import Path
import argparse
import shutil


COPY_ITEMS = [
    "src/crypto_ai_swing",
    "config/swing.yaml",
    "config/risk.yaml",
    "config/universe.yaml",
    "config/models.yaml",
    "config/research.yaml",
    "config/execution.yaml",
    "config/data_quality.yaml",
    "config/integrations.yaml",
    "config/references.yaml",
    "config/compliance.yaml",
    "scripts/clone_references.py",
    "scripts/workers/reference_health_worker.py",
    "docs",
    "tests",
]


def copy_item(src: Path, dst: Path, overwrite: bool) -> None:
    if src.is_dir():
        if dst.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite directory: {dst}")
        shutil.copytree(src, dst, dirs_exist_ok=overwrite)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite file: {dst}")
        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = Path(__file__).resolve().parents[1]
    target = args.target.resolve()
    if not target.exists():
        raise SystemExit(f"Target repository does not exist: {target}")

    for rel in COPY_ITEMS:
        src = source_root / rel
        if src.exists():
            copy_item(src, target / rel, args.overwrite)
            print("COPIED", rel)
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
