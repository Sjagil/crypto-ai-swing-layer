from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", ".venvs", "__pycache__", ".pytest_cache"}

for path in sorted(ROOT.rglob("*")):
    if any(part in SKIP for part in path.parts):
        continue
    rel = path.relative_to(ROOT)
    depth = len(rel.parts) - 1
    print("  " * depth + ("[D] " if path.is_dir() else "[F] ") + rel.name)
