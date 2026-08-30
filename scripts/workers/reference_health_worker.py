from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    payload = request.get("payload", {})
    repo = Path(payload.get("repo", ".")).resolve()

    commit = None
    if (repo / ".git").exists():
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            commit = proc.stdout.strip()

    response = {
        "protocol_version": request["protocol_version"],
        "request_id": request["request_id"],
        "ok": repo.exists(),
        "payload": {
            "repo": str(repo),
            "exists": repo.exists(),
            "git_commit": commit,
            "execution_authority": "NONE",
        },
        "error": None if repo.exists() else "REFERENCE_REPO_NOT_FOUND",
    }
    args.response.parent.mkdir(parents=True, exist_ok=True)
    args.response.write_text(json.dumps(response, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
