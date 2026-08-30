from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import tempfile
import uuid

from .worker_protocol import WorkerRequest


SAFE_ENV = {
    "HOME", "PATH", "TMPDIR", "LANG", "LC_ALL", "TZ",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
}


def run_worker(
    python_executable: Path,
    worker: Path,
    action: str,
    payload: dict,
    timeout_seconds: int = 900,
    pass_env: tuple[str, ...] = (),
) -> dict:
    request = WorkerRequest("1.0", str(uuid.uuid4()), action, payload)
    env = {k: v for k, v in os.environ.items() if k in SAFE_ENV or k in pass_env}

    with tempfile.TemporaryDirectory() as td:
        req = Path(td) / "request.json"
        res = Path(td) / "response.json"
        req.write_text(request.dumps(), encoding="utf-8")

        proc = subprocess.run(
            [
                str(python_executable),
                str(worker),
                "--request",
                str(req),
                "--response",
                str(res),
            ],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Worker exited with code {proc.returncode}")
        response = json.loads(res.read_text(encoding="utf-8"))
        if response.get("request_id") != request.request_id:
            raise RuntimeError("Worker response request_id mismatch")
        return response
