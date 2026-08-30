from __future__ import annotations

from pathlib import Path
import json
import subprocess
import tempfile


def build_kronos_request(market: str, parquet_path: Path, horizon: int, config: dict) -> dict:
    return {
        "protocol_version": "1.0",
        "action": "forecast",
        "market": market,
        "parquet_path": str(parquet_path.resolve()),
        "horizon": int(horizon),
        "model": config.get("model", "NeoQuasar/Kronos-mini"),
        "tokenizer": config.get("tokenizer", "NeoQuasar/Kronos-Tokenizer-2k"),
        "lookback": int(config.get("lookback", 256)),
        "execution_authority": "NONE",
    }


def run_isolated_worker(
    python_executable: Path,
    worker: Path,
    request: dict,
    timeout: int = 3600,
) -> dict:
    with tempfile.TemporaryDirectory() as td:
        req = Path(td) / "request.json"
        res = Path(td) / "response.json"
        req.write_text(json.dumps(request), encoding="utf-8")
        proc = subprocess.run(
            [str(python_executable), str(worker), "--request", str(req), "--response", str(res)],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Kronos worker failed with code {proc.returncode}")
        return json.loads(res.read_text(encoding="utf-8"))
