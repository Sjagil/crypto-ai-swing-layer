#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def _configure_ml_cache_environment() -> dict[str, str]:
    # Normalize Hugging Face cache aliases for persistent macOS runtime.
    root = Path.home() / ".cache" / "huggingface"
    windows_path = re.compile(r"^[A-Za-z]:[\\/]")
    aliases = (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_ASSETS_CACHE",
        "HF_XET_CACHE",
        "SENTENCE_TRANSFORMERS_HOME",
        "TRANSFORMERS_CACHE",
    )
    for key in aliases:
        raw = str(os.environ.get(key) or "").strip()
        if raw and windows_path.match(raw):
            os.environ.pop(key, None)

    normalized = {
        "HF_HOME": root,
        "HF_HUB_CACHE": root / "hub",
        "HUGGINGFACE_HUB_CACHE": root / "hub",
        "HF_ASSETS_CACHE": root / "assets",
        "HF_XET_CACHE": root / "xet",
        "SENTENCE_TRANSFORMERS_HOME": root / "sentence_transformers",
    }
    for key, value in normalized.items():
        os.environ.setdefault(key, str(value))

    if not str(os.environ.get("TRANSFORMERS_CACHE") or "").strip():
        os.environ.pop("TRANSFORMERS_CACHE", None)

    return {key: str(os.environ[key]) for key in normalized}


_CONFIGURED_ML_CACHE = _configure_ml_cache_environment()

from crypto_ai_swing.orchestration.round43_controller import (
    Round43AutomationController,
)
from crypto_ai_swing.settings import Settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("shadow", "paper"),
        default="shadow",
    )
    parser.add_argument("--forever", action="store_true")
    args = parser.parse_args()

    settings = Settings.load(Path.cwd())
    # Canonical crypto environment hydration happens inside Settings.load().
    # Re-sanitize cache-only variables afterwards so a stale Windows path
    # cannot be recreated as a literal directory on macOS.
    _configure_ml_cache_environment()
    controller = Round43AutomationController(
        settings,
        mode=args.mode,
    )
    try:
        if args.forever:
            controller.run_forever()
            return 0
        payload = controller.run_once()
        print(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return 0 if payload.get("state") != "DEGRADED" else 2
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
