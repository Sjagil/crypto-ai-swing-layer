from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crypto_ai_swing.bridge.native_foundation import NativeFoundationBridge

DEFAULT_CAMPAIGNS = (
    "residual-momentum",
    "residual-reversal",
    "peer-residual-reversal",
    "multi-horizon-trend",
    "multi-alpha-v2",
)


def _classification(payload: dict[str, Any]) -> str:
    if payload.get("status") == "BLOCKED":
        return "BLOCKED_INFRASTRUCTURE"
    summary = dict(payload.get("summary") or {})
    if bool(summary.get("live_ready")):
        return "LIVE_READY_NATIVE_EVIDENCE"
    if bool(summary.get("paper_candidate_permitted")):
        return "PAPER_CANDIDATE_NATIVE_EVIDENCE"
    if (
        summary.get("statistical_pass") is True
        and summary.get("economic_pass") is True
    ):
        return "RESEARCH_EVIDENCE_PASSED"
    return "REJECTED_OR_COLLECTING_EVIDENCE"


def run_native_alpha_tournament(
    *,
    crypto_repo_root: Path,
    campaigns: Iterable[str] = DEFAULT_CAMPAIGNS,
) -> dict[str, Any]:
    bridge = NativeFoundationBridge(crypto_repo_root)
    rows: list[dict[str, Any]] = []
    for campaign in campaigns:
        selected = str(campaign).strip().lower()
        if not selected:
            continue
        try:
            payload = bridge.run_alpha_campaign(selected)
            row = {
                "campaign": selected,
                "classification": _classification(payload),
                "summary": payload.get("summary"),
                "source": payload.get("source"),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            row = {
                "campaign": selected,
                "classification": "BLOCKED_INFRASTRUCTURE",
                "summary": None,
                "source": None,
                "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
            }
        rows.append(row)

    return {
        "schema_version": "crypto_ai_swing_native_alpha_tournament_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "campaigns": rows,
        "counts": {
            "total": len(rows),
            "blocked_infrastructure": sum(
                row["classification"] == "BLOCKED_INFRASTRUCTURE"
                for row in rows
            ),
            "rejected_or_collecting": sum(
                row["classification"]
                == "REJECTED_OR_COLLECTING_EVIDENCE"
                for row in rows
            ),
            "research_evidence_passed": sum(
                row["classification"] == "RESEARCH_EVIDENCE_PASSED"
                for row in rows
            ),
            "paper_candidates": sum(
                row["classification"]
                == "PAPER_CANDIDATE_NATIVE_EVIDENCE"
                for row in rows
            ),
            "live_ready": sum(
                row["classification"] == "LIVE_READY_NATIVE_EVIDENCE"
                for row in rows
            ),
        },
        "ranking_policy": "NO_SYNTHETIC_SCORE_USE_NATIVE_EVIDENCE_ONLY",
        "authority": "RESEARCH_ONLY",
        "live_decision_influence": False,
        "automatic_live_promotion": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def write_tournament(
    project_root: Path,
    payload: dict[str, Any],
) -> Path:
    root = (
        Path(project_root)
        / "output/crypto_ai_swing/research/native_alpha_tournament"
    )
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    history = root / f"{timestamp}.json"
    latest = root / "latest.json"
    text = json.dumps(payload, indent=2, default=str)
    history.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")
    return latest
