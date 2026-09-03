from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crypto_ai_swing.bridge.native_foundation import NativeFoundationBridge
from crypto_ai_swing.research.native_dependencies import (
    inspect_native_campaign_dependencies,
)

DEFAULT_CAMPAIGNS = (
    "residual-momentum",
    "residual-reversal",
    "peer-residual-reversal",
    "multi-horizon-trend",
    "multi-alpha-v2",
)


def _forward_collecting(summary: dict[str, Any]) -> bool:
    forward = dict(summary.get("forward_evidence") or {})
    statuses = {
        str(value).upper()
        for value in (forward.get("statuses") or [])
        if value is not None
    }
    return "COLLECTING_FORWARD_DATA" in statuses


def _classification(payload: dict[str, Any]) -> str:
    if payload.get("status") == "BLOCKED":
        return "BLOCKED_INFRASTRUCTURE"

    summary = dict(payload.get("summary") or {})
    if bool(summary.get("live_ready")):
        return "LIVE_READY_NATIVE_EVIDENCE"
    if bool(summary.get("paper_candidate_permitted")):
        return "PAPER_CANDIDATE_NATIVE_EVIDENCE"
    if summary.get("research_pass") is True:
        return "RESEARCH_EVIDENCE_PASSED"

    economic = summary.get("economic_pass")
    statistical = summary.get("statistical_pass")
    collecting = _forward_collecting(summary)

    if economic is True and statistical is True:
        return (
            "FORWARD_EVIDENCE_COLLECTING"
            if collecting
            else "HISTORICAL_EVIDENCE_PASSED"
        )
    if collecting and (economic is False or statistical is False):
        return "HISTORICAL_GATES_FAILED_FORWARD_COLLECTING"
    if collecting:
        return "FORWARD_EVIDENCE_COLLECTING_UNQUALIFIED"
    return "REJECTED_NATIVE_EVIDENCE"


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
            preflight = inspect_native_campaign_dependencies(
                crypto_repo_root=crypto_repo_root,
                campaign=selected,
            )
        except Exception as exc:  # noqa: BLE001
            preflight = {
                "schema_version": "crypto_ai_swing_native_campaign_dependencies_v1",
                "campaign": selected,
                "ready": False,
                "classification": "BLOCKED_DEPENDENCY_INSPECTION",
                "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                "repair_commands": [],
                "orders_generated": 0,
                "orders_submitted": 0,
            }

        if not bool(preflight.get("ready")):
            rows.append(
                {
                    "campaign": selected,
                    "classification": (
                        "BLOCKED_PREREQUISITE_EVIDENCE"
                        if preflight.get("classification")
                        == "BLOCKED_PREREQUISITE_EVIDENCE"
                        else "BLOCKED_INFRASTRUCTURE"
                    ),
                    "summary": None,
                    "source": None,
                    "dependency_preflight": preflight,
                    "error": preflight.get("error"),
                }
            )
            continue

        try:
            payload = bridge.run_alpha_campaign(selected)
            row = {
                "campaign": selected,
                "classification": _classification(payload),
                "summary": payload.get("summary"),
                "source": payload.get("source"),
                "dependency_preflight": preflight,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            row = {
                "campaign": selected,
                "classification": "BLOCKED_INFRASTRUCTURE",
                "summary": None,
                "source": None,
                "dependency_preflight": preflight,
                "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
            }
        rows.append(row)

    classifications = [row["classification"] for row in rows]
    return {
        "schema_version": "crypto_ai_swing_native_alpha_tournament_v3",
        "generated_at": datetime.now(UTC).isoformat(),
        "campaigns": rows,
        "counts": {
            "total": len(rows),
            "blocked_prerequisite_evidence": classifications.count(
                "BLOCKED_PREREQUISITE_EVIDENCE"
            ),
            "blocked_infrastructure": classifications.count(
                "BLOCKED_INFRASTRUCTURE"
            ),
            "historical_gates_failed_forward_collecting": classifications.count(
                "HISTORICAL_GATES_FAILED_FORWARD_COLLECTING"
            ),
            "forward_evidence_collecting": classifications.count(
                "FORWARD_EVIDENCE_COLLECTING"
            ),
            "forward_evidence_collecting_unqualified": classifications.count(
                "FORWARD_EVIDENCE_COLLECTING_UNQUALIFIED"
            ),
            "historical_evidence_passed": classifications.count(
                "HISTORICAL_EVIDENCE_PASSED"
            ),
            "rejected_native_evidence": classifications.count(
                "REJECTED_NATIVE_EVIDENCE"
            ),
            "research_evidence_passed": classifications.count(
                "RESEARCH_EVIDENCE_PASSED"
            ),
            "paper_candidates": classifications.count(
                "PAPER_CANDIDATE_NATIVE_EVIDENCE"
            ),
            "live_ready": classifications.count(
                "LIVE_READY_NATIVE_EVIDENCE"
            ),
        },
        "ranking_policy": "NO_SYNTHETIC_SCORE_USE_NATIVE_EVIDENCE_ONLY",
        "classification_policy": (
            "SEPARATE_HISTORICAL_GATES_FROM_FORWARD_DIAGNOSTIC_COLLECTION"
        ),
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
