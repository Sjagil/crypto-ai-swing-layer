from __future__ import annotations


SYSTEM_BOUNDARY = (
    "You summarize already-computed crypto research evidence. "
    "You cannot approve, size, route or submit trades. "
    "You cannot override risk, compliance, data-quality or execution gates."
)


def build_summary_payload(signal: dict, research_metrics: dict, blockers: list[str]) -> dict:
    return {
        "system_boundary": SYSTEM_BOUNDARY,
        "signal": signal,
        "research_metrics": research_metrics,
        "blockers": blockers,
        "execution_authority": "NONE",
    }
