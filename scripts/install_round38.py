from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EDGE = ROOT / "src/crypto_ai_swing/agents/edge_manager.py"
AUTONOMY = ROOT / "config/autonomy.yaml"


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def patch_edge_manager() -> bool:
    original = EDGE.read_text(encoding="utf-8")
    text = original
    import_anchor = (
        "from crypto_ai_swing.research.entry_selector import "
        "ProspectiveSwingEntrySelector\n"
    )
    import_replacement = (
        import_anchor
        + "from crypto_ai_swing.research.selector_runtime import "
        "ProspectiveSelectorRuntime\n"
    )
    text = replace_once(
        text,
        import_anchor,
        import_replacement,
        label="selector runtime import",
    )

    init_anchor = (
        "        self.entry_selector = ProspectiveSwingEntrySelector(\n"
        "            settings, mode=mode\n"
        "        )\n"
    )
    init_replacement = (
        init_anchor
        + "        self.selector_runtime = ProspectiveSelectorRuntime(\n"
        "            settings, selector=self.entry_selector, mode=mode\n"
        "        )\n"
    )
    text = replace_once(
        text,
        init_anchor,
        init_replacement,
        label="selector runtime init",
    )

    eval_anchor = (
        "        selector_engine = getattr(self, \"entry_selector\", None)\n"
        "        selector_gate = (\n"
        "            selector_engine.evaluate_context(context)\n"
        "            if selector_engine is not None\n"
        "            else {\"status\": \"COLLECTING\", \"passes\": False}\n"
        "        )\n"
    )
    eval_replacement = (
        "        selector_engine = getattr(self, \"entry_selector\", None)\n"
        "        selector_runtime = getattr(self, \"selector_runtime\", None)\n"
        "        if selector_runtime is not None:\n"
        "            selector_gate = selector_runtime.evaluate_context(context)\n"
        "        else:\n"
        "            selector_gate = (\n"
        "                selector_engine.evaluate_context(context)\n"
        "                if selector_engine is not None\n"
        "                else {\"status\": \"COLLECTING\", \"passes\": False}\n"
        "            )\n"
    )
    text = replace_once(
        text,
        eval_anchor,
        eval_replacement,
        label="selector runtime evaluation",
    )

    if text != original:
        EDGE.write_text(text, encoding="utf-8")
        return True
    return False


def patch_autonomy() -> bool:
    original = AUTONOMY.read_text(encoding="utf-8")
    if "runtime_ledger_path:" in original:
        return False
    anchor = "  minimum_feature_coverage: 0.6\n"
    insertion = (
        anchor
        + "  runtime_minimum_feature_coverage: 0.6\n"
        + "  runtime_maximum_policy_age_seconds: 3600\n"
        + "  runtime_require_canonical_context: true\n"
        + "  runtime_require_core_sections: true\n"
        + "  runtime_ledger_path: "
        "output/crypto_ai_swing/research/entry_selector/runtime_decisions.sqlite\n"
    )
    # `minimum_feature_coverage` exists in swing_geometry and entry_selector.
    # Patch the final occurrence, which belongs to entry_selector.
    index = original.rfind(anchor)
    if index < 0:
        raise RuntimeError("autonomy entry_selector feature-coverage anchor missing")
    text = original[:index] + insertion + original[index + len(anchor) :]
    AUTONOMY.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    edge_changed = patch_edge_manager()
    autonomy_changed = patch_autonomy()
    print(
        {
            "status": "APPLIED",
            "edge_manager_changed": edge_changed,
            "autonomy_changed": autonomy_changed,
            "live_decision_influence": False,
            "automatic_live_promotion": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
