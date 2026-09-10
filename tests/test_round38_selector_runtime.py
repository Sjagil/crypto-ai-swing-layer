from __future__ import annotations

import importlib.util
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


# Standalone overlay validation support. In the real repository the existing
# v0.27 entry_selector module is imported normally.
try:
    entry_spec = importlib.util.find_spec(
        "crypto_ai_swing.research.entry_selector"
    )
except ModuleNotFoundError:
    entry_spec = None
if entry_spec is None:
    fake_entry = types.ModuleType("crypto_ai_swing.research.entry_selector")
    fake_entry.SELECTOR_FEATURES = ("technical", "mtf", "orderflow")
    fake_entry.selector_feature_vector = lambda context: {}

    class FakeSelectorBase:
        pass

    fake_entry.ProspectiveSwingEntrySelector = FakeSelectorBase
    sys.modules["crypto_ai_swing.research.entry_selector"] = fake_entry

import crypto_ai_swing.research.selector_runtime as runtime_module  # noqa: E402
from crypto_ai_swing.research.selector_runtime import (  # noqa: E402
    ProspectiveSelectorRuntime,
    SelectorDecisionLedger,
    selector_context_quality,
)


TEST_FEATURES = ("technical", "mtf", "orderflow")


def fake_features(context):
    return {
        "technical": context.get("technical_feature"),
        "mtf": context.get("mtf_feature"),
        "orderflow": context.get("orderflow_feature"),
    }


@pytest.fixture(autouse=True)
def deterministic_test_features(monkeypatch):
    monkeypatch.setattr(runtime_module, "SELECTOR_FEATURES", TEST_FEATURES)
    monkeypatch.setattr(
        runtime_module,
        "selector_feature_vector",
        fake_features,
    )


class FakeSelector:
    def __init__(self, payload, decision):
        self._latest = dict(payload)
        self._decision = dict(decision)
        self.calls = 0

    def evaluate_context(self, context, *, policy=None):
        del context, policy
        self.calls += 1
        return dict(self._decision)


def settings(tmp_path: Path):
    return SimpleNamespace(
        project_root=tmp_path,
        autonomy={
            "entry_selector": {
                "runtime_minimum_feature_coverage": 2 / 3,
                "runtime_maximum_policy_age_seconds": 3600,
                "runtime_require_canonical_context": True,
                "runtime_require_core_sections": True,
            }
        },
    )


def policy():
    return {
        "status": "QUALIFIED",
        "qualified": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "model_contract_hash": "model-1",
    }


def full_context():
    return {
        "market": "BTC-EUR",
        "decision_at": "2026-09-10T00:00:00+00:00",
        "data_source": "Sjagil/crypto",
        "timeframe_pipeline": {"states": {"1h": {"score": 0.5}}},
        "mtf_challenger": {"score": 0.7},
        "universe_screen": {"execution_adjusted_score": 0.8},
        "technical_feature": 0.8,
        "mtf_feature": 0.7,
        "orderflow_feature": 0.6,
    }


def test_context_quality_counts_finite_features():
    quality = selector_context_quality(full_context())
    assert quality["feature_coverage"] == 1.0
    assert quality["canonical_source"] is True


def test_runtime_fail_closed_on_missing_canonical_context(tmp_path):
    selector = FakeSelector(
        policy(),
        {"status": "READY", "action": "SELECT_24H", "passes": True},
    )
    runtime = ProspectiveSelectorRuntime(
        settings(tmp_path), selector=selector, mode="paper"
    )
    try:
        context = full_context()
        context["data_source"] = "legacy-direct-client"
        decision = runtime.evaluate_context(context)
        assert decision["action"] == "ABSTAIN"
        assert decision["passes"] is False
        assert selector.calls == 0
        assert "CANONICAL_SOURCE_FAILED" in decision["reason_codes"]
        assert decision["live_decision_influence"] is False
    finally:
        runtime.close()


def test_runtime_records_qualified_advisory_decision_idempotently(tmp_path):
    selector = FakeSelector(
        policy(),
        {
            "status": "READY",
            "action": "SELECT_24H",
            "passes": True,
            "horizon_hours": 24,
            "reason_codes": [],
        },
    )
    runtime = ProspectiveSelectorRuntime(
        settings(tmp_path), selector=selector, mode="paper"
    )
    try:
        first = runtime.evaluate_context(full_context())
        second = runtime.evaluate_context(full_context())
        assert first["passes"] is True
        assert first["authority"] == "ADVISORY_ONLY"
        assert first["live_decision_influence"] is False
        assert first["selector_runtime"]["ledger_inserted"] is True
        assert second["selector_runtime"]["ledger_inserted"] is False
        status = runtime.status()
        assert status["decisions"] == 1
        assert status["selected"] == 1
        assert status["orders_submitted"] == 0
        assert status["raw_context_persisted"] is False
    finally:
        runtime.close()


def test_ledger_never_requires_raw_context(tmp_path):
    ledger = SelectorDecisionLedger(tmp_path / "selector.sqlite")
    try:
        inserted = ledger.record(
            {
                "decision_id": "abc",
                "recorded_at": datetime.now(UTC).isoformat(),
                "decision_at": None,
                "market": "ETH-EUR",
                "action": "ABSTAIN",
                "passes": False,
                "feature_coverage": 0.5,
                "context_hash": "hash",
                "policy_hash": None,
                "reason_codes": ["FEATURE_COVERAGE_FAILED"],
            }
        )
        assert inserted is True
        assert ledger.status()["decisions"] == 1
        assert ledger.status()["secrets_persisted"] is False
    finally:
        ledger.close()
