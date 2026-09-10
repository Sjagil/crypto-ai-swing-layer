from pathlib import Path

from crypto_ai_swing.bridge.crypto_operations import REUSED_NATIVE_INTERFACES

ROOT = Path(__file__).resolve().parents[1]


def test_local_bitvavo_transport_is_removed():
    assert not (ROOT / "src/crypto_ai_swing/execution/bitvavo.py").exists()


def test_canonical_bitvavo_execution_contract_is_reused():
    names = set(REUSED_NATIVE_INTERFACES["execution.execution"])
    assert {"BitvavoSpotClient", "ExecutionMarketRules", "market_rules_from_bitvavo_metadata", "plan_bounded_entry_order"} <= names
