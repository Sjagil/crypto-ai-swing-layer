from crypto_ai_swing.research.native import NativeResearchBridge


class FakeCrypto:
    def __init__(self):
        self.called = None

    def import_module(self, name):
        parent = self

        class Module:
            @staticmethod
            def build_research_factory_artifact(settings, maximum_rows, execute_exact):
                parent.called = (settings, maximum_rows, execute_exact)
                return {
                    "schema_version": "x",
                    "run_id": "r",
                    "created_at": "now",
                    "stage0": {"tested_variant_count": 20, "survivors": [1], "status_counts": {}},
                    "exact_validation": {"status": "NOT_RUN_NO_STAGE0_SURVIVOR"},
                    "promotion_table": [{"promotion_state": "EXACT_VALIDATION"}],
                    "forward_candidates": [],
                    "missing_dataset_sources": [],
                }

        return Module

    def settings(self):
        return "settings"


def test_native_research_campaign_defaults_to_no_exact():
    bridge = object.__new__(NativeResearchBridge)
    bridge.crypto = FakeCrypto()
    payload = bridge.run_factory_campaign(maximum_rows=1234, execute_exact=False)
    assert bridge.crypto.called == ("settings", 1234, False)
    summary = bridge.factory_summary(payload)
    assert summary["stage0_survivors"] == 1
    assert summary["automatic_live_promotion"] is False
