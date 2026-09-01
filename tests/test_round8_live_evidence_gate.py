from types import SimpleNamespace

from crypto_ai_swing.orchestration.proactive import ProactiveTrader


class FakeForward:
    def canary_readiness(self, **kwargs):
        return {
            "eligible": False,
            "status": "COLLECTING",
            "blockers": ["INSUFFICIENT_UNBLOCKED_BUY_OUTCOMES"],
        }


class FakeAuthority:
    def __init__(self):
        self.called = False

    def submit_buy(self, intent):
        self.called = True
        raise AssertionError("live authority must not be called before evidence gate")


def test_live_buy_submission_is_blocked_before_crypto_authority():
    trader = object.__new__(ProactiveTrader)
    trader.mode = "live"
    trader.settings = SimpleNamespace(
        autonomy={
            "forward_evidence": {
                "canary_readiness": {
                    "primary_horizon_hours": 4,
                    "minimum_unblocked_buy_outcomes": 30,
                    "minimum_distinct_markets": 5,
                    "minimum_observation_span_hours": 72,
                    "minimum_mean_return_bps": 0.0,
                    "minimum_positive_return_rate": 0.5,
                }
            }
        }
    )
    trader.forward = FakeForward()
    trader.execution_authority = FakeAuthority()
    intent = SimpleNamespace(intent_id="test-intent")

    result = trader._execute_buy(intent)

    assert result["accepted"] is False
    assert result["reason_code"] == "PROSPECTIVE_CANARY_EVIDENCE_NOT_READY"
    assert result["orders_submitted"] == 0
    assert trader.execution_authority.called is False
