from pathlib import Path

from crypto_ai_swing.research.quant_foundation import QuantFoundationAudit
from crypto_ai_swing.settings import Settings


def test_round42_quant_foundation_uses_pinned_canonical_crypto():
    settings = Settings.load(Path.cwd())
    payload = QuantFoundationAudit(settings).run()
    assert payload["ready"] is True
    assert payload["missing_research_modules"] == []
    assert payload["native_research"]["required_scope_ready"] is True
    assert "catalog_ready" in payload["native_research"]
    assert payload["contracts"]["canonical_kelly"] is True
    assert payload["contracts"]["canonical_backtester"] is True
    assert payload["contracts"]["canonical_walk_forward"] is True
    assert payload["contracts"]["canonical_stochastic_validation"] is True
    assert payload["orders_submitted"] == 0
