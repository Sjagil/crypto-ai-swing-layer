from crypto_ai_swing.quant.bayesian import beta_binomial_posterior, normal_inverse_gamma_posterior
from crypto_ai_swing.agents.panel import build_research_agent_panel

def test_bayesian_hit_rate():
    p = beta_binomial_posterior(8, 10)
    assert p.posterior_mean > .5 and p.probability_above_half > .8

def test_bayesian_positive_mean():
    p = normal_inverse_gamma_posterior([.01, .02, .015, .007, .013])
    assert p.posterior_mean > 0 and p.probability_mean_positive > .5

def test_panel_is_advisory():
    x = build_research_agent_panel(predictions={"alpha_probability": .7, "regime_probability": .7, "predicted_return": .02, "predicted_mae": .01}, context={"spread_bps": 3, "book_imbalance": .2, "cvd_ratio": .1, "microprice_edge_bps": 1}, bayesian_evidence={"mean_posterior": {"probability_mean_positive": .8}}, stochastic_evidence={"passed": True})
    assert x["authority"] == "ADVISORY_ONLY" and x["live_decision_influence"] is False
