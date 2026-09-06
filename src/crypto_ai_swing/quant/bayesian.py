from __future__ import annotations
from dataclasses import asdict, dataclass
from math import sqrt
from typing import Iterable
import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import t as student_t

@dataclass(frozen=True)
class BetaBinomialPosterior:
    successes: int
    trials: int
    alpha: float
    beta: float
    posterior_mean: float
    credible_interval: tuple[float, float]
    probability_above_half: float
    def to_dict(self) -> dict:
        return asdict(self)

@dataclass(frozen=True)
class NormalInverseGammaPosterior:
    observations: int
    posterior_mean: float
    kappa: float
    alpha: float
    beta: float
    student_t_df: float
    student_t_scale: float
    credible_interval: tuple[float, float]
    probability_mean_positive: float
    def to_dict(self) -> dict:
        return asdict(self)

def beta_binomial_posterior(successes: int, trials: int, *, prior_alpha: float = 1.0, prior_beta: float = 1.0, credibility: float = 0.90) -> BetaBinomialPosterior:
    successes, trials = int(successes), int(trials)
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes/trials are inconsistent")
    if prior_alpha <= 0 or prior_beta <= 0 or not 0 < credibility < 1:
        raise ValueError("invalid prior/credibility")
    a, b = prior_alpha + successes, prior_beta + trials - successes
    q = (1 - credibility) / 2
    return BetaBinomialPosterior(successes, trials, float(a), float(b), float(a / (a + b)), (float(beta_distribution.ppf(q, a, b)), float(beta_distribution.ppf(1 - q, a, b))), float(1 - beta_distribution.cdf(0.5, a, b)))

def normal_inverse_gamma_posterior(observations: Iterable[float], *, prior_mean: float = 0.0, prior_kappa: float = 1.0, prior_alpha: float = 2.0, prior_beta: float = 1e-4, credibility: float = 0.90) -> NormalInverseGammaPosterior:
    x = np.asarray(list(observations), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("at least one finite observation is required")
    if min(prior_kappa, prior_alpha, prior_beta) <= 0 or not 0 < credibility < 1:
        raise ValueError("invalid prior/credibility")
    n = len(x)
    m = float(x.mean())
    ss = float(np.square(x - m).sum())
    k = prior_kappa + n
    mu = (prior_kappa * prior_mean + n * m) / k
    a = prior_alpha + n / 2
    b = prior_beta + 0.5 * ss + (prior_kappa * n * (m - prior_mean) ** 2) / (2 * k)
    df = 2 * a
    scale = sqrt(b / (a * k))
    q = (1 - credibility) / 2
    ci = (float(student_t.ppf(q, df=df, loc=mu, scale=scale)), float(student_t.ppf(1 - q, df=df, loc=mu, scale=scale)))
    return NormalInverseGammaPosterior(n, float(mu), float(k), float(a), float(b), float(df), float(scale), ci, float(1 - student_t.cdf(0, df=df, loc=mu, scale=scale)))

def bayesian_edge_evidence(net_returns: Iterable[float], *, positive_threshold: float = 0.0, credibility: float = 0.90) -> dict:
    x = np.asarray(list(net_returns), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"status": "INSUFFICIENT_EVIDENCE", "authority": "RESEARCH_ONLY", "live_decision_influence": False}
    mean = normal_inverse_gamma_posterior(x - positive_threshold, credibility=credibility)
    hit = beta_binomial_posterior(int((x > positive_threshold).sum()), len(x), credibility=credibility)
    return {"status": "READY", "observations": len(x), "mean_net_return": float(x.mean()), "median_net_return": float(np.median(x)), "mean_posterior": mean.to_dict(), "positive_hit_rate_posterior": hit.to_dict(), "authority": "RESEARCH_ONLY", "live_decision_influence": False, "automatic_promotion": False}
