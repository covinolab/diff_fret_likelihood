"""``prior=None`` -> true MLE, the isolated ``prior_penalty`` term, config cleanup.

Simulator-independent: tiny synthetic potentials / photon streams built here
(same style as ``test_gp_prior.py``).
"""

import pytest
import torch

import diff_fret_likelihood as dfl
from diff_fret_likelihood.forward import marginal_loglik_batch
from diff_fret_likelihood.objective import neg_log_posterior, prior_penalty, gauge_penalty
from diff_fret_likelihood.sample import build_log_prob


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _grid(n=16, lo=4.0, hi=8.0):
    return dfl.GridConfig(lo, hi, n).build()


def _spline(grid, theta):
    pot = dfl.build_potential(
        dfl.PotentialConfig(n_knots=len(theta)), grid
    )
    with torch.no_grad():
        pot.theta.copy_(torch.as_tensor(theta, dtype=torch.float64))
    return pot


def _tiny_batch():
    torch.manual_seed(3)
    n = 8
    gaps = torch.rand(n) * 0.01
    gaps[0] = 0.0
    ipt = gaps[None, :]
    colors = torch.randint(0, 2, (n,))[None, :]
    mask = torch.ones(1, n, dtype=torch.bool)
    return ipt, colors, mask


class _Batch:
    """Minimal batch object exposing the attrs ``infer.fit`` reads."""

    def __init__(self, ipt, colors, mask):
        self.ipt, self.colors, self.mask = ipt, colors, mask

    def to(self, device):
        return self


def _setup():
    grid = _grid(16)
    pot = _spline(grid, [0.2, -0.8, 0.9, -0.5, 0.7])
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300, .85, .85, 25, 50)
    D = torch.tensor(10.0)
    ipt, colors, mask = _tiny_batch()
    return grid, pot, consts, C, rates, D, ipt, colors, mask


# --------------------------------------------------------------------------- #
# 1. prior=None gives exactly the negative log-likelihood (true MLE)
# --------------------------------------------------------------------------- #
def test_neg_log_posterior_none_is_pure_loglik():
    grid, pot, consts, C, rates, D, ipt, colors, mask = _setup()

    ll = marginal_loglik_batch(ipt, colors, mask, pot, D, rates, grid, C, consts.R0)
    nlp_none = neg_log_posterior(
        ipt, colors, mask, pot, D, rates, grid, C, consts.R0, None
    )
    assert torch.allclose(nlp_none, -ll, atol=1e-12, rtol=0.0)


def test_none_matches_all_off_config():
    """prior=None == PriorConfig.none() numerically (the latter just does 0*curv)."""
    grid, pot, consts, C, rates, D, ipt, colors, mask = _setup()

    nlp_none = neg_log_posterior(
        ipt, colors, mask, pot, D, rates, grid, C, consts.R0, None
    )
    nlp_off = neg_log_posterior(
        ipt, colors, mask, pot, D, rates, grid, C, consts.R0, dfl.PriorConfig.none()
    )
    assert torch.allclose(nlp_none, nlp_off, atol=1e-12, rtol=0.0)


# --------------------------------------------------------------------------- #
# 2. prior_penalty is the single isolated term; 0 for None / all-off
# --------------------------------------------------------------------------- #
def test_prior_penalty_zero_for_none_and_all_off():
    grid, pot, consts, C, rates, D, ipt, colors, mask = _setup()

    z_none = prior_penalty(pot, D, grid, None)
    assert float(z_none) == 0.0 and z_none.dtype == grid.dtype

    z_off = prior_penalty(pot, D, grid, dfl.PriorConfig.none())
    assert float(z_off) == pytest.approx(0.0, abs=1e-12)


def test_neg_log_posterior_is_ll_plus_prior_penalty():
    """The objective decomposes exactly as -ll + prior_penalty for a real prior."""
    grid, pot, consts, C, rates, D, ipt, colors, mask = _setup()
    prior = dfl.PriorConfig(curvature_weight=0.05, gp_sigma=2.0)

    ll = marginal_loglik_batch(ipt, colors, mask, pot, D, rates, grid, C, consts.R0)
    reg = prior_penalty(pot, D, grid, prior)
    nlp = neg_log_posterior(
        ipt, colors, mask, pot, D, rates, grid, C, consts.R0, prior
    )
    assert torch.allclose(nlp, -ll + reg, atol=1e-12, rtol=0.0)
    assert float(reg) > 0.0


# --------------------------------------------------------------------------- #
# 3. fit(prior=None) runs a pure-MLE point estimate and does not increase loss
# --------------------------------------------------------------------------- #
def test_fit_prior_none_smoke():
    grid, pot, consts, C, rates, D, ipt, colors, mask = _setup()
    batch = _Batch(ipt, colors, mask)

    # fit now minimises neg_log_posterior + gauge_penalty; compare like-for-like.
    init_loss = float(
        neg_log_posterior(ipt, colors, mask, pot, D, rates, grid, C, consts.R0, None)
        + gauge_penalty(pot, grid)
    )
    # fit returns the best-loss iterate (loss is evaluated at the init params on the
    # first step), so the reported MLE loss can never exceed the initial loss.
    optim = dfl.OptimConfig(steps=5, log_every=1)
    res = dfl.fit(
        batch, grid, pot, C, consts.R0, D_init=10.0, rates_init=rates,
        prior=None, optim=optim, fit_D=True, verbose=False,
    )
    assert isinstance(res, dfl.FitResult)
    assert torch.isfinite(torch.tensor(res.best_loss))
    assert res.best_loss <= init_loss + 1e-6


# --------------------------------------------------------------------------- #
# 4. config introspection + validation
# --------------------------------------------------------------------------- #
def test_none_and_active_terms():
    assert dfl.PriorConfig.none().active_terms() == []
    assert dfl.PriorConfig.none().describe() == "MLE (no prior)"

    default = dfl.PriorConfig()  # prior-free by default (curvature_weight=0.0)
    assert default.active_terms() == []

    full = dfl.PriorConfig(curvature_weight=0.05, logD_mean=0.0,
                           gp_sigma=2.0, l2_weight=1e-3)
    assert full.active_terms() == ["curvature", "logD", "gp", "l2"]
    assert "MAP prior:" in full.describe()


def test_weight_validation():
    with pytest.raises(ValueError):
        dfl.PriorConfig(curvature_weight=-1.0)
    with pytest.raises(ValueError):
        dfl.PriorConfig(logD_std=0.0)
    with pytest.raises(ValueError):
        dfl.PriorConfig(l2_weight=-1e-3)


# --------------------------------------------------------------------------- #
# 5. the sampler refuses prior=None (improper posterior; HMC won't mix)
# --------------------------------------------------------------------------- #
def test_sampler_rejects_prior_none():
    grid, pot, consts, C, rates, D, ipt, colors, mask = _setup()
    batch = _Batch(ipt, colors, mask)
    with pytest.raises(ValueError):
        build_log_prob(batch, grid, pot, C, consts.R0, None, rates, D_init=10.0)
    # and the existing guard: a real config still needs a proper (gp) prior
    with pytest.raises(ValueError):
        build_log_prob(batch, grid, pot, C, consts.R0,
                       dfl.PriorConfig(curvature_weight=0.05), rates, D_init=10.0)
