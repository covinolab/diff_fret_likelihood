"""Laplace-proposal importance sampling: shapes/gauge, exact-Gaussian recovery,
the no-Jacobian invariant, the proper-prior guard, and a CRB cross-check.

Kept tiny (few traces, n_knots=5) so it runs in seconds; the core IS math is tested
against a synthetic Gaussian target that needs no repo model (and no pyro).
"""

import math

import pytest
import torch

import diff_fret_likelihood as dfl
from diff_fret_likelihood import laplace_is as L
from diff_fret_likelihood.laplace_is import laplace_importance_samples
from diff_fret_likelihood.sample import PosteriorSamples, build_log_prob

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# fixture: a tiny problem + a FitResult refined to an approximate MAP.
#
# The unit tests use random (non-model) photons, so the *posterior* is weakly
# identified and non-Gaussian -> importance-sampling ESS is legitimately low and
# Pareto-k high here (the diagnostics correctly flag it). High-ESS behaviour is
# only expected on real model-simulated data at a real MAP, which is exercised in
# the manual end-to-end verification (needs the Cython simulator), NOT here. What
# these tests pin is the plumbing (shapes/gauge/positivity), the exact IS math
# (against a synthetic Gaussian target), and the API guards.
# --------------------------------------------------------------------------- #
def _setup(n_knots=5, n_grid=12, n_traces=3, n_ph=8):
    torch.manual_seed(0)
    grid = dfl.GridConfig(4.0, 8.0, n_grid).build()
    pot = dfl.build_potential(dfl.PotentialConfig(kind="spline", n_knots=n_knots), grid)
    with torch.no_grad():
        pot.theta.copy_(torch.linspace(0.3, -0.3, n_knots))
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300, .85, .85, 25, 50)

    ipt = torch.zeros(n_traces, n_ph, dtype=torch.float64)
    colors = torch.zeros(n_traces, n_ph, dtype=torch.int64)
    mask = torch.ones(n_traces, n_ph, dtype=torch.bool)
    for b in range(n_traces):
        gaps = torch.rand(n_ph) * 0.01
        gaps[0] = 0.0
        ipt[b] = gaps
        colors[b] = torch.randint(0, 2, (n_ph,))
    lengths = torch.full((n_traces,), n_ph, dtype=torch.int64)
    T = ipt.sum(1)
    batch = dfl.simulate.Batch(ipt, colors, mask, lengths, T)
    prior = dfl.PriorConfig(curvature_weight=0.0, gp_sigma=2.0, gp_lengthscale=1.0)
    result = dfl.FitResult(potential=pot, D=10.0, rates=rates, best_loss=0.0)
    return dict(result=result, batch=batch, grid=grid, C=C, R0=consts.R0, prior=prior)


def _approx_map(s, steps=250, lr=0.05):
    """Refine the hand-set point to an approximate posterior mode (a few Adam steps
    on the real log_prob_func) so the Laplace precondition -- a genuine MAP with a
    positive-definite Hessian -- holds. Returns an updated FitResult."""
    r = s["result"]
    logp, z0, info = build_log_prob(
        s["batch"], s["grid"], r.potential, s["C"], s["R0"], s["prior"], r.rates,
        D_init=float(r.D),
    )
    z = z0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        (-logp(z)).backward()
        opt.step()
    z = z.detach()
    npot = info["npot"]
    with torch.no_grad():
        r.potential.theta.copy_(z[:npot])
    lr_ = z[npot + 1:npot + 5].exp()
    rates = dfl.EffectiveRates(lr_[0], lr_[1], lr_[2], lr_[3])
    return dfl.FitResult(potential=r.potential, D=float(z[npot].exp()),
                         rates=rates, best_loss=float(-logp(z)))


# --------------------------------------------------------------------------- #
# 1. end-to-end: well-formed, equally-weighted PosteriorSamples in the right gauge
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore::RuntimeWarning")   # low ESS on toy data is expected
def test_shapes_gauge_and_positivity():
    s = _setup()
    res = _approx_map(s)
    n = 128
    ps = laplace_importance_samples(
        res, s["batch"], s["grid"], s["C"], s["R0"], s["prior"],
        n_samples=n, oversample=2, seed=0, verbose=False,
    )
    G = s["grid"].shape[0]
    npot = res.potential.theta.numel()
    dim = npot + 1 + 4
    assert isinstance(ps, PosteriorSamples)
    assert ps.U.shape == (n, G)
    assert ps.D.shape == (n,)
    assert ps.rates.shape == (n, 4)
    assert ps.z.shape == (n, dim)
    assert ps.theta.shape == (n, npot)
    assert torch.isfinite(ps.U).all() and torch.isfinite(ps.D).all()
    assert (ps.D > 0).all() and (ps.rates > 0).all()
    # grid-mean-zero reporting gauge (same as sample.sample_posterior)
    assert torch.allclose(ps.U.mean(dim=1), torch.zeros(n, dtype=torch.float64), atol=1e-9)
    # band helpers work and are monotone
    band = ps.U_band((0.1, 0.9))
    assert band.shape == (2, G)
    assert (band[1] >= band[0] - 1e-12).all()
    assert ps.U_mean().shape == (G,)
    # diagnostics attached
    assert 0.0 < ps.ess_frac <= 1.0
    assert math.isfinite(ps.log_evidence)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_gaussian_proposal_option_runs():
    s = _setup()
    res = _approx_map(s)
    ps = laplace_importance_samples(
        res, s["batch"], s["grid"], s["C"], s["R0"], s["prior"],
        n_samples=64, oversample=2, proposal="gaussian", cov_scale=1.0,
        seed=1, verbose=False,
    )
    assert (ps.D > 0).all() and torch.isfinite(ps.U).all()


# --------------------------------------------------------------------------- #
# 2. exact-Gaussian target: proposal == target => constant weights, moments recovered
#    (validates the proposal / weight / self-normalisation math)
# --------------------------------------------------------------------------- #
def _gaussian_logprob(mu, Sigma):
    P = torch.linalg.inv(Sigma)

    def log_prob_func(z):
        d = z - mu
        return -0.5 * (d @ P @ d)
    return log_prob_func


def test_gaussian_target_recovers_moments():
    dim = 4
    torch.manual_seed(3)
    mu = torch.tensor([0.5, -1.0, 2.0, 0.0], dtype=torch.float64)
    A = torch.randn(dim, dim, dtype=torch.float64)
    Sigma = A @ A.T + dim * torch.eye(dim, dtype=torch.float64)   # SPD
    lp = _gaussian_logprob(mu, Sigma)

    out = L._importance_sample(
        lp, mu.clone(), n_samples=6000, oversample=1, dist="gaussian",
        df=5.0, cov_scale=1.0, eig_rtol=1e-12, seed=0, verbose=False,
    )
    z, W = out["z"], out["weights"]
    # proposal exactly equals target -> weights constant -> ESS ~ full
    assert out["ess_frac"] > 0.98
    mean = (W[:, None] * z).sum(0)
    dz = z - mean
    cov = (W[:, None, None] * (dz[:, :, None] * dz[:, None, :])).sum(0)
    assert torch.allclose(mean, mu, atol=0.15)
    assert torch.allclose(cov, Sigma, rtol=0.2, atol=0.2)


# --------------------------------------------------------------------------- #
# 3. NO Jacobian: weighted mean of exp(z) matches the analytic log-normal mean.
#    If an exp-map Jacobian were wrongly folded into the weights, this would bias.
# --------------------------------------------------------------------------- #
def test_no_jacobian_lognormal_mean():
    mu, sig = 1.3, 0.4
    mu_t = torch.tensor([mu], dtype=torch.float64)

    def lp(z):
        return -0.5 * ((z[0] - mu) / sig) ** 2

    out = L._importance_sample(
        lp, mu_t.clone(), n_samples=8000, oversample=1, dist="gaussian",
        df=5.0, cov_scale=1.0, eig_rtol=1e-12, seed=0, verbose=False,
    )
    z, W = out["z"], out["weights"]
    est = float((W * z[:, 0].exp()).sum())            # E[exp(X)], X ~ N(mu, sig^2)
    analytic = math.exp(mu + 0.5 * sig ** 2)
    assert abs(est - analytic) / analytic < 0.05


# --------------------------------------------------------------------------- #
# 4. requires a proper GP prior (guard fires before any Hessian work)
# --------------------------------------------------------------------------- #
def test_requires_gp_prior():
    s = _setup()
    bad = dfl.PriorConfig(curvature_weight=0.05)      # gp_sigma is None
    with pytest.raises(ValueError):
        laplace_importance_samples(
            s["result"], s["batch"], s["grid"], s["C"], s["R0"], bad,
            n_samples=16, oversample=1, verbose=False,
        )
    with pytest.raises(ValueError):
        laplace_importance_samples(
            s["result"], s["batch"], s["grid"], s["C"], s["R0"], None,
            n_samples=16, oversample=1, verbose=False,
        )


def test_student_t_df_guard():
    s = _setup()
    with pytest.raises(ValueError):
        laplace_importance_samples(
            s["result"], s["batch"], s["grid"], s["C"], s["R0"], s["prior"],
            n_samples=16, oversample=1, proposal="student_t", df=1.5, verbose=False,
        )


# --------------------------------------------------------------------------- #
# 5. cross-check the Laplace covariance against the Cramer-Rao bound on the
#    gauge-independent components (logD + 4 log-rates). Loose: the Laplace cov
#    includes the GP/rate priors + gauge anchor and is at the (hand-set) point,
#    whereas CRB is the pure-Fisher pseudo-inverse -- they agree only in order.
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_crb_crosscheck_order_of_magnitude():
    s = _setup()
    res = _approx_map(s)
    ps = laplace_importance_samples(
        res, s["batch"], s["grid"], s["C"], s["R0"], s["prior"],
        n_samples=64, oversample=2, seed=0, verbose=False,
    )
    crb = dfl.cramer_rao_bound(
        s["batch"], s["grid"], res.potential, float(res.D), res.rates,
        s["C"], s["R0"],
    )
    npot = res.potential.theta.numel()
    lap_sigma = torch.sqrt(torch.diag(ps.cov))        # z-space marginal std
    crb_sigma = crb.sigma
    # logD + the 4 log-rates: unaffected by the U additive gauge
    for i in range(npot, npot + 5):
        a = float(lap_sigma[i])
        b = float(crb_sigma[i])
        assert math.isfinite(a) and a > 0
        assert math.isfinite(b) and b > 0
        ratio = a / b
        assert 1e-2 < ratio < 1e2, f"index {i}: laplace/crb sigma ratio {ratio:.3g}"
