"""HMC posterior sampler: log-prob grads, gauge anchor, round-trips, short chain.

Kept tiny (few traces, n_knots=5, a handful of HMC steps) so it runs in seconds.
"""

import math

import pytest
import torch

import diff_fret_likelihood as dfl
from diff_fret_likelihood import sample as S


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
    return dict(batch=batch, grid=grid, pot=pot, C=C, R0=consts.R0,
                prior=prior, rates=rates)


# --------------------------------------------------------------------------- #
# 11. log_prob is finite and its gradient flows to the flat vector
# --------------------------------------------------------------------------- #
def test_log_prob_grad_flows():
    s = _setup()
    logp, z0, info = S.build_log_prob(
        s["batch"], s["grid"], s["pot"], s["C"], s["R0"], s["prior"], s["rates"],
        D_init=10.0,
    )
    z = z0.clone().requires_grad_(True)
    lp = logp(z)
    assert lp.shape == () and torch.isfinite(lp)
    (g,) = torch.autograd.grad(lp, z)
    assert g.shape == z.shape
    assert torch.isfinite(g).all()
    assert float(g.abs().sum()) > 0.0
    assert info["dim"] == z0.numel() == info["npot"] + 1 + S.N_RATES


# --------------------------------------------------------------------------- #
# 12. evaluating log_prob does NOT mutate the potential's parameters
# --------------------------------------------------------------------------- #
def test_log_prob_leaves_module_intact():
    s = s0 = _setup()
    theta_before = s["pot"].theta.detach().clone()
    logp, z0, _ = S.build_log_prob(
        s["batch"], s["grid"], s["pot"], s["C"], s["R0"], s["prior"], s["rates"],
        D_init=10.0,
    )
    _ = logp(z0.clone().requires_grad_(True))
    # functional_call restores params; theta must be an unchanged nn.Parameter
    assert isinstance(s["pot"].theta, torch.nn.Parameter)
    assert torch.allclose(s["pot"].theta.detach(), theta_before)


# --------------------------------------------------------------------------- #
# 13. flatten / unflatten round-trip
# --------------------------------------------------------------------------- #
def test_flatten_unflatten_roundtrip():
    s = _setup()
    z0 = S._flatten_init(s["pot"], 10.0, s["rates"])
    specs = S._param_specs(s["pot"])
    npot = sum(n for _, _, n in specs)
    pdict = S._unflatten(z0[:npot], specs)
    assert torch.allclose(pdict["theta"], s["pot"].theta.detach())
    assert torch.allclose(z0[npot].exp(), torch.tensor(10.0, dtype=torch.float64))
    a_g = z0[npot + 1].exp()
    assert torch.allclose(a_g, s["rates"].a_g)


# --------------------------------------------------------------------------- #
# 14. requires a proper GP prior
# --------------------------------------------------------------------------- #
def test_requires_proper_prior():
    s = _setup()
    bad = dfl.PriorConfig(curvature_weight=0.05)  # gp_sigma None
    with pytest.raises(ValueError):
        S.build_log_prob(s["batch"], s["grid"], s["pot"], s["C"], s["R0"], bad,
                         s["rates"], D_init=10.0)


# --------------------------------------------------------------------------- #
# 15. gauge anchor pins mean(theta); likelihood is shift-invariant
# --------------------------------------------------------------------------- #
def test_gauge_anchor_and_shift_invariance():
    s = _setup()
    gauge_sd = 0.1
    logp, z0, info = S.build_log_prob(
        s["batch"], s["grid"], s["pot"], s["C"], s["R0"], s["prior"], s["rates"],
        D_init=10.0, gauge_sd=gauge_sd,
    )
    npot = info["npot"]
    c = 0.5
    z1 = z0.clone()
    z1[:npot] += c  # add a constant to U (pure gauge)

    lp0 = float(logp(z0))
    lp1 = float(logp(z1))
    m0 = float(z0[:npot].mean())
    # everything except the gauge anchor is invariant to a constant shift of U,
    # so the whole change must equal minus the change in the anchor term
    expected = -0.5 * (((m0 + c) ** 2 - m0 ** 2) / gauge_sd ** 2)
    assert math.isclose(lp1 - lp0, expected, rel_tol=1e-6, abs_tol=1e-6)


# --------------------------------------------------------------------------- #
# 16. a short HMC chain runs and returns well-formed PosteriorSamples
# --------------------------------------------------------------------------- #
def test_short_chain_smoke():
    pytest.importorskip("pyro")
    s = _setup()
    ps = S.sample_posterior(
        s["batch"], s["grid"], s["pot"], s["C"], s["R0"], s["prior"], s["rates"],
        D_init=10.0, num_samples=4, warmup=2, num_steps_per_sample=2,
        step_size=0.002, sampler="hmc", map_warmstart=False, verbose=False,
    )
    G = s["grid"].shape[0]
    Sn = ps.U.shape[0]
    assert Sn >= 1
    assert ps.U.shape == (Sn, G)
    assert ps.D.shape == (Sn,)
    assert ps.rates.shape == (Sn, S.N_RATES)
    assert torch.isfinite(ps.U).all() and torch.isfinite(ps.D).all()
    assert (ps.D > 0).all() and (ps.rates > 0).all()
    # gauge-fixed landscapes have min 0
    assert torch.allclose(ps.U.min(dim=1).values, torch.zeros(Sn, dtype=torch.float64),
                          atol=1e-9)
    # band helpers
    band = ps.U_band((0.1, 0.9))
    assert band.shape == (2, G)
    assert (band[1] >= band[0] - 1e-12).all()
    assert ps.U_mean().shape == (G,)


def test_to_arviz():
    pytest.importorskip("arviz")
    pytest.importorskip("pyro")
    s = _setup()
    ps = S.sample_posterior(
        s["batch"], s["grid"], s["pot"], s["C"], s["R0"], s["prior"], s["rates"],
        D_init=10.0, num_samples=4, warmup=2, num_steps_per_sample=2,
        step_size=0.002, sampler="hmc", map_warmstart=False, verbose=False,
    )
    idata = ps.to_arviz()
    assert "logD" in idata.posterior
    assert "theta_0" in idata.posterior


# --------------------------------------------------------------------------- #
# 18. a short chain runs from a MAP warm start (pyro adapts step size + mass matrix)
# --------------------------------------------------------------------------- #
def test_chain_runs_with_map_warmstart():
    pytest.importorskip("pyro")
    s = _setup()
    ps = S.sample_posterior(
        s["batch"], s["grid"], s["pot"], s["C"], s["R0"], s["prior"], s["rates"],
        D_init=10.0, num_samples=4, warmup=3, num_steps_per_sample=3,
        step_size=0.005, sampler="hmc", map_warmstart=True, verbose=False,
    )
    assert torch.isfinite(ps.U).all() and torch.isfinite(ps.D).all()
    assert (ps.D > 0).all() and (ps.rates > 0).all()


# --------------------------------------------------------------------------- #
# 19. multi-chain assembles an arviz InferenceData with a real chain dim (R-hat)
# --------------------------------------------------------------------------- #
def test_multi_chain_rhat():
    pytest.importorskip("pyro")
    pytest.importorskip("arviz")
    s = _setup()
    mc = S.sample_posterior_multi(
        s["batch"], s["grid"], s["pot"], s["C"], s["R0"], s["prior"], s["rates"],
        num_chains=2, overdisperse=0.2, base_seed=0,
        D_init=10.0, num_samples=6, warmup=3, num_steps_per_sample=2,
        step_size=0.01, sampler="hmc", map_warmstart=False, verbose=False,
    )
    assert len(mc.chains) == 2
    assert mc.idata is not None
    assert mc.idata.posterior.sizes["chain"] == 2
    summ = mc.summary()
    assert "r_hat" in summ.columns
    assert mc.U.shape[0] == mc.chains[0].U.shape[0] + mc.chains[1].U.shape[0]
