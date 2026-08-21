"""HMC posterior sampler: the target identity, warm starts, and short chains.

The load-bearing test here is ``test_target_is_fit_objective``: the sampler's whole
contract is that its target density is ``infer.fit``'s objective negated, so that a chain
and a fit run on the same ``prior``/``gauge_sd`` describe the same posterior. Everything
else is about the warm starts and the plumbing around it.

Kept tiny (few traces, n_knots=5, a handful of HMC steps) so it runs in seconds.
"""

import copy
import math
import warnings

import pytest
import torch

import diff_fret_likelihood as dfl
from diff_fret_likelihood import sample as S
from diff_fret_likelihood.objective import neg_log_posterior, gauge_penalty

# A background prior placed deliberately OFF the rates it will be evaluated at: the Gamma
# is exactly 0 at bg == mean, so equal values would silently zero the term and make the
# identity test pass vacuously.
BG_PRIOR = dict(bg_g_mean=18.0, bg_g_sd=2.0, bg_r_mean=61.0, bg_r_sd=5.0)

# Likewise the knot heights must be CURVED: a linear theta has an exactly zero second
# difference, so the curvature prior would contribute nothing.
CURVED_THETA = (1.3, -0.8, 0.9, -1.1, 0.4)


def _batch(n_traces, n_ph, seed=0, gap=0.01):
    g = torch.Generator().manual_seed(seed)
    ipt = torch.rand(n_traces, n_ph, generator=g, dtype=torch.float64) * gap
    ipt[:, 0] = 0.0
    colors = torch.randint(0, 2, (n_traces, n_ph), generator=g)
    mask = torch.ones(n_traces, n_ph, dtype=torch.bool)
    return dfl.simulate.Batch(ipt, colors, mask,
                              torch.full((n_traces,), n_ph), ipt.sum(1))


def _setup(n_knots=5, n_grid=12, n_traces=3, n_ph=8, theta=CURVED_THETA):
    torch.manual_seed(0)
    grid = dfl.GridConfig(4.0, 8.0, n_grid).build()
    pot = dfl.build_potential(dfl.PotentialConfig(n_knots=n_knots), grid)
    with torch.no_grad():
        pot.theta.copy_(torch.tensor(theta[:n_knots], dtype=torch.float64))
    consts = dfl.PhysicsConstants()
    return dict(batch=_batch(n_traces, n_ph), grid=grid, pot=pot,
                C=consts.crosstalk_tensor(), R0=consts.R0, consts=consts,
                prior=dfl.PriorConfig(curvature_weight=0.05),
                rates=dfl.EffectiveRates.from_physics(300, .85, .85, 25, 50))


def _build(s, prior="default", **kw):
    """``build_log_prob`` with warnings silenced (they are asserted separately)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return S.build_log_prob(s["batch"], s["grid"], s["pot"], s["C"], s["R0"],
                                s["prior"] if prior == "default" else prior,
                                s["rates"], **kw)


# --------------------------------------------------------------------------- #
# the target IS infer.fit's objective
# --------------------------------------------------------------------------- #
def _fit_objective(s, prior, z, npot, n_rates, gauge_sd):
    """``infer.fit``'s ``closure_value()``, evaluated at the params encoded in ``z``."""
    pot = copy.deepcopy(s["pot"])
    with torch.no_grad():
        pot.theta.copy_(z[:npot])
    D = z[npot].exp()
    r = z[npot + 1:npot + 1 + n_rates].exp()
    rates = dfl.EffectiveRates(r[0], r[1], r[2], r[3])
    b = s["batch"]
    return float(neg_log_posterior(b.ipt, b.colors, b.mask, pot, D, rates,
                                   s["grid"], s["C"], s["R0"], prior)
                 + gauge_penalty(pot, s["grid"], gauge_sd))


@pytest.mark.parametrize("name,prior", [
    ("none",          None),
    ("bare",          dfl.PriorConfig()),
    ("curvature_l2",  dfl.PriorConfig(curvature_weight=0.05)),
    ("curvature_l1",  dfl.PriorConfig(curvature_weight=0.05, curvature_norm="l1")),
    ("bg",            dfl.PriorConfig(**BG_PRIOR)),
    ("curvature_bg",  dfl.PriorConfig(curvature_weight=0.05, **BG_PRIOR)),
])
def test_target_is_fit_objective(name, prior):
    """log_prob(z) == -(neg_log_posterior + gauge_penalty) -- the sampler's contract.

    Checked away from z0 as well: an identity that only held at the initial point would
    say nothing about the density the chain actually explores.
    """
    s = _setup()
    gsd = 0.7
    logp, z0, info = _build(s, prior, D_init=2.0, gauge_sd=gsd)
    npot, n_rates = info["npot"], info["n_sampled_rates"]

    # guard against a vacuous check: every prior except the two prior-free ones must
    # actually move the objective, or this test would pass on a sampler that ignored it.
    base = _fit_objective(s, None, z0, npot, n_rates, gsd)
    if name not in ("none", "bare"):
        assert abs(_fit_objective(s, prior, z0, npot, n_rates, gsd) - base) > 1e-9

    for k in range(4):
        g = torch.Generator().manual_seed(k)
        z = z0 + (0.0 if k == 0 else
                  0.25 * torch.randn(z0.numel(), generator=g, dtype=torch.float64))
        got = float(logp(z.clone().requires_grad_(True)))
        want = -_fit_objective(s, prior, z, npot, n_rates, gsd)
        assert math.isclose(got, want, rel_tol=1e-12), f"{name} @ draw {k}"


def test_background_prior_counted_once():
    """A bg prior must shift the target by exactly one bg_penalty, not two."""
    from diff_fret_likelihood.objective import bg_penalty
    s = _setup()
    off = dfl.PriorConfig(curvature_weight=0.05)
    on = dfl.PriorConfig(curvature_weight=0.05, **BG_PRIOR)
    (lp_off, z0, _), (lp_on, _, _) = (_build(s, off, D_init=2.0),
                                      _build(s, on, D_init=2.0))
    expect = float(bg_penalty(s["rates"], on))
    assert expect > 1e-6                       # the term is actually active
    assert math.isclose(float(lp_off(z0)) - float(lp_on(z0)), expect, rel_tol=1e-10)


# --------------------------------------------------------------------------- #
# prior handling: no gate, one warning
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prior", [
    None,
    dfl.PriorConfig(),
    dfl.PriorConfig(curvature_weight=0.05),
    dfl.PriorConfig(**BG_PRIOR),
])
def test_accepts_any_prior(prior):
    """Nothing is rejected -- prior=None is pure MLE here exactly as it is in fit."""
    s = _setup()
    logp, z0, _ = _build(s, prior, D_init=10.0)
    assert torch.isfinite(logp(z0.clone().requires_grad_(True)))


def test_warns_only_when_there_is_no_prior():
    s = _setup()
    for prior in (None, dfl.PriorConfig()):
        with pytest.warns(RuntimeWarning, match="no prior at all"):
            S.build_log_prob(s["batch"], s["grid"], s["pot"], s["C"], s["R0"], prior,
                             s["rates"], D_init=10.0)
    for prior in (dfl.PriorConfig(curvature_weight=0.05), dfl.PriorConfig(**BG_PRIOR)):
        with warnings.catch_warnings():
            warnings.simplefilter("error")      # any warning fails the test
            S.build_log_prob(s["batch"], s["grid"], s["pot"], s["C"], s["R0"], prior,
                             s["rates"], D_init=10.0)


def test_l1_curvature_warns():
    s = _setup()
    prior = dfl.PriorConfig(curvature_weight=0.05, curvature_norm="l1")
    with pytest.warns(RuntimeWarning, match="not differentiable"):
        S.build_log_prob(s["batch"], s["grid"], s["pot"], s["C"], s["R0"], prior,
                         s["rates"], D_init=10.0)


@pytest.mark.parametrize("dead", ["rate_sd", "logD_sd"])
def test_removed_kwargs_are_gone(dead):
    """Both were sampler-invented priors with no counterpart in fit; removed, not shimmed."""
    s = _setup()
    with pytest.raises(TypeError):
        _build(s, D_init=10.0, **{dead: 1.0})


# --------------------------------------------------------------------------- #
# flat vector plumbing
# --------------------------------------------------------------------------- #
def test_log_prob_grad_flows():
    s = _setup()
    logp, z0, info = _build(s, D_init=10.0)
    z = z0.clone().requires_grad_(True)
    lp = logp(z)
    assert lp.shape == () and torch.isfinite(lp)
    (g,) = torch.autograd.grad(lp, z)
    assert g.shape == z.shape and torch.isfinite(g).all()
    assert float(g.abs().sum()) > 0.0
    assert info["dim"] == z0.numel() == info["npot"] + 1 + S.N_RATES


def test_log_prob_leaves_module_intact():
    s = _setup()
    theta_before = s["pot"].theta.detach().clone()
    logp, z0, _ = _build(s, D_init=10.0)
    _ = logp(z0.clone().requires_grad_(True))
    assert isinstance(s["pot"].theta, torch.nn.Parameter)
    assert torch.allclose(s["pot"].theta.detach(), theta_before)


def test_flatten_unflatten_roundtrip():
    s = _setup()
    z0 = S._flatten_init(s["pot"], 10.0, s["rates"])
    specs = S._param_specs(s["pot"])
    npot = sum(n for _, _, n in specs)
    assert torch.allclose(S._unflatten(z0[:npot], specs)["theta"], s["pot"].theta.detach())
    assert torch.allclose(z0[npot].exp(), torch.tensor(10.0, dtype=torch.float64))
    assert torch.allclose(z0[npot + 1].exp(), s["rates"].a_g)


def test_sample_bg_false_drops_the_backgrounds_from_z():
    s = _setup()
    logp, z0, info = _build(s, D_init=10.0, sample_bg=False)
    assert info["n_sampled_rates"] == 2
    assert info["rate_names"] == ("a_g", "a_r")
    assert z0.numel() == info["npot"] + 1 + 2
    # the frozen pair still reaches the likelihood: the target stays finite and moves
    assert torch.isfinite(logp(z0.clone().requires_grad_(True)))


def test_gauge_anchor_and_shift_invariance():
    """Everything but the anchor is invariant to a constant shift of U."""
    s = _setup()
    gauge_sd = 0.1
    logp, z0, info = _build(s, D_init=10.0, gauge_sd=gauge_sd)
    npot, c = info["npot"], 0.5
    z1 = z0.clone()
    z1[:npot] += c
    m0 = float(z0[:npot].mean())
    expected = -0.5 * (((m0 + c) ** 2 - m0 ** 2) / gauge_sd ** 2)
    assert math.isclose(float(logp(z1)) - float(logp(z0)), expected,
                        rel_tol=1e-6, abs_tol=1e-6)


# --------------------------------------------------------------------------- #
# warm starts
# --------------------------------------------------------------------------- #
def test_kde_warm_start_runs_by_default():
    """The default path sets theta from the FRET histogram and profiles out a D."""
    s = _setup(n_traces=6, n_ph=400)
    theta_before = s["pot"].theta.detach().clone()
    rates, D_kde = S._warm_start(
        s["batch"], s["grid"], s["pot"], s["C"], s["R0"], s["prior"], None,
        physics=None, kde_warmstart=True, kde_bin_ms=None,
        kde_kwargs=dict(bin_ms_grid=(1.0, 2.0), D_grid=[1.0, 4.0]),
        compile_mode=None, propagate_dtype=None, verbose=False)
    assert not torch.allclose(s["pot"].theta.detach(), theta_before)
    assert D_kde is not None and math.isfinite(D_kde) and D_kde > 0
    assert float(rates.a_g) > 0                      # built by stream_rates


def test_kde_seeds_backgrounds_from_the_prior_calibration():
    s = _setup(n_traces=6, n_ph=400)
    prior = dfl.PriorConfig(curvature_weight=0.05, **BG_PRIOR)
    rates, _ = S._warm_start(
        s["batch"], s["grid"], s["pot"], s["C"], s["R0"], prior, None,
        physics=None, kde_warmstart=False, kde_bin_ms=None, kde_kwargs=None,
        compile_mode=None, propagate_dtype=None, verbose=False)
    assert float(rates.bg_g) == pytest.approx(BG_PRIOR["bg_g_mean"])
    assert float(rates.bg_r) == pytest.approx(BG_PRIOR["bg_r_mean"])


def test_kde_warm_start_falls_back_when_the_stream_is_too_short():
    """Too few photons to bin is a reason to skip the init, not to fail the run."""
    s = _setup()
    sparse = _batch(3, 8, gap=1.0)                   # ~0.5 ms between photons
    theta_before = s["pot"].theta.detach().clone()
    with pytest.warns(RuntimeWarning, match="KDE warm start skipped"):
        rates, D_kde = S._warm_start(
            sparse, s["grid"], s["pot"], s["C"], s["R0"], s["prior"], s["rates"],
            physics=None, kde_warmstart=True, kde_bin_ms=0.05, kde_kwargs=None,
            compile_mode=None, propagate_dtype=None, verbose=False)
    assert D_kde is None
    assert torch.allclose(s["pot"].theta.detach(), theta_before)


def test_physics_is_rebuilt_from_the_crosstalk_matrix():
    consts = dfl.PhysicsConstants(R0=5.5, C_gg=0.88, C_gr=0.12, C_rg=0.07, C_rr=0.93)
    got = S._physics_from(consts.crosstalk_tensor(), consts.R0)
    assert got == consts


def test_gauge_sd_reaches_the_map_warm_start():
    """Tightening the anchor must pull the MAP's mean(theta) toward zero.

    The chain and the warm start have to share one gauge; if ``gauge_sd`` stopped at
    ``sample_posterior`` the chain would start at the MAP of a different objective.
    A very loose anchor (>= 50) is left out on purpose: it lets the offset drift far
    enough that the generator goes ill-conditioned and ``eigh`` stops converging.
    """
    s = _setup(n_traces=6, n_ph=200)
    optim = dfl.OptimConfig(steps=10, log_every=1000)

    def fitted_offset(gauge_sd):
        pot = copy.deepcopy(s["pot"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S.sample_posterior(
                s["batch"], s["grid"], pot, s["C"], s["R0"], s["prior"], s["rates"],
                kde_warmstart=False, map_warmstart=True, map_optim=optim,
                fit_rates=False, gauge_sd=gauge_sd, D_init=5.0, num_samples=1,
                warmup=1, num_steps_per_sample=1, step_size=1e-6, sampler="hmc",
                verbose=False)
        return abs(float(pot.theta.mean()))

    offsets = [fitted_offset(g) for g in (0.02, 0.2, 2.0)]
    assert offsets[0] < offsets[1] < offsets[2], offsets
    assert offsets[0] < 0.1 * abs(float(s["pot"].theta.mean()))


# --------------------------------------------------------------------------- #
# chains
# --------------------------------------------------------------------------- #
def _chain(s, **kw):
    pytest.importorskip("pyro")
    kw.setdefault("D_init", 10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return S.sample_posterior(
            s["batch"], s["grid"], s["pot"], s["C"], s["R0"], s["prior"], s["rates"],
            kde_warmstart=False, num_samples=4, warmup=2, num_steps_per_sample=2,
            step_size=0.002, sampler="hmc", verbose=False, **kw)


def test_short_chain_smoke():
    s = _setup()
    ps = _chain(s, map_warmstart=False)
    G, Sn = s["grid"].shape[0], ps.U.shape[0]
    assert Sn >= 1
    assert ps.U.shape == (Sn, G) and ps.D.shape == (Sn,)
    assert ps.rates.shape == (Sn, S.N_RATES)
    assert torch.isfinite(ps.U).all() and torch.isfinite(ps.D).all()
    assert (ps.D > 0).all() and (ps.rates > 0).all()
    assert torch.allclose(ps.U.mean(dim=1), torch.zeros(Sn, dtype=torch.float64),
                          atol=1e-9)
    band = ps.U_band((0.1, 0.9))
    assert band.shape == (2, G) and (band[1] >= band[0] - 1e-12).all()
    assert ps.U_mean().shape == (G,)


def test_reported_U_matches_the_potential():
    """The vectorised basis matmul must agree with evaluating the potential per draw."""
    s = _setup()
    ps = _chain(s, map_warmstart=False)
    pot = copy.deepcopy(s["pot"])
    for i in range(ps.theta.shape[0]):
        with torch.no_grad():
            pot.theta.copy_(ps.theta[i])
            u = pot.on_grid(s["grid"])
            u = u - u.mean()
        assert torch.allclose(ps.U[i], u, atol=1e-10)


def test_fit_bg_false_freezes_the_backgrounds():
    s = _setup()
    ps = _chain(s, map_warmstart=False, fit_rates=True, fit_bg=False)
    for k, name in ((2, "bg_g"), (3, "bg_r")):
        col = ps.rates[:, k]
        assert torch.allclose(col, col[0].expand_as(col)), name
    assert float(ps.rates[0, 2]) == pytest.approx(float(s["rates"].bg_g))
    assert float(ps.rates[0, 3]) == pytest.approx(float(s["rates"].bg_r))
    assert ps.n_sampled_rates == 2          # only a_g/a_r were in z
    assert ps.z.shape[1] == ps.theta.shape[1] + 1 + 2


def test_fit_rates_does_not_reach_the_chain():
    """fit_rates is a MAP-fit knob; the chain still samples the brightnesses.

    Only fit_bg reaches z. Coupling the two would freeze the backgrounds whenever the
    warm start held the rates, which is a different model than the caller asked for.
    """
    s = _setup()
    ps = _chain(s, map_warmstart=False, fit_rates=False, fit_bg=True)
    assert ps.n_sampled_rates == S.N_RATES
    assert ps.z.shape[1] == ps.theta.shape[1] + 1 + S.N_RATES


def test_chain_runs_with_map_warmstart():
    s = _setup()
    ps = _chain(s, map_warmstart=True,
                map_optim=dfl.OptimConfig(steps=10, log_every=1000))
    assert torch.isfinite(ps.U).all() and torch.isfinite(ps.D).all()
    assert (ps.D > 0).all() and (ps.rates > 0).all()


def test_to_arviz():
    pytest.importorskip("arviz")
    s = _setup()
    idata = _chain(s, map_warmstart=False).to_arviz()
    assert "logD" in idata.posterior and "theta_0" in idata.posterior
    assert "log_a_g" in idata.posterior


def test_arviz_skips_frozen_rates():
    """A constant column would come back with a NaN R-hat; leave it out instead."""
    pytest.importorskip("arviz")
    s = _setup()
    idata = _chain(s, map_warmstart=False, fit_rates=True, fit_bg=False).to_arviz()
    assert "log_a_g" in idata.posterior
    assert "log_bg_g" not in idata.posterior and "log_bg_r" not in idata.posterior


def test_multi_chain_rhat():
    pytest.importorskip("pyro")
    pytest.importorskip("arviz")
    s = _setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mc = S.sample_posterior_multi(
            s["batch"], s["grid"], s["pot"], s["C"], s["R0"], s["prior"], s["rates"],
            num_chains=2, overdisperse=0.2, base_seed=0, kde_warmstart=False,
            map_warmstart=False, D_init=10.0, num_samples=6, warmup=3,
            num_steps_per_sample=2, step_size=0.01, sampler="hmc", verbose=False)
    assert len(mc.chains) == 2
    assert mc.idata is not None and mc.idata.posterior.sizes["chain"] == 2
    assert "r_hat" in mc.summary().columns
    assert mc.U.shape[0] == mc.chains[0].U.shape[0] + mc.chains[1].U.shape[0]


def _multi(s, **kw):
    pytest.importorskip("pyro")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return S.sample_posterior_multi(
            s["batch"], s["grid"], s["pot"], s["C"], s["R0"], s["prior"], s["rates"],
            num_chains=2, overdisperse=0.2, base_seed=0, kde_warmstart=False,
            map_warmstart=False, D_init=10.0, num_samples=5, warmup=2,
            num_steps_per_sample=2, step_size=0.01, sampler="hmc", verbose=False, **kw)


def test_parallel_chains_match_sequential():
    """``n_parallel`` is a scheduling choice, not a modelling one.

    Each chain seeds its own RNG from ``base_seed + c`` and the MAP warm start is
    deterministic, so which process a chain ran in must not change its draws. Compared with
    a tolerance rather than exactly: the point is that the chains are the same, not that
    two interpreters accumulate floats in identical order.
    """
    seq = _multi(_setup(), n_parallel=1)
    par = _multi(_setup(), n_parallel=2)
    assert len(par.chains) == len(seq.chains) == 2
    for c, (a, b) in enumerate(zip(seq.chains, par.chains)):
        assert torch.allclose(a.z, b.z, atol=1e-8), f"chain {c} z"
        assert torch.allclose(a.U, b.U, atol=1e-8), f"chain {c} U"
        assert torch.allclose(a.D, b.D, atol=1e-8), f"chain {c} D"


def test_parallel_returns_the_same_shape_of_result():
    """Same devices, same dtypes, same container -- and n_parallel clamps to num_chains."""
    s = _setup()
    seq, par = _multi(s, n_parallel=1), _multi(s, n_parallel=9)   # 9 -> clamped to 2
    assert len(par.chains) == 2
    for a, b in zip(seq.chains, par.chains):
        for name in ("U", "D", "rates", "theta", "z", "grid"):
            x, y = getattr(a, name), getattr(b, name)
            assert x.shape == y.shape and x.dtype == y.dtype and x.device == y.device, name
        assert a.n_sampled_rates == b.n_sampled_rates
    if par.idata is not None:
        assert par.idata.posterior.sizes["chain"] == 2
