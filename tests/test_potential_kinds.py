"""The landscape contract (``Potential``) and its three implementations.

``SplinePotential`` adopts the interface without changing a single number: every hook
delegates to the code the consumers ran before, and the tests here pin that with
``torch.equal`` (bits, not tolerances).  ``FixedPotential`` wraps a callable and emits no
parameters, so ``fit`` optimises only ``D`` and the photophysics.  ``ParametricPotential``
wraps a closed form with named, all-free parameters.

Kept tiny (3 traces x 8 photons, 12 grid points, 5 knots) so the fits run in seconds.
"""

import copy
import math

import pytest
import torch

import diff_fret_likelihood as dfl
from diff_fret_likelihood import init
from diff_fret_likelihood import sample as S
from diff_fret_likelihood.objective import (
    bg_penalty, curvature_penalty_spline, gauge_penalty, neg_log_posterior, prior_penalty,
)
from diff_fret_likelihood.potential import SplinePotential

CURVED_THETA = (1.3, -0.8, 0.9, -1.1, 0.4)
ZERO = torch.zeros((), dtype=torch.float64)


def _batch(n_traces, n_ph, seed=0, gap=0.01):
    g = torch.Generator().manual_seed(seed)
    ipt = torch.rand(n_traces, n_ph, generator=g, dtype=torch.float64) * gap
    ipt[:, 0] = 0.0
    colors = torch.randint(0, 2, (n_traces, n_ph), generator=g)
    mask = torch.ones(n_traces, n_ph, dtype=torch.bool)
    return dfl.simulate.Batch(ipt, colors, mask,
                              torch.full((n_traces,), n_ph), ipt.sum(1))


def _setup(n_knots=5, n_grid=12, n_traces=3, n_ph=8):
    torch.manual_seed(0)
    grid = dfl.GridConfig(4.0, 8.0, n_grid).build()
    pot = dfl.build_potential(dfl.PotentialConfig(n_knots=n_knots), grid)
    with torch.no_grad():
        pot.theta.copy_(torch.tensor(CURVED_THETA[:n_knots], dtype=torch.float64))
    consts = dfl.PhysicsConstants()
    return dict(batch=_batch(n_traces, n_ph), grid=grid, pot=pot,
                C=consts.crosstalk_tensor(), R0=consts.R0, D=torch.tensor(10.0),
                rates=dfl.EffectiveRates.from_physics(300, .85, .85, 25, 50))


def _harmonic(x, x0, k):
    return 0.5 * k * (x - x0) ** 2


def _fit(s, pot, **kw):
    b = s["batch"]
    kw.setdefault("optim", dfl.OptimConfig(steps=3, lbfgs_lr=0.1))
    return dfl.fit(b, s["grid"], pot, s["C"], s["R0"], D_init=10.0, rates_init=s["rates"],
                   prior=None, verbose=False, **kw)


# --------------------------------------------------------------------------- #
# the contract itself
# --------------------------------------------------------------------------- #
def test_base_class_defaults():
    grid = dfl.GridConfig(4.0, 8.0, 12).build()

    class OnlyGrid(dfl.Potential):
        def on_grid(self, grid):
            return torch.zeros_like(grid)

    p = OnlyGrid()
    assert p.gauge_free is False
    assert list(p.parameters()) == []
    assert torch.equal(p.gauge_offset(grid), ZERO)
    with pytest.raises(NotImplementedError):
        p(grid)                                   # forward is optional
    with pytest.raises(ValueError, match="curvature_weight=0"):
        p.curvature_penalty("l2")
    with pytest.raises(NotImplementedError):
        p.project(grid, torch.zeros_like(grid))
    with pytest.raises(NotImplementedError):
        dfl.Potential().on_grid(grid)             # on_grid is the one required method


# --------------------------------------------------------------------------- #
# SplinePotential: conforms, and every number is the same as before
# --------------------------------------------------------------------------- #
def test_spline_ctor_without_grid():
    grid = dfl.GridConfig(4.0, 8.0, 12).build()
    cfg = dfl.PotentialConfig(x_center=6.0, x_scale=2.0, n_knots=5)
    pot = SplinePotential(cfg)
    ref = dfl.build_potential(dfl.PotentialConfig(n_knots=5), grid)
    assert torch.equal(pot.knots_x, ref.knots_x)
    assert torch.equal(pot.theta, ref.theta)
    assert torch.equal(pot.on_grid(grid), ref.on_grid(grid))
    with pytest.raises(TypeError):
        SplinePotential(cfg, grid)                # the grid is not a construction parameter
    with pytest.raises(ValueError):
        SplinePotential(dfl.PotentialConfig(n_knots=5))   # ... but the knot window is


def test_spline_conforms_and_is_bit_identical():
    s = _setup()
    pot, grid, D = s["pot"], s["grid"], s["D"]
    assert isinstance(pot, dfl.Potential)
    assert pot.gauge_free is True

    # hooks == the legacy expressions, bit for bit
    assert torch.equal(pot.gauge_offset(grid), pot.theta.mean())
    for norm in ("l2", "l1"):
        assert torch.equal(pot.curvature_penalty(norm),
                           curvature_penalty_spline(pot.theta, pot.knots_x, norm))

    # consumer level: values AND theta-gradients
    sd = 0.7
    gp = gauge_penalty(pot, grid, sd)
    gp_legacy = 0.5 * (pot.theta.mean() / sd) ** 2
    assert torch.equal(gp, gp_legacy)
    (g,) = torch.autograd.grad(gp, pot.theta)
    (g_legacy,) = torch.autograd.grad(gp_legacy, pot.theta)
    assert torch.equal(g, g_legacy)
    for norm in ("l2", "l1"):
        prior = dfl.PriorConfig(curvature_weight=0.05, curvature_norm=norm)
        pp = prior_penalty(pot, D, grid, prior)
        pp_legacy = ZERO + 0.05 * curvature_penalty_spline(pot.theta, pot.knots_x, norm)
        assert torch.equal(pp, pp_legacy)
        (g,) = torch.autograd.grad(pp, pot.theta)
        (g_legacy,) = torch.autograd.grad(pp_legacy, pot.theta)
        assert torch.equal(g, g_legacy)

    # project == the least-squares warm start
    target = pot._basis(grid) @ torch.tensor([0.0, 1.0, -0.5, 0.8, -0.3], dtype=torch.float64)
    sol = torch.linalg.lstsq(pot._basis(grid), target.unsqueeze(1)).solution.reshape(-1)
    out = pot.project(grid, target)
    assert out is pot
    assert torch.allclose(pot.theta.detach(), sol, atol=1e-12, rtol=0)


def test_warmstart_dispatches_to_project():
    grid = dfl.GridConfig(4.0, 8.0, 50).build()
    pot = dfl.build_potential(dfl.PotentialConfig(n_knots=6), grid)
    target = pot._basis(grid) @ torch.tensor([0.0, 1.0, -0.5, 0.8, -0.3, 0.4], dtype=torch.float64)
    assert init.warmstart_potential(pot, grid, target) is pot
    assert torch.allclose(pot.on_grid(grid), target, atol=1e-8)
    for other in (dfl.FixedPotential(pot),
                  dfl.ParametricPotential(_harmonic, dict(x0=6.0, k=2.0))):
        with pytest.raises(NotImplementedError):       # NOT ValueError: sample._warm_start swallows that
            init.warmstart_potential(other, grid, target)


# --------------------------------------------------------------------------- #
# FixedPotential: a callable, no parameters
# --------------------------------------------------------------------------- #
def test_fixed_potential_wraps_a_callable():
    s = _setup()
    pot, grid = s["pot"], s["grid"]

    fixed = dfl.FixedPotential(pot)                    # a fitted spline is a callable
    assert isinstance(fixed, dfl.Potential)
    assert fixed.gauge_free is False
    assert list(fixed.parameters()) == []
    u = fixed.on_grid(grid)
    assert torch.equal(u, pot.on_grid(grid).detach())
    assert not u.requires_grad
    assert fixed.on_grid(grid) is u                    # cached per grid object, like the spline
    grid2 = dfl.GridConfig(4.0, 8.0, 7).build()
    assert torch.equal(fixed.on_grid(grid2), pot.on_grid(grid2).detach())

    f = dfl.FixedPotential(lambda x: 0.5 * (x - 6.0) ** 2)
    x = torch.linspace(4.5, 7.5, 5, dtype=torch.float64)
    assert torch.equal(f(x), 0.5 * (x - 6.0) ** 2)
    assert torch.allclose(f.force(x), -(x - 6.0))
    assert torch.equal(f.on_grid(grid), 0.5 * (grid - 6.0) ** 2)
    assert torch.equal(f.gauge_offset(grid), ZERO)
    assert torch.equal(f.curvature_penalty("l2"), ZERO)


def test_fixed_potential_fit_moves_only_D_and_rates():
    s = _setup()
    pot, grid, b = s["pot"], s["grid"], s["batch"]
    theta_before = pot.theta.detach().clone()
    fixed = dfl.FixedPotential(pot)

    res = _fit(s, fixed, fit_D=True, fit_rates=True)
    assert res.potential is fixed
    assert torch.equal(pot.theta.detach(), theta_before)     # the source spline is untouched
    assert math.isfinite(res.D) and res.D > 0 and res.D != 10.0
    assert res.log_D_param is not None
    assert len(list(res.free_rates.parameters())) == 4

    # same landscape => same objective as the spline at the same D / rates
    nlp_fixed = neg_log_posterior(b.ipt, b.colors, b.mask, fixed, s["D"], s["rates"],
                                  grid, s["C"], s["R0"], None)
    nlp_spline = neg_log_posterior(b.ipt, b.colors, b.mask, pot, s["D"], s["rates"],
                                   grid, s["C"], s["R0"], None)
    assert torch.allclose(nlp_fixed, nlp_spline, atol=1e-12, rtol=0)
    assert torch.equal(gauge_penalty(fixed, grid, 1.0), ZERO)
    assert torch.equal(dfl.recovered_potential(fixed, grid), fixed.on_grid(grid))   # unshifted


def test_fit_without_free_parameters_raises():
    s = _setup()
    fixed = dfl.FixedPotential(s["pot"])
    with pytest.raises(ValueError, match="nothing to optimise"):
        _fit(s, fixed, fit_D=False, fit_rates=False)
    with pytest.raises(ValueError, match="nothing to optimise"):
        dfl.fit_multi([s["batch"]], s["grid"], fixed, [s["C"]], [s["R0"]], D_init=10.0,
                      rates_init_list=[s["rates"]], prior=None, fit_D=False, fit_rates=False,
                      verbose=False)


# --------------------------------------------------------------------------- #
# ParametricPotential: a closed form with named, free parameters
# --------------------------------------------------------------------------- #
def test_parametric_potential_physical_and_fit():
    s = _setup()
    grid = s["grid"]
    par = dfl.ParametricPotential(_harmonic, dict(x0=6.0, k=2.0), positive=("k",))
    assert isinstance(par, dfl.Potential)
    assert [n for n, _ in par.named_parameters()] == ["x0", "log_k"]   # dict order
    phys = par.physical()
    assert set(phys) == {"x0", "k"}
    assert torch.allclose(phys["k"], torch.tensor(2.0, dtype=torch.float64))
    assert torch.allclose(par.on_grid(grid), _harmonic(grid, 6.0, 2.0))
    x = torch.linspace(4.5, 7.5, 5, dtype=torch.float64)
    assert torch.allclose(par(x), _harmonic(x, 6.0, 2.0))
    assert torch.allclose(par.force(x), -2.0 * (x - 6.0))
    assert par.gauge_free is False
    assert torch.equal(par.gauge_offset(grid), ZERO)

    before = torch.stack([p.detach().clone() for p in par.parameters()])
    res = _fit(s, par, fit_D=False, fit_rates=False)
    assert res.potential is par
    after = torch.stack([p.detach() for p in par.parameters()])
    assert not torch.equal(before, after)
    assert float(par.physical()["k"]) > 0


def test_parametric_gauge_param_is_anchored():
    s = _setup()
    grid = s["grid"]

    def fn(x, x0, c):
        return 0.5 * (x - x0) ** 2 + c

    par = dfl.ParametricPotential(fn, dict(x0=6.0, c=1.5), gauge_param="c")
    assert par.gauge_free is True
    sd = 0.7
    assert torch.equal(par.gauge_offset(grid), par.physical()["c"])
    assert torch.equal(gauge_penalty(par, grid, sd), 0.5 * (par.physical()["c"] / sd) ** 2)
    u = par.on_grid(grid).detach()
    assert torch.allclose(dfl.recovered_potential(par, grid), u - u.mean())


def test_curvature_prior_needs_knots():
    s = _setup()
    grid, D, rates = s["grid"], s["D"], s["rates"]
    curv = dfl.PriorConfig(curvature_weight=0.05)
    par = dfl.ParametricPotential(_harmonic, dict(x0=6.0, k=2.0), positive=("k",))
    with pytest.raises(ValueError, match="curvature_weight=0"):
        prior_penalty(par, D, grid, curv)
    assert torch.equal(prior_penalty(par, D, grid, None), ZERO)
    assert torch.equal(prior_penalty(par, D, grid, dfl.PriorConfig(curvature_weight=0.0)), ZERO)
    # no parameters => the curvature term is a constant => exactly zero
    assert torch.equal(prior_penalty(dfl.FixedPotential(s["pot"]), D, grid, curv), ZERO)
    # the background prior is about the rates and works for any potential
    bg = dfl.PriorConfig(bg_g_mean=18.0, bg_g_sd=2.0, bg_r_mean=61.0, bg_r_sd=5.0)
    assert torch.equal(prior_penalty(par, D, grid, bg, rates=rates), bg_penalty(rates, bg))


# --------------------------------------------------------------------------- #
# CRB and the sampler stay spline-only and say so
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["fixed", "parametric"])
def test_crb_and_sampler_refuse_non_spline(kind):
    s = _setup()
    b, grid, C, R0, rates = s["batch"], s["grid"], s["C"], s["R0"], s["rates"]
    pot = (dfl.FixedPotential(s["pot"]) if kind == "fixed"
           else dfl.ParametricPotential(_harmonic, dict(x0=6.0, k=2.0), positive=("k",)))
    with pytest.raises(NotImplementedError):
        dfl.cramer_rao_bound(b, grid, pot, s["D"], rates, C, R0)
    with pytest.raises(NotImplementedError):
        S.build_log_prob(b, grid, pot, C, R0, None, rates, D_init=10.0)
    with pytest.raises(NotImplementedError):
        S.sample_posterior(b, grid, pot, C, R0, None, rates, kde_warmstart=False,
                           map_warmstart=False, num_samples=2, warmup=2, verbose=False)
    with pytest.raises(NotImplementedError):
        S.sample_posterior_multi(b, grid, pot, C, R0, None, rates, num_chains=2,
                                 kde_warmstart=False, map_warmstart=False,
                                 num_samples=2, warmup=2, verbose=False)
