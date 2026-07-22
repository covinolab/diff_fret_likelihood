"""Regression: forward-backward occupancy is self-consistent (audit fix)."""

import math

import torch

import diff_fret_likelihood as dfl
from diff_fret_likelihood.forward import build_propagator_from_u, _BasePotential_on_grid
from diff_fret_likelihood.generator import stationary


class _OneTraceBatch:
    def __init__(self, ipt, colors, n):
        self.ipt = ipt[None, :]
        self.colors = colors[None, :]
        self.mask = torch.ones(1, n, dtype=torch.bool)
        self.lengths = torch.tensor([n])
        self.n_traces = 1

    def to(self, device):
        return self


def test_forward_backward_consistency():
    """Sum_i a~_k * b~_k must be constant over k and equal the marginal likelihood."""
    torch.manual_seed(1)
    grid = dfl.GridConfig(4, 8, 20).build()
    pot = dfl.build_potential(dfl.PotentialConfig(kind="spline", n_knots=6), grid)
    with torch.no_grad():
        pot.theta.copy_(torch.tensor([0.2, -0.8, 0.9, -0.5, 0.7, 0.1]))
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300, .85, .85, 25, 50)
    D = torch.tensor(10.0)
    dx = float(grid[1] - grid[0])
    u = _BasePotential_on_grid(pot, grid)
    prop = build_propagator_from_u(u, D, rates, grid, C, consts.R0, dx)
    s = prop.s
    n = 9
    gaps = torch.rand(n) * 0.01
    gaps[0] = 0.0
    cols = torch.randint(0, 2, (n,))

    a, v = [], stationary(u) / s
    for k in range(n):
        v = prop.propagate(v, gaps[k])
        v = v * (prop.mu_G if cols[k] == 0 else prop.mu_R)
        a.append(v.clone())
    b = [None] * n
    b[n - 1] = s.clone()
    for k in range(n - 1, 0, -1):
        em = prop.mu_G if cols[k] == 0 else prop.mu_R
        b[k - 1] = prop.propagate(b[k] * em, gaps[k])
    smoothed = torch.stack([(a[k] * b[k]).sum() for k in range(n)])
    assert (smoothed.max() / smoothed.min() - 1).abs() < 1e-9

    times = torch.cumsum(gaps, 0)
    ll = dfl.marginal_loglik(times, cols, float(times[-1]), pot, D, rates, grid, C, consts.R0)
    assert abs(math.log(float(smoothed[0])) - float(ll)) < 1e-8


def test_fit_enforces_mean_theta_zero_and_preserves_identified():
    """The fit's gauge anchor pins mean(theta)=0 with ZERO bias on identified quantities.

    Fit the same data twice from identical init: once with the anchor on (gauge_sd=1)
    and once with it effectively off (gauge_sd large). A proper prior determines the
    non-offset directions, so the *only* thing the anchor may change is the pure-gauge
    offset. Assert the anchor pins mean(theta)=0 while D, the grid-mean-zero shape, and
    the barrier height are unchanged.
    """
    torch.manual_seed(0)
    grid = dfl.GridConfig(4, 8, 24).build()
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300, .85, .85, 25, 50)
    n = 60
    gaps = torch.rand(n) * 0.01
    gaps[0] = 0.0
    cols = torch.randint(0, 2, (n,))
    batch = _OneTraceBatch(gaps, cols, n)

    # proper prior on the shape + a logD prior so D stays identified (avoids the
    # D-valley blow-up on tiny data); none of these terms constrain the offset.
    prior = dfl.PriorConfig(curvature_weight=0.1, gp_sigma=2.0,
                            logD_mean=math.log(10.0), logD_std=0.5)
    optim = dfl.OptimConfig(adam_steps=1500, adam_lr=0.05, log_every=200)
    init_theta = torch.tensor([0.5, 0.0, -0.9, 0.8, -0.3, 0.4])

    def fresh_pot():
        pot = dfl.build_potential(dfl.PotentialConfig(kind="spline", n_knots=6), grid)
        with torch.no_grad():
            pot.theta.copy_(init_theta)
        return pot

    # blur="none" -> plain deterministic Adam (no homotopy noise), so the two fits
    # differ only by the gauge anchor, as the test intends.
    res_anchor = dfl.fit(batch, grid, fresh_pot(), C, consts.R0, D_init=10.0,
                         rates_init=rates, prior=prior, optim=optim, fit_D=True,
                         verbose=False, gauge_sd=1.0, blur="none")
    res_base = dfl.fit(batch, grid, fresh_pot(), C, consts.R0, D_init=10.0,
                       rates_init=rates, prior=prior, optim=optim, fit_D=True,
                       verbose=False, gauge_sd=1e6, blur="none")

    # (1) the anchor pins the enforcement gauge mean(theta)=0; the baseline does not
    #     (nothing else constrains the offset, so the baseline stays near its init mean).
    assert abs(float(res_anchor.potential.theta.mean())) < 5e-3
    assert abs(float(res_base.potential.theta.mean())) > 3e-2

    # (2) reporting gauge is grid-mean-zero by definition.
    U_a = dfl.recovered_potential(res_anchor.potential, grid)
    U_b = dfl.recovered_potential(res_base.potential, grid)
    assert abs(float(U_a.mean())) < 1e-8

    # (3) ZERO BIAS: D, the (grid-mean-zero) shape, and the barrier height are unchanged
    #     -- only the unobservable offset differed between the two fits.
    assert abs(res_anchor.D - res_base.D) < 1e-2 * res_base.D
    assert float(torch.sqrt(((U_a - U_b) ** 2).mean())) < 1e-2
    assert abs(float(U_a.max() - U_a.min()) - float(U_b.max() - U_b.min())) < 1e-2
