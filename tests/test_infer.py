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


def test_occupancy_normalisation():
    """posterior_occupancy sums to the number of photons (each smoothed marginal = 1)."""
    torch.manual_seed(2)
    grid = dfl.GridConfig(4, 8, 24).build()
    pot = dfl.build_potential(dfl.PotentialConfig(kind="spline", n_knots=5), grid)
    with torch.no_grad():
        pot.theta.copy_(torch.tensor([0.0, -0.7, 0.6, -0.4, 0.2]))
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300, .85, .85, 25, 50)
    n = 12
    gaps = torch.rand(n) * 0.01
    gaps[0] = 0.0
    cols = torch.randint(0, 2, (n,))
    batch = _OneTraceBatch(gaps, cols, n)
    occ = dfl.posterior_occupancy(batch, pot, 10.0, rates, grid, C, consts.R0)
    assert abs(float(occ.sum()) - n) < 1e-6
    assert (occ >= 0).all()
