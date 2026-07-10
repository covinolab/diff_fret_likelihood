"""Proper GP prior over U(x): kernel, penalty math, gauge, backward-compat, grads.

All simulator-independent: synthetic potentials / tiny photon streams built here.
"""

import math

import pytest
import torch

import diff_fret_likelihood as dfl
from diff_fret_likelihood.objective import (
    _gp_corr, _interp, gp_penalty, neg_log_posterior,
)

KERNELS = ["rbf", "matern32", "matern52"]


def _grid(n=20, lo=4.0, hi=8.0):
    return dfl.GridConfig(lo, hi, n).build()


def _spline(grid, theta):
    pot = dfl.build_potential(
        dfl.PotentialConfig(kind="spline", n_knots=len(theta)), grid
    )
    with torch.no_grad():
        pot.theta.copy_(torch.as_tensor(theta, dtype=torch.float64))
    return pot


def _x_ctrl(grid, n_ctrl):
    n = int(min(max(n_ctrl, 4), grid.shape[0]))
    return torch.linspace(float(grid.min()), float(grid.max()), n, dtype=torch.float64)


# --------------------------------------------------------------------------- #
# 1. kernel is PSD
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kernel", KERNELS)
def test_kernel_psd(kernel):
    x = torch.linspace(3.0, 10.0, 16, dtype=torch.float64)
    Kc = _gp_corr(x, lengthscale=1.0, kernel=kernel)
    assert torch.allclose(Kc, Kc.T, atol=1e-12)
    assert torch.diag(Kc).allclose(torch.ones(16, dtype=torch.float64))  # unit variance
    eig = torch.linalg.eigvalsh(Kc + 1e-6 * torch.eye(16, dtype=torch.float64))
    assert float(eig.min()) > 0.0


# --------------------------------------------------------------------------- #
# 2. penalty matches an explicit 0.5 r^T K^-1 r
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kernel", KERNELS)
def test_penalty_matches_explicit_inverse(kernel):
    grid = _grid(20)
    pot = _spline(grid, [0.3, -0.9, 0.7, -0.4, 0.6, 0.1])
    prior = dfl.PriorConfig(curvature_weight=0.0, gp_sigma=2.5,
                            gp_lengthscale=1.2, gp_kernel=kernel, gp_n_ctrl=8)
    got = gp_penalty(pot, grid, prior)

    x = _x_ctrl(grid, 8)
    u = pot(x)
    r = u - u.mean()
    Kc = _gp_corr(x, prior.gp_lengthscale, kernel)
    K = prior.gp_sigma ** 2 * (Kc + prior.gp_jitter * torch.eye(8, dtype=torch.float64))
    want = 0.5 * (r @ torch.linalg.solve(K, r))
    assert torch.allclose(got, want, atol=1e-10, rtol=1e-10), (float(got), float(want))


# --------------------------------------------------------------------------- #
# 3. zero at the prior mean
# --------------------------------------------------------------------------- #
def test_zero_on_flat_potential():
    grid = _grid(20)
    pot = _spline(grid, [0.7, 0.7, 0.7, 0.7, 0.7])   # flat -> r == 0
    prior = dfl.PriorConfig(curvature_weight=0.0, gp_sigma=2.0)
    assert float(gp_penalty(pot, grid, prior)) == pytest.approx(0.0, abs=1e-12)


def test_zero_when_mean_equals_potential():
    """gp_mean == U on the grid, with x_ctrl == grid nodes -> exact zero."""
    grid = _grid(16)
    pot = _spline(grid, [0.2, -0.8, 0.9, -0.5, 0.7])
    u_grid = pot(grid).detach()
    prior = dfl.PriorConfig(curvature_weight=0.0, gp_sigma=2.0,
                            gp_n_ctrl=grid.shape[0], gp_mean=u_grid)
    assert float(gp_penalty(pot, grid, prior)) == pytest.approx(0.0, abs=1e-10)


# --------------------------------------------------------------------------- #
# 4. quadratic in the residual, inverse-square in sigma
# --------------------------------------------------------------------------- #
def test_quadratic_and_sigma_scaling():
    grid = _grid(20)
    theta = [0.3, -0.9, 0.7, -0.4, 0.6, 0.1]
    p1 = gp_penalty(_spline(grid, theta),
                    grid, dfl.PriorConfig(curvature_weight=0.0, gp_sigma=2.0))
    # residual x2 -> penalty x4 (gp_mean=None, so scaling theta scales r)
    p2 = gp_penalty(_spline(grid, [2 * t for t in theta]),
                    grid, dfl.PriorConfig(curvature_weight=0.0, gp_sigma=2.0))
    assert torch.allclose(p2, 4.0 * p1, rtol=1e-10)
    # sigma x2 -> penalty /4
    ps = gp_penalty(_spline(grid, theta),
                    grid, dfl.PriorConfig(curvature_weight=0.0, gp_sigma=4.0))
    assert torch.allclose(ps, p1 / 4.0, rtol=1e-10)


# --------------------------------------------------------------------------- #
# 5. smoothness ordering (n_knots == n_ctrl so u_ctrl == theta exactly)
# --------------------------------------------------------------------------- #
def test_smoothness_ordering():
    grid = _grid(24)
    K = 12
    idx = torch.arange(K, dtype=torch.float64)
    smooth = torch.sin(2 * math.pi * idx / K)                 # 1 period
    wiggly = torch.sin(2 * math.pi * idx / 2.0)               # Nyquist-ish
    smooth = smooth - smooth.mean(); wiggly = wiggly - wiggly.mean()
    smooth = smooth / smooth.norm(); wiggly = wiggly / wiggly.norm()  # same amplitude

    def pen(theta, ls):
        prior = dfl.PriorConfig(curvature_weight=0.0, gp_sigma=1.0,
                                gp_lengthscale=ls, gp_kernel="matern52", gp_n_ctrl=K)
        return float(gp_penalty(_spline(grid, theta.tolist()), grid, prior))

    assert pen(wiggly, 1.0) > pen(smooth, 1.0)               # rough costs more
    # a LONGER correlation length imposes more smoothness, so the same wiggle
    # costs more (as ls->0 the kernel -> I and any residual costs only 0.5||r||^2)
    assert pen(wiggly, 1.0) > pen(wiggly, 0.5)


# --------------------------------------------------------------------------- #
# 6. gauge invariance (constant shift of U leaves the penalty unchanged)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kernel", KERNELS)
def test_gauge_invariance(kernel):
    grid = _grid(20)
    theta = [0.3, -0.9, 0.7, -0.4, 0.6, 0.1]
    prior = dfl.PriorConfig(curvature_weight=0.0, gp_sigma=2.0, gp_kernel=kernel)
    p0 = gp_penalty(_spline(grid, theta), grid, prior)
    p1 = gp_penalty(_spline(grid, [t + 3.14 for t in theta]), grid, prior)
    assert torch.allclose(p0, p1, atol=1e-9)


# --------------------------------------------------------------------------- #
# 7. linear-interp helper is exact on affine data
# --------------------------------------------------------------------------- #
def test_interp_exact_on_affine():
    grid = _grid(20)
    y = 1.7 * grid - 2.3
    q = torch.linspace(4.1, 7.9, 9, dtype=torch.float64)
    assert torch.allclose(_interp(grid, y, q), 1.7 * q - 2.3, atol=1e-12)


# --------------------------------------------------------------------------- #
# 8. backward-compat: GP off is inert; GP on is a clean additive term
# --------------------------------------------------------------------------- #
def _tiny_batch():
    torch.manual_seed(3)
    n = 8
    gaps = torch.rand(n) * 0.01
    gaps[0] = 0.0
    ipt = gaps[None, :]
    colors = torch.randint(0, 2, (n,))[None, :]
    mask = torch.ones(1, n, dtype=torch.bool)
    return ipt, colors, mask


def test_backward_compat_gp_off_and_additive():
    grid = _grid(16)
    pot = _spline(grid, [0.2, -0.8, 0.9, -0.5, 0.7])
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300, .85, .85, 25, 50)
    D = torch.tensor(10.0)
    ipt, colors, mask = _tiny_batch()

    prior_off = dfl.PriorConfig(curvature_weight=0.05)          # gp_sigma None
    prior_on = dfl.PriorConfig(curvature_weight=0.05, gp_sigma=2.0)

    # GP term is exactly zero (and correct dtype/device) when off
    z = gp_penalty(pot, grid, prior_off)
    assert float(z) == 0.0 and z.dtype == grid.dtype

    nlp_off = neg_log_posterior(ipt, colors, mask, pot, D, rates, grid, C, consts.R0, prior_off)
    nlp_on = neg_log_posterior(ipt, colors, mask, pot, D, rates, grid, C, consts.R0, prior_on)
    gp = gp_penalty(pot, grid, prior_on)
    # the ONLY difference between the two objectives is the additive GP term
    assert torch.allclose(nlp_on - nlp_off, gp, atol=1e-10)
    assert float(gp) > 0.0


# --------------------------------------------------------------------------- #
# 9. gradients flow (spline theta via gradcheck; MLP net params finite)
# --------------------------------------------------------------------------- #
def test_gradcheck_wrt_control_values():
    grid = _grid(16)
    prior = dfl.PriorConfig(curvature_weight=0.0, gp_sigma=2.0,
                            gp_kernel="matern52", gp_n_ctrl=6)
    k = 4
    theta0 = torch.tensor([0.4, -0.7, 0.3, 0.9], dtype=torch.float64, requires_grad=True)

    def f(theta):
        pot = lambda x: torch.stack([x ** j for j in range(k)], -1) @ theta  # noqa: E731
        return gp_penalty(pot, grid, prior)

    assert torch.autograd.gradcheck(f, (theta0,), atol=1e-6)


def test_grad_flows_to_mlp_params():
    grid = _grid(16)
    pot = dfl.build_potential(dfl.PotentialConfig(kind="mlp", hidden=(8, 8)), grid)
    prior = dfl.PriorConfig(curvature_weight=0.0, gp_sigma=2.0)
    gp_penalty(pot, grid, prior).backward()
    grads = [p.grad for p in pot.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)


# --------------------------------------------------------------------------- #
# 10. n_ctrl clamp + determinism
# --------------------------------------------------------------------------- #
def test_n_ctrl_clamp_and_determinism():
    grid = _grid(16)
    pot = _spline(grid, [0.2, -0.8, 0.9, -0.5, 0.7])
    # too many control points -> clamped to n_grid; too few -> clamped to 4
    hi = gp_penalty(pot, grid, dfl.PriorConfig(curvature_weight=0.0, gp_sigma=2.0, gp_n_ctrl=1000))
    lo = gp_penalty(pot, grid, dfl.PriorConfig(curvature_weight=0.0, gp_sigma=2.0, gp_n_ctrl=1))
    assert torch.isfinite(hi) and torch.isfinite(lo)
    # determinism
    p = dfl.PriorConfig(curvature_weight=0.0, gp_sigma=2.0)
    assert float(gp_penalty(pot, grid, p)) == float(gp_penalty(pot, grid, p))


def test_config_validation():
    with pytest.raises(ValueError):
        dfl.PriorConfig(gp_sigma=-1.0)
    with pytest.raises(ValueError):
        dfl.PriorConfig(gp_sigma=2.0, gp_lengthscale=0.0)
    with pytest.raises(ValueError):
        dfl.PriorConfig(gp_sigma=2.0, gp_kernel="cauchy")
