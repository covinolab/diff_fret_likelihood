"""Gates 4-6: Poisson identity, reference match, batched==single, gradcheck."""

import math

import pytest
import torch

import diff_fret_likelihood as dfl
from diff_fret_likelihood.generator import smoluchowski, stationary
from diff_fret_likelihood.photophysics import emission_rates
from diff_fret_likelihood.forward import (
    reference_loglik, _forward_recursion_single, build_propagator_from_u,
)


def _emission_on_grid(grid, rates, C, R0):
    return emission_rates(grid, rates, C, R0)


def test_G1_poisson_identity():
    """G=1: marginal_loglik == log(mu_G^{#G} mu_R^{#R}) - mu T  (SPEC gate 4)."""
    grid = torch.tensor([6.0])  # single grid point
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300.0, 0.85, 0.85, 25.0, 50.0)
    # flat potential
    pcfg = dfl.PotentialConfig(kind="spline", n_knots=2, x_center=6.0, x_scale=1.0)
    pot = dfl.build_potential(pcfg, grid)

    mu_G, mu_R = _emission_on_grid(grid, rates, C, consts.R0)
    mu = float(mu_G + mu_R)

    times = torch.tensor([0.001, 0.004, 0.010, 0.017])
    colors = torch.tensor([0, 1, 0, 1])
    T = 0.025
    ll = dfl.marginal_loglik(times, colors, T, pot, torch.tensor(5.0),
                             rates, grid, C, consts.R0)
    n_g = int((colors == 0).sum())
    n_r = int((colors == 1).sum())
    expect = n_g * math.log(float(mu_G)) + n_r * math.log(float(mu_R)) - mu * T
    assert abs(float(ll) - expect) < 1e-8, f"{float(ll)} vs {expect}"


def test_matches_reference_multistate():
    """Fast eigendecomposition path == slow matrix_exp reference (SPEC gate 5)."""
    torch.manual_seed(3)
    G = 6
    grid = torch.linspace(4.0, 8.0, G)
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300.0, 0.85, 0.85, 25.0, 50.0)
    pcfg = dfl.PotentialConfig(kind="spline", n_knots=4)
    pot = dfl.build_potential(pcfg, grid)
    with torch.no_grad():
        pot.theta.copy_(torch.tensor([1.0, -0.5, 0.7, -1.2]))
    D = torch.tensor(8.0)
    dx = float(grid[1] - grid[0])

    u = pot.on_grid(grid); u = u - u.min()
    L = smoluchowski(u, D, dx)
    mu_G, mu_R = _emission_on_grid(grid, rates, C, consts.R0)
    p0 = stationary(u)

    gaps = torch.rand(10) * 0.02
    times = torch.cumsum(gaps, 0)
    colors = torch.randint(0, 2, (10,))
    T = float(times[-1]) + 0.005

    ref = reference_loglik(L, mu_G, mu_R, times, colors, T, p0)
    fast = dfl.marginal_loglik(times, colors, T, pot, D, rates, grid, C, consts.R0)
    assert abs(float(ref) - float(fast)) < 1e-8, f"{float(ref)} vs {float(fast)}"


def test_two_state_fast_vs_reference():
    """G=2: fast eigh path == matrix_exp reference (self-consistency).

    NOTE: this is an internal fast-vs-reference cross-check at G=2 (both are the
    *same* model), not an independent Gopich-Szabo oracle -- see
    ``test_two_state_gopich_szabo_handcoded`` for the external oracle (SPEC 8.1).
    """
    torch.manual_seed(1)
    G = 2
    grid = torch.linspace(4.5, 7.0, G)
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(250.0, 0.9, 0.9, 20.0, 30.0)
    pcfg = dfl.PotentialConfig(kind="spline", n_knots=2)
    pot = dfl.build_potential(pcfg, grid)
    with torch.no_grad():
        pot.theta.copy_(torch.tensor([0.0, 1.3]))  # 2 wells at different depths
    D = torch.tensor(4.0)
    dx = float(grid[1] - grid[0])
    u = pot.on_grid(grid); u = u - u.min()
    L = smoluchowski(u, D, dx)
    mu_G, mu_R = _emission_on_grid(grid, rates, C, consts.R0)
    p0 = stationary(u)

    gaps = torch.rand(12) * 0.03
    times = torch.cumsum(gaps, 0)
    colors = torch.randint(0, 2, (12,))
    T = float(times[-1])

    ref = reference_loglik(L, mu_G, mu_R, times, colors, T, p0)
    fast = dfl.marginal_loglik(times, colors, T, pot, D, rates, grid, C, consts.R0)
    assert abs(float(ref) - float(fast)) < 1e-8


# ---------------------------------------------------------------------------
# Independent two-state Gopich-Szabo oracle (SPEC section 8.1)
# ---------------------------------------------------------------------------
def _exp2x2(M: torch.Tensor, tau: float) -> torch.Tensor:
    """Closed-form ``exp(M*tau)`` for a 2x2 matrix via Cayley-Hamilton.

    Independent of ``torch.linalg.eigh`` and ``matrix_exp`` -- used purely as an
    external oracle.  ``exp(M t) = alpha I + beta M`` with alpha/beta from the two
    (real) eigenvalues of M; a degenerate branch guards equal eigenvalues.
    """
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    tr = a + d
    det = a * d - b * c
    disc = torch.sqrt((tr * tr - 4.0 * det).clamp_min(0.0))
    eye = torch.eye(2, dtype=M.dtype)
    if float(disc) < 1e-14:                      # equal eigenvalues
        lam = 0.5 * tr
        return torch.exp(lam * tau) * (eye + (M - lam * eye) * tau)
    lam_p = 0.5 * (tr + disc)
    lam_m = 0.5 * (tr - disc)
    ep, em = torch.exp(lam_p * tau), torch.exp(lam_m * tau)
    alpha = (lam_p * em - lam_m * ep) / (lam_p - lam_m)
    beta = (ep - em) / (lam_p - lam_m)
    return alpha * eye + beta * M


def _two_state_gs_loglik(k01, k10, mu_G, mu_R, times, colors, T, p0):
    """Hand-coded 2-state Gopich-Szabo photon-by-photon log-likelihood.

    2x2 rate matrix ``K`` (``K[j,i]`` = rate i->j), killed generator
    ``M = K - diag(mu_G+mu_R)``, closed-form 2x2 propagator, scaled forward
    recursion.  No grid, no eigh, no ``smoluchowski`` -- a fully independent
    oracle for the continuous evaluator restricted to two wells.
    """
    dtype = p0.dtype
    K = torch.tensor([[-k01, k10], [k01, -k10]], dtype=dtype)   # cols sum to 0
    M = K - torch.diag(mu_G + mu_R)
    v = p0.clone()
    c0 = v.sum(); v = v / c0; log_norm = torch.log(c0)
    t_prev = 0.0
    for k in range(times.shape[0]):
        v = _exp2x2(M, float(times[k]) - t_prev) @ v
        emit = mu_G if int(colors[k]) == 0 else mu_R
        v = v * emit
        c = v.sum(); v = v / c; log_norm = log_norm + torch.log(c)
        t_prev = float(times[k])
    v = _exp2x2(M, float(T) - t_prev) @ v
    return torch.log(v.sum()) + log_norm


def test_two_state_gopich_szabo_handcoded():
    """G=2 grid evaluator == independent hand-coded 2-state GS oracle (SPEC 8.1).

    The 2x2 rate matrix is built directly from the sqrt-approx (NOT via
    ``smoluchowski``) and propagated by a closed-form 2x2 exp (NOT eigh /
    matrix_exp), so this is a genuine external oracle, not a self-consistency
    check.  It also catches a bug in the G=2 generator construction.
    """
    torch.manual_seed(7)
    x0, x1 = 4.8, 6.6
    grid = torch.tensor([x0, x1])
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(280.0, 0.88, 0.92, 18.0, 33.0)
    D = torch.tensor(6.0)
    dx = float(grid[1] - grid[0])

    u0, u1 = 0.0, 1.1                      # two wells (kBT); heights
    delta = u1 - u0
    pref = float(D) / (dx * dx)            # sqrt-approx 2-state rates == smoluchowski(G=2)
    k01 = pref * math.exp(-delta / 2.0)    # 0 -> 1
    k10 = pref * math.exp(+delta / 2.0)    # 1 -> 0

    mu_G, mu_R = _emission_on_grid(grid, rates, C, consts.R0)
    w = torch.tensor([math.exp(-u0), math.exp(-u1)])
    p0 = w / w.sum()                       # stationary pi ∝ e^{-u}

    gaps = torch.rand(14) * 0.02
    times = torch.cumsum(gaps, 0)
    colors = torch.randint(0, 2, (14,))
    T = float(times[-1]) + 0.004

    oracle = _two_state_gs_loglik(k01, k10, mu_G, mu_R, times, colors, T, p0)

    # grid evaluator: spline with 2 knots at the grid points -> u(grid) = theta
    pot = dfl.build_potential(dfl.PotentialConfig(kind="spline", n_knots=2), grid)
    with torch.no_grad():
        pot.theta.copy_(torch.tensor([u0, u1]))
    fast = dfl.marginal_loglik(times, colors, T, pot, D, rates, grid, C, consts.R0)
    assert abs(float(oracle) - float(fast)) < 1e-8, f"{float(oracle)} vs {float(fast)}"


def test_batched_equals_single(small_setup):
    """Batched lockstep recursion == per-trace single recursion."""
    s = small_setup
    # build two traces
    g1 = torch.rand(6) * 0.01; g1[0] = 0.0
    g2 = torch.rand(9) * 0.01; g2[0] = 0.0
    c1 = torch.randint(0, 2, (6,)); c2 = torch.randint(0, 2, (9,))
    Kmax = 9
    ipt = torch.zeros(2, Kmax); colors = torch.zeros(2, Kmax, dtype=torch.long)
    mask = torch.zeros(2, Kmax, dtype=torch.bool)
    ipt[0, :6] = g1; colors[0, :6] = c1; mask[0, :6] = True
    ipt[1, :9] = g2; colors[1, :9] = c2; mask[1, :9] = True

    batch_ll = dfl.marginal_loglik_batch(ipt, colors, mask, s["pot"], s["D"],
                                         s["rates"], s["grid"], s["C"], s["R0"],
                                         reduce="none")
    # single-trace: times = cumsum(gaps), T = times[-1]
    t1 = torch.cumsum(g1, 0)
    t2 = torch.cumsum(g2, 0)
    ll1 = dfl.marginal_loglik(t1, c1, float(t1[-1]), s["pot"], s["D"], s["rates"],
                              s["grid"], s["C"], s["R0"])
    ll2 = dfl.marginal_loglik(t2, c2, float(t2[-1]), s["pot"], s["D"], s["rates"],
                              s["grid"], s["C"], s["R0"])
    assert abs(float(batch_ll[0]) - float(ll1)) < 1e-8
    assert abs(float(batch_ll[1]) - float(ll2)) < 1e-8


def test_gradcheck_marginal_loglik():
    """gradcheck of marginal_loglik wrt D, effective rates, and spline theta."""
    G = 8
    grid = torch.linspace(4.0, 8.0, G)
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    pcfg = dfl.PotentialConfig(kind="spline", n_knots=4)
    pot = dfl.build_potential(pcfg, grid)
    gaps = torch.rand(6) * 0.02
    times = torch.cumsum(gaps, 0)
    colors = torch.tensor([0, 1, 0, 0, 1, 1])
    T = float(times[-1])

    def f(logD, a_g, a_r, bg_g, bg_r, theta):
        with torch.no_grad():
            pass
        # rebuild spline theta as a differentiable input
        pot.theta.data = theta  # not tracked; use functional instead
        rates = dfl.EffectiveRates(a_g, a_r, bg_g, bg_r)
        # inject theta via a temporary parameter swap that keeps grad
        u = (pot._basis(grid) @ theta)
        from diff_fret_likelihood.forward import build_propagator_from_u, _forward_recursion_single
        u = u - u.min()
        prop = build_propagator_from_u(u, logD.exp(), rates, grid, C, consts.R0,
                                       float(grid[1] - grid[0]))
        return _forward_recursion_single(times, colors, T, prop, u, None)

    logD = torch.tensor(1.0, requires_grad=True)
    a_g = torch.tensor(200.0, requires_grad=True)
    a_r = torch.tensor(210.0, requires_grad=True)
    bg_g = torch.tensor(20.0, requires_grad=True)
    bg_r = torch.tensor(40.0, requires_grad=True)
    theta = torch.tensor([0.3, -0.5, 0.6, -0.2], requires_grad=True)
    assert torch.autograd.gradcheck(f, (logD, a_g, a_r, bg_g, bg_r, theta),
                                    atol=1e-5, rtol=1e-3, eps=1e-6)


# ---------------------------------------------------------------------------
# SPEC Remark 1: gaps tile [0, T] exactly once (no double-counted first interval)
# ---------------------------------------------------------------------------
def test_no_double_count_first_gap():
    """G=1 closed form with an explicit leading AND trailing gap (SPEC Remark 1).

    logL = #G logμ_G + #R logμ_R - μ·T over the FULL window T.  A double-counted
    pre-first interval [0,t1] would give the survival exponent -μ·(T+t1) and fail
    by μ·t1 (>> tol) -- this test locks that in.
    """
    grid = torch.tensor([6.0])
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300.0, 0.85, 0.85, 25.0, 50.0)
    pot = dfl.build_potential(
        dfl.PotentialConfig(kind="spline", n_knots=2, x_center=6.0, x_scale=1.0), grid)
    mu_G, mu_R = _emission_on_grid(grid, rates, C, consts.R0)
    mu = float(mu_G + mu_R)

    t1 = 0.003                                   # explicit non-zero leading gap
    times = torch.tensor([t1, 0.006, 0.013])
    colors = torch.tensor([0, 1, 1])
    T = 0.021                                    # explicit trailing gap (> t_K)
    ll = dfl.marginal_loglik(times, colors, T, pot, torch.tensor(5.0),
                             rates, grid, C, consts.R0)
    n_g = int((colors == 0).sum()); n_r = int((colors == 1).sum())
    expect = n_g * math.log(float(mu_G)) + n_r * math.log(float(mu_R)) - mu * T
    assert abs(float(ll) - expect) < 1e-8, f"{float(ll)} vs {expect}"
    assert mu * t1 > 1e-6                         # the failure mode is well above tol


def test_marginal_loglik_gap_tiling_guard():
    """SPEC Remark 1 safeguard: reject mis-tiled input (unsorted / T < t_K)."""
    grid = torch.linspace(4.0, 8.0, 6)
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300.0, 0.85, 0.85, 25.0, 50.0)
    pot = dfl.build_potential(dfl.PotentialConfig(kind="spline", n_knots=4), grid)
    D = torch.tensor(8.0)
    times = torch.tensor([0.002, 0.005, 0.011])
    colors = torch.tensor([0, 1, 0])

    # valid: sorted times, T >= t_K -> no raise
    dfl.marginal_loglik(times, colors, 0.02, pot, D, rates, grid, C, consts.R0)

    # T < times[-1] -> negative trailing gap -> raise
    with pytest.raises(AssertionError):
        dfl.marginal_loglik(times, colors, 0.008, pot, D, rates, grid, C, consts.R0)

    # unsorted arrival times -> raise
    bad = torch.tensor([0.002, 0.011, 0.005])
    with pytest.raises(AssertionError):
        dfl.marginal_loglik(bad, colors, 0.02, pot, D, rates, grid, C, consts.R0)
