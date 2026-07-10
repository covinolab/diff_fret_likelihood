"""Initializers: histogram landscape warm-start, external profiles, D estimate.

All simulator-independent: synthetic Batches are built directly from a known
bimodal x-distribution (no compiled `smFRET_sbi` needed).
"""

import math

import torch

import diff_fret_likelihood as dfl
from diff_fret_likelihood import init


def _bimodal_batch(n_traces=8, n_x=500, R0=6.0, wells=(4.9, 7.1), sd=0.2,
                   ppx=20, seed=0):
    """Batch whose photons reflect x drawn from a 2-well mixture.

    ``ppx`` photons per sampled x, each red with prob ``E(x)`` (so a window of
    ``ppx`` photons reads back ~E(x)).  Timing is arbitrary (unused by the
    histogram init; a constant gap is fine for the D smoke test).
    """
    g = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(n_traces):
        comp = torch.randint(0, 2, (n_x,), generator=g)
        x = torch.where(
            comp == 0,
            wells[0] + sd * torch.randn(n_x, generator=g),
            wells[1] + sd * torch.randn(n_x, generator=g),
        ).clamp_min(1.0)
        E = R0 ** 6 / (R0 ** 6 + x ** 6)
        E_rep = E.repeat_interleave(ppx)
        c = (torch.rand(E_rep.shape, generator=g) < E_rep).long()
        rows.append(c)

    Kmax = max(r.numel() for r in rows)
    B = len(rows)
    ipt = torch.zeros(B, Kmax, dtype=torch.float64)
    colors = torch.zeros(B, Kmax, dtype=torch.int64)
    mask = torch.zeros(B, Kmax, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.int64)
    T = torch.zeros(B, dtype=torch.float64)
    for b, c in enumerate(rows):
        nb = c.numel()
        colors[b, :nb] = c
        mask[b, :nb] = True
        lengths[b] = nb
        ipt[b, 1:nb] = 0.01                       # arbitrary constant gap
        T[b] = float(ipt[b, :nb].sum())
    return dfl.simulate.Batch(ipt, colors, mask, lengths, T)


def test_occupancy_hist_init_recovers_wells():
    R0 = 6.0
    batch = _bimodal_batch(R0=R0)
    grid = dfl.GridConfig(3.5, 9.0, 120).build()
    u = init.occupancy_hist_init(batch, grid, R0)

    left, right = grid < 6.0, grid > 6.0
    xl = float(grid[left][u[left].argmin()])
    xr = float(grid[right][u[right].argmin()])
    assert abs(xl - 4.9) < 0.6, xl
    assert abs(xr - 7.1) < 0.6, xr

    barrier = int((grid - 6.0).abs().argmin())
    assert float(u[barrier]) > float(u[left].min()) + 0.2
    assert float(u[barrier]) > float(u[right].min()) + 0.2


def test_occupancy_hist_init_coarse_bins():
    """Coarse `hist_bins` on a fine grid: still recovers wells, far smoother."""
    R0 = 6.0
    batch = _bimodal_batch(R0=R0)
    grid = dfl.GridConfig(3.5, 9.0, 160).build()

    u_coarse = init.occupancy_hist_init(batch, grid, R0, hist_bins=20)
    u_fine = init.occupancy_hist_init(batch, grid, R0, hist_bins=160)   # grid-resolution

    assert u_coarse.shape[0] == 160                                     # returned on fine grid
    left, right = grid < 6.0, grid > 6.0
    assert abs(float(grid[left][u_coarse[left].argmin()]) - 4.9) < 0.6
    assert abs(float(grid[right][u_coarse[right].argmin()]) - 7.1) < 0.6

    rough = lambda u: float((u[2:] - 2 * u[1:-1] + u[:-2]).pow(2).sum())
    assert rough(u_coarse) < 0.1 * rough(u_fine)                        # much smoother


def test_resolve_u_target_external():
    grid = dfl.GridConfig(4.0, 8.0, 40).build()
    R0 = 6.0

    # (a) [G] tensor -> gauge-fixed passthrough
    prof = (grid - 6.0) ** 2
    out = init.resolve_u_target(prof, grid, R0)
    assert torch.allclose(out, prof - prof.min(), atol=1e-12)

    # (b) (x, u) external profile -> interpolation; same nodes => identity
    out2 = init.resolve_u_target((grid, prof), grid, R0)
    assert torch.allclose(out2, prof - prof.min(), atol=1e-10)

    # (c) callable u(x)
    out3 = init.resolve_u_target(lambda x: (x - 5.0) ** 2, grid, R0)
    tgt = (grid - 5.0) ** 2
    assert torch.allclose(out3, tgt - tgt.min(), atol=1e-12)


def test_warmstart_spline_exact():
    grid = dfl.GridConfig(4.0, 8.0, 50).build()
    pot = dfl.build_potential(dfl.PotentialConfig(kind="spline", n_knots=6), grid)
    # a target inside the spline's range space is recovered to numerical precision
    theta_true = torch.tensor([0.0, 1.0, -0.5, 0.8, -0.3, 0.4], dtype=torch.float64)
    target = pot._basis(grid) @ theta_true

    init.warmstart_potential(pot, grid, target)
    assert torch.allclose(pot.on_grid(grid), target, atol=1e-8)


def test_warmstart_mlp_reduces_error():
    grid = dfl.GridConfig(4.0, 8.0, 60).build()
    pot = dfl.build_potential(dfl.PotentialConfig(kind="mlp", hidden=(32, 32)), grid)
    target = (grid - 6.0) ** 2
    target = target - target.min()

    mse0 = float(((pot.on_grid(grid) - target) ** 2).mean())
    init.warmstart_potential(pot, grid, target, steps=800, lr=0.02)
    mse1 = float(((pot.on_grid(grid) - target) ** 2).mean())
    assert mse1 < 0.3 * mse0, (mse0, mse1)


def test_estimate_D_from_series_ou():
    """OU series with known D = Var/tau is recovered within a small factor."""
    torch.manual_seed(0)
    dt, tau, V = 1.0, 20.0, 4.0
    D_true = V / tau                                  # 0.2
    n = 20000
    a = math.exp(-dt / tau)
    noise = math.sqrt(V * (1.0 - a * a))
    eps = torch.randn(n, dtype=torch.float64)
    x = torch.zeros(n, dtype=torch.float64)
    for t in range(1, n):
        x[t] = a * x[t - 1] + noise * eps[t]

    D_est = init.estimate_D_from_series(x, dt, max_lag=200)
    assert 0.5 * D_true < D_est < 2.0 * D_true, D_est


def test_estimate_D_init_smoke():
    R0 = 6.0
    batch = _bimodal_batch(R0=R0)
    grid = dfl.GridConfig(3.5, 9.0, 120).build()
    D = init.estimate_D_init(batch, grid, R0, min_windows=4)
    assert math.isfinite(D) and D > 0.0
