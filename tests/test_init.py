"""Initializers: warm-starting a potential, and the FRET-histogram (KDE) warm start.

The ``warmstart_*`` tests are simulator-independent -- their targets are built
analytically on a grid.  The KDE tests are NOT: they need a real photon stream, so
they simulate a small one and skip when the Cython extension is missing.
"""

import numpy as np
import pytest
import torch

import diff_fret_likelihood as dfl
from diff_fret_likelihood import init


def test_warmstart_spline_exact():
    grid = dfl.GridConfig(4.0, 8.0, 50).build()
    pot = dfl.build_potential(dfl.PotentialConfig(n_knots=6), grid)
    # a target inside the spline's range space is recovered to numerical precision
    theta_true = torch.tensor([0.0, 1.0, -0.5, 0.8, -0.3, 0.4], dtype=torch.float64)
    target = pot._basis(grid) @ theta_true

    init.warmstart_potential(pot, grid, target)
    assert torch.allclose(pot.on_grid(grid), target, atol=1e-8)


def test_warmstart_reduces_error_off_span():
    """A target the knot basis cannot represent exactly is still projected onto it.

    The test above covers a target built FROM the basis, where the solve is exact.
    Here the target (a parabola sampled on 60 grid points, 12 knots) is not in the
    span, so this checks the least-squares projection actually descends rather than
    only reproducing what it was handed.
    """
    grid = dfl.GridConfig(4.0, 8.0, 60).build()
    pot = dfl.build_potential(dfl.PotentialConfig(n_knots=12), grid)
    target = (grid - 6.0) ** 2
    target = target - target.min()

    mse0 = float(((pot.on_grid(grid) - target) ** 2).mean())
    init.warmstart_potential(pot, grid, target)
    mse1 = float(((pot.on_grid(grid) - target) ** 2).mean())
    assert mse1 < 0.3 * mse0, (mse0, mse1)


# ---------------------------------------------------------------------------
# FRET-histogram (KDE) warm start
# ---------------------------------------------------------------------------
# a single wide well, so p(x) is unambiguous at this trace count
X_KNOTS = np.linspace(3.0, 9.0, 7)
Y_KNOTS = np.array([20.0, 8.0, 1.5, 0.0, 1.5, 8.0, 20.0])
KD, ETA, BETA_G, BETA_R = 30.0, 0.8, 1.6, 4.0
C_GR, C_RG, R0, D_TRUE = 0.03, 0.08, 6.0, 1.5
X_MIN, X_MAX = 4.0, 8.0


@pytest.fixture(scope="module")
def kde_setup():
    """A small simulated photon stream plus everything the KDE init needs."""
    pytest.importorskip(
        "diff_fret_likelihood.simulator",
        reason="the Cython simulator extension is not built (needs GSL)",
    )
    batch = dfl.simulate.simulate_equilibrium(
        X_KNOTS, Y_KNOTS, D_TRUE, R0, KD, BETA_G, BETA_R, ETA, ETA, C_GR, C_RG,
        50.0, 5.0e-6, n_traces=32, n_workers=4, seed=0, device="cpu", verbose=False,
    )
    grid = dfl.GridConfig(X_MIN, X_MAX, 96).build()
    physics = dfl.PhysicsConstants(R0=R0, C_gg=1.0 - C_GR, C_gr=C_GR,
                                   C_rg=C_RG, C_rr=1.0 - C_RG)
    rates = init.stream_rates(batch, bg_g=BETA_G, bg_r=BETA_R)
    return batch, grid, physics, rates


def _spline(grid, n_knots=9):
    return dfl.build_potential(dfl.PotentialConfig(n_knots=n_knots), grid)


def _u_true(grid):
    """The simulated U on the fit grid, mean-centred like the KDE landscape."""
    from scipy.interpolate import CubicSpline
    u = CubicSpline(X_KNOTS, Y_KNOTS)(grid.numpy())
    return u - u.mean()


def test_kde_init_writes_the_landscape_into_the_potential(kde_setup):
    """``potential.on_grid`` reproduces the returned ``u_grid`` -- up to the gauge."""
    batch, grid, physics, rates = kde_setup
    pot = _spline(grid)
    out = init.kde_potential_init(pot, batch, grid, physics, rates, bin_ms=3.0)

    assert out.bin_ms == 3.0 and out.bandwidth > 0 and np.isnan(out.D)
    assert out.u_grid.shape == grid.shape and abs(out.u_grid.mean()) < 1e-9

    d = pot.on_grid(grid).detach().numpy() - out.u_grid
    # a 9-knot spline cannot follow the KDE exactly; what must hold is that the
    # residual is a small *shape* error, not a failure to transfer the landscape
    assert d.std() < 0.1 * out.u_grid.std(), (d.std(), out.u_grid.std())


def test_kde_init_beats_a_flat_landscape(kde_setup):
    """The point of the warm start: it must be closer to truth than flat."""
    batch, grid, physics, rates = kde_setup
    u_t = _u_true(grid)
    rmse = lambda u: float(np.sqrt(((u - u.mean() - u_t) ** 2).mean()))

    pot = _spline(grid)
    init.kde_potential_init(pot, batch, grid, physics, rates, bin_ms=3.0)
    got = rmse(pot.on_grid(grid).detach().numpy())
    flat = rmse(np.zeros_like(u_t))
    assert got < 0.7 * flat, (got, flat)


def test_fret_positions_respect_the_domain_and_min_photons(kde_setup):
    batch, grid, physics, rates = kde_setup
    x = init.fret_positions(batch, physics, rates, bin_ms=3.0,
                            x_min=X_MIN, x_max=X_MAX)
    assert len(x) > 100 and x.min() >= X_MIN and x.max() <= X_MAX
    # a longer bin means fewer, better-determined windows
    assert len(init.fret_positions(batch, physics, rates, bin_ms=6.0)) < \
           len(init.fret_positions(batch, physics, rates, bin_ms=3.0))
    with pytest.raises(ValueError, match="min_photons"):
        init.fret_positions(batch, physics, rates, bin_ms=3.0, min_photons=10**6)


def test_stream_rates_recovers_the_simulated_brightness(kde_setup):
    """a_g = a_r = eta*kD, recovered from the measured rate given the true backgrounds."""
    _, _, _, rates = kde_setup
    for a in (rates.a_g, rates.a_r):
        assert float(a) == pytest.approx(ETA * KD, rel=0.05)
    assert float(rates.bg_g) == BETA_G and float(rates.bg_r) == BETA_R


def test_silverman_bandwidth_matches_the_closed_form():
    x = np.linspace(0.0, 1.0, 101)
    assert init.silverman_bandwidth(x) == pytest.approx(
        1.06 * x.std() * 101 ** (-0.2))


def test_select_bin_ms_returns_a_candidate_from_its_own_grid(kde_setup):
    """Smoke test of the held-out selection: it must choose, and choose legally."""
    batch, grid, physics, rates = kde_setup
    bins = (2.0, 3.0, 5.0)
    out = init.select_bin_ms(batch, grid, _spline(grid), physics, rates,
                             bin_ms_grid=bins, holdout_frac=0.25,
                             D_grid=torch.logspace(-1, 1, 5))

    assert out.bin_ms in bins
    assert len(out.table) == len(bins)
    assert 0.1 <= out.D <= 10.0 and np.isfinite(out.D)
    # the reported winner is the argmax of the held-out column, not an arbitrary row
    assert out.bin_ms == max(out.table, key=lambda r: r[3])[0]
    assert all(np.isfinite(r[3]) for r in out.table)


def test_kde_landscape_floors_the_empty_region():
    """Where there is no data, U must saturate rather than run off to hundreds of kT.

    Samples fill only the left half of the grid.  Without the one-sample density floor
    the right half reads ``-ln 0`` and the relief is unbounded; with it the relief is
    capped at ``ln(samples in the modal bandwidth-window)``, which is at most ``ln n``.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(4.5, 0.25, 2000)
    grid = dfl.GridConfig(4.0, 8.0, 128).build()
    u = init.kde_landscape(x, grid, bandwidth=init.silverman_bandwidth(x))

    assert np.isfinite(u).all()
    assert u.max() - u.min() <= np.log(len(x)) + 1e-9
    # the populated half still carries the real shape: a well at the sample mean
    assert abs(grid.numpy()[u.argmin()] - 4.5) < 0.1
