"""Knot domains whose last grid point rounds past x_hi.

``draw_boltzmann_x0`` (simulator.pyx) and the committor integral of ``doob_tpt`` evaluate the
spline on ``x_lo + j*dx``, j = 1..N_grid-1.  On about 1 % of round domains the last point lands
one ulp beyond ``x_hi`` -- [2.4, 7.2] gives 7.200000000000001 -- where GSL returns NaN with an
error status.  Unclamped, the integrals turned NaN and both failed silently: every trace started
at ``x_knots[-1]``, and every transition-path time came back as a single step ``dt``.

Needs the compiled extensions, so every test here skips without them.
"""

import numpy as np
import pytest

simulator = pytest.importorskip(
    "diff_fret_likelihood.simulator",
    reason="the Cython simulator extension is not built (needs GSL)",
)
doob = pytest.importorskip(
    "diff_fret_likelihood.doob_tpt",
    reason="the Cython TPT extension is not built (needs GSL)",
)


def _overshoots(lo, hi, n_grid=2000):
    """True if the last grid point ``x_lo + (n_grid - 1) * dx`` rounds past ``x_hi``."""
    dx = (hi - lo) / (n_grid - 1)
    return lo + (n_grid - 1) * dx > hi


def test_start_positions_are_boltzmann_when_the_grid_overshoots():
    """One step at a huge photon rate: a trace's red fraction is its start's FRET signal.

    Boltzmann starts on this symmetric double well give a mean red fraction of 0.613 (quadrature
    of the same spline); starts pinned to x_hi = 7.2 give 0.18.
    """
    x = np.linspace(2.4, 7.2, 13)
    y = 3.0 * (((x - 4.8) / 1.2) ** 2 - 1.0) ** 2           # wells 3.6 / 6.0, walls 27 kT
    assert _overshoots(x[0], x[-1])
    C_gr, C_rg, dt = 0.03, 0.08, 5e-6
    np.random.seed(0)
    red = []
    for _ in range(300):
        G, R = simulator.smFRET_simulator(1.0, x, y, 5.5, 1e7, 0.0, 0.0, 0.8, 0.8,
                                          1.0 - C_gr, 1.0 - C_rg, C_gr, C_rg, 2 * dt, dt)
        red.append(len(R) / (len(G) + len(R)))
    assert abs(np.mean(red) - 0.613) < 0.06, np.mean(red)


def test_doob_tpt_when_the_grid_overshoots():
    """Well-to-well double well (minima at the domain edges, 3 kT barrier): transition paths take
    ~0.2-0.3 ms here, while the NaN committor made every sample return a single step dt."""
    x = np.linspace(1.2, 3.4, 13)
    y = 3.0 * (((x - 2.3) / 1.1) ** 2 - 1.0) ** 2
    assert _overshoots(x[0], x[-1])
    dt = 1e-5
    np.random.seed(0)
    tpt = np.array([doob.doob_tpt(x, y, 1.0, 0.05, dt, 10**7) for _ in range(5)])
    assert np.all(tpt > 100 * dt), tpt
