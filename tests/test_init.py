"""Initializers: warm-starting a potential to an external target profile.

Simulator-independent -- the targets are built analytically on a grid, so nothing
here needs the compiled Cython extension.
"""

import torch

import diff_fret_likelihood as dfl
from diff_fret_likelihood import init


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
