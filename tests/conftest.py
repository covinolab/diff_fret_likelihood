"""Shared fixtures: float64 default, seeds, small synthetic setups."""

import math
import os
import sys

import numpy as np
import pytest
import torch

# make `import diff_fret_likelihood` work when running pytest from anywhere.
# NOTE: insert(0, ...) means the SOURCE TREE WINS over any installed copy -- edits are
# what gets tested. Consequence: the tree must also carry the compiled extensions
# (`*.so`, gitignored), or `diff_fret_likelihood.simulator` is missing here even when
# `pip install .` built it into site-packages, and every simulator test skips. CI
# therefore runs `python build_cython.py` after installing (see .github/workflows/ci.yml).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

torch.set_default_dtype(torch.float64)

import diff_fret_likelihood as dfl  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def small_setup():
    """A tiny grid + potential + rates + a short random photon stream."""
    gcfg = dfl.GridConfig(x_min=4.0, x_max=8.0, n_grid=12)
    grid = gcfg.build()
    pcfg = dfl.PotentialConfig(n_knots=5)
    pot = dfl.build_potential(pcfg, grid)
    with torch.no_grad():
        pot.theta.copy_(torch.tensor([0.5, -0.3, 0.8, -0.2, 0.4]))
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(kD=300.0, eta_g=0.85, eta_r=0.85,
                                            beta_g=21.25, beta_r=42.5)
    D = torch.tensor(10.0)
    # short random increasing photon stream (ms)
    K = 8
    gaps = torch.rand(K) * 0.01
    times = torch.cumsum(gaps, 0)
    colors = torch.randint(0, 2, (K,))
    return dict(grid=grid, pot=pot, C=C, rates=rates, D=D,
                times=times, colors=colors, R0=consts.R0)
