"""Gate 9 (light): marginal_loglik stabilises as the grid is refined."""

import torch

import diff_fret_likelihood as dfl


def _ll_at(G, times, colors, T, theta_vals):
    grid = dfl.GridConfig(4.0, 8.0, G).build()
    pot = dfl.build_potential(dfl.PotentialConfig(kind="spline", n_knots=5), grid)
    with torch.no_grad():
        pot.theta.copy_(theta_vals)
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300.0, 0.85, 0.85, 25.0, 50.0)
    return float(dfl.marginal_loglik(times, colors, T, pot, torch.tensor(10.0),
                                     rates, grid, C, consts.R0))


def test_grid_convergence():
    torch.manual_seed(2)
    gaps = torch.rand(20) * 0.01
    times = torch.cumsum(gaps, 0)
    colors = torch.randint(0, 2, (20,))
    T = float(times[-1])
    theta = torch.tensor([0.5, -0.3, 0.8, -0.2, 0.4])

    lls = [_ll_at(G, times, colors, T, theta) for G in (60, 120, 240)]
    # successive differences shrink -> converging
    d1 = abs(lls[1] - lls[0])
    d2 = abs(lls[2] - lls[1])
    assert d2 < d1, f"not converging: {lls}"
    assert d2 < 0.5, f"still moving a lot at G=240: {lls}"
