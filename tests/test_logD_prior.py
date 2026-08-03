"""The Gaussian prior on ``log D`` must actually reach the objective.

``PriorConfig.logD_mean``/``logD_std`` are documented in ``neg_log_posterior`` ("the
curvature and ``logD`` priors act on ``u_grid``/``D``") and reported by
``PriorConfig.active_terms()``, so they must show up in ``prior_penalty`` -- the single
place the prior enters the objective -- and hence in ``neg_log_posterior``.  They were
silently dropped once: ``objective.logD_penalty`` existed but nothing called it, which
made every ``logD_mean`` in every config a no-op in the fit path.
"""

import math

import torch

import diff_fret_likelihood as dfl
from diff_fret_likelihood.objective import logD_penalty, prior_penalty


def _grid(n=16, lo=4.0, hi=8.0):
    return dfl.GridConfig(lo, hi, n).build()


def _spline(grid, theta):
    pot = dfl.build_potential(
        dfl.PotentialConfig(kind="spline", n_knots=len(theta)), grid
    )
    with torch.no_grad():
        pot.theta.copy_(torch.as_tensor(theta, dtype=torch.float64))
    return pot


def _setup():
    grid = _grid()
    pot = _spline(grid, [0.0, 1.0, 0.5, 2.0, 0.0, 1.0])
    return grid, pot


def test_logD_term_enters_prior_penalty():
    """prior_penalty(logD on) - prior_penalty(logD off) == logD_penalty exactly."""
    grid, pot = _setup()
    D = torch.tensor(0.6, dtype=torch.float64)
    mean, std = math.log(1.5), 0.31

    off = dfl.PriorConfig(curvature_weight=0.01)
    on = dfl.PriorConfig(curvature_weight=0.01, logD_mean=mean, logD_std=std)

    delta = float(prior_penalty(pot, D, grid, on) - prior_penalty(pot, D, grid, off))
    expect = 0.5 * ((math.log(0.6) - mean) / std) ** 2

    assert delta == torch.tensor(expect).item() or abs(delta - expect) < 1e-12, (
        f"logD prior contributed {delta} to prior_penalty, expected {expect}"
    )
    assert abs(float(logD_penalty(D, on)) - expect) < 1e-12


def test_logD_prior_pulls_D_toward_its_mean():
    """The penalty is minimised at D = exp(logD_mean) and grows on both sides."""
    grid, pot = _setup()
    mean, std = math.log(1.5), 0.31
    prior = dfl.PriorConfig(logD_mean=mean, logD_std=std)

    at_mean = float(prior_penalty(pot, torch.tensor(1.5, dtype=torch.float64), grid, prior))
    below = float(prior_penalty(pot, torch.tensor(0.6, dtype=torch.float64), grid, prior))
    above = float(prior_penalty(pot, torch.tensor(3.5, dtype=torch.float64), grid, prior))

    assert at_mean < 1e-12
    assert below > 3.0 and above > 3.0


def test_logD_prior_gradient_flows_to_D():
    """The term is differentiable w.r.t. D (it is optimised through log_D)."""
    grid, pot = _setup()
    log_D = torch.tensor(math.log(0.6), dtype=torch.float64, requires_grad=True)
    prior = dfl.PriorConfig(logD_mean=math.log(1.5), logD_std=0.31)

    prior_penalty(pot, log_D.exp(), grid, prior).backward()

    # d/dlogD of 0.5*((logD - m)/s)^2 = (logD - m)/s^2 ; negative below the mean
    expect = (math.log(0.6) - math.log(1.5)) / 0.31 ** 2
    assert log_D.grad is not None
    assert abs(float(log_D.grad) - expect) < 1e-10


def test_active_terms_matches_what_the_objective_charges():
    """`active_terms()` listing 'logD' must mean the objective actually charges for it."""
    grid, pot = _setup()
    D = torch.tensor(0.6, dtype=torch.float64)
    prior = dfl.PriorConfig(logD_mean=math.log(1.5), logD_std=0.31)

    assert "logD" in prior.active_terms()
    assert float(prior_penalty(pot, D, grid, prior)) > 0.0
