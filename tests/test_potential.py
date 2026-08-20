"""Gate 1-2: force correctness (gradcheck) and gauge invariance."""

import torch

import diff_fret_likelihood as dfl


def test_force_gradcheck_in_theta():
    """gradcheck of ``force`` w.r.t. the KNOT HEIGHTS (SPEC gate 1).

    The spline's force is analytic in ``theta`` and only in ``theta``: the basis is
    built in NumPy, so ``x`` is not on the autograd graph at all.  That is the whole
    point -- the joint objective needs d(force)/d(theta) -- so gradcheck must be run
    against theta.  (Until 0.2.0 this checked d/dx, which was meaningful only for the
    MLP parameterisation; against a spline it silently compared two zeros, because a
    fresh potential has theta = 0 and the derivative basis is contracted with it.)
    """
    grid = dfl.GridConfig(4.0, 8.0, 10).build()
    pot = dfl.build_potential(dfl.PotentialConfig(n_knots=6), grid)
    x = torch.linspace(4.5, 7.5, 5, dtype=torch.float64)
    M_der = pot._basis(x, deriv=1)

    def f(theta):
        return -(M_der @ theta)

    theta0 = torch.tensor([0.3, -0.9, 0.6, 0.2, -0.4, 0.8],
                          dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (theta0,), atol=1e-6, rtol=1e-4)
    with torch.no_grad():
        pot.theta.copy_(theta0)
    assert torch.allclose(pot.force(x), f(theta0), atol=1e-12)


def test_spline_force_matches_numeric():
    """SplinePotential.force (analytic derivative basis) must not raise and must
    match a finite-difference of forward, and stay differentiable in theta."""
    grid = dfl.GridConfig(4.0, 8.0, 50).build()
    pot = dfl.build_potential(dfl.PotentialConfig(n_knots=6), grid)
    with torch.no_grad():
        pot.theta.copy_(torch.tensor([0.2, -0.6, 0.9, -0.3, 0.5, -0.1]))
    x = torch.linspace(4.3, 7.7, 9)
    F = pot.force(x)  # must not raise
    h = 1e-5
    fd = -(pot(x + h) - pot(x - h)) / (2 * h)
    assert torch.allclose(F, fd, atol=1e-4), f"{F} vs {fd}"
    # differentiable in theta
    loss = pot.force(x).sum()
    (g,) = torch.autograd.grad(loss, pot.theta)
    assert g.abs().sum() > 0


def test_gauge_invariance(small_setup):
    """Adding a constant to u leaves marginal_loglik unchanged (~1e-10)."""
    s = small_setup
    ll0 = dfl.marginal_loglik(s["times"], s["colors"], None, s["pot"], s["D"],
                              s["rates"], s["grid"], s["C"], s["R0"])
    # shift the potential by a constant via the spline knot heights
    with torch.no_grad():
        s["pot"].theta.add_(3.14159)
    ll1 = dfl.marginal_loglik(s["times"], s["colors"], None, s["pot"], s["D"],
                              s["rates"], s["grid"], s["C"], s["R0"])
    assert torch.allclose(ll0, ll1, atol=1e-9), f"{float(ll0)} vs {float(ll1)}"
