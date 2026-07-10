"""Gate 1-2: force correctness (gradcheck) and gauge invariance."""

import torch

import diff_fret_likelihood as dfl


def test_force_equals_neg_grad():
    grid = dfl.GridConfig(4.0, 8.0, 10).build()
    pot = dfl.build_potential(dfl.PotentialConfig(kind="mlp", hidden=(8, 8)), grid)
    x = torch.linspace(4.2, 7.8, 7, requires_grad=True)
    u = pot(x)
    (g,) = torch.autograd.grad(u.sum(), x, create_graph=True)
    assert torch.allclose(pot.force(x), -g, atol=1e-10)


def test_force_gradcheck():
    """torch.autograd.gradcheck on a small net in float64 (SPEC gate 1)."""
    grid = dfl.GridConfig(4.0, 8.0, 10).build()
    pot = dfl.build_potential(dfl.PotentialConfig(kind="mlp", hidden=(6,)), grid)

    def f(x):
        return pot.force(x)

    x = torch.linspace(4.5, 7.5, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (x,), atol=1e-6, rtol=1e-4)


def test_spline_force_matches_numeric():
    """SplinePotential.force (analytic derivative basis) must not raise and must
    match a finite-difference of forward, and stay differentiable in theta."""
    grid = dfl.GridConfig(4.0, 8.0, 50).build()
    pot = dfl.build_potential(dfl.PotentialConfig(kind="spline", n_knots=6), grid)
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
