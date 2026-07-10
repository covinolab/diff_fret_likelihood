"""Gate 3: detailed-balance generator validity."""

import torch

import diff_fret_likelihood as dfl
from diff_fret_likelihood.generator import smoluchowski, stationary


def _random_u(G):
    x = torch.linspace(0, 1, G)
    # a wiggly but smooth potential
    return 3.0 * torch.sin(4 * x) + 2.0 * x ** 2


def test_generator_valid_interior_and_boundary():
    G = 40
    u = _random_u(G)
    D = torch.tensor(7.0)
    dx = 1.0 / (G - 1)
    L = smoluchowski(u, D, dx)
    # this asserts column sums 0 (incl boundaries), DB, stationary=e^{-u},
    # L_sym symmetric, eigenvalues <= 0
    dfl.assert_generator_valid(L, u, atol=1e-8)


def test_stationary_matches_boltzmann():
    G = 30
    u = _random_u(G)
    D = torch.tensor(3.3)
    dx = 1.0 / (G - 1)
    L = smoluchowski(u, D, dx)
    # null vector of L (via eig) should be proportional to e^{-u}
    w, V = torch.linalg.eig(L)
    idx = int(w.real.abs().argmin())
    null = V[:, idx].real
    null = null / null.sum()
    assert torch.allclose(null, stationary(u), atol=1e-6)


def test_reflecting_no_leak():
    """Total probability is conserved: 1^T L = 0 (columns sum to zero)."""
    G = 25
    u = _random_u(G)
    L = smoluchowski(u, torch.tensor(1.0), 1.0 / (G - 1))
    ones = torch.ones(G)
    assert torch.allclose(ones @ L, torch.zeros(G), atol=1e-9)
