"""Detailed-balance Smoluchowski generator on a 1-D grid (SPEC section 4.6).

The generator acts on probability *column* vectors as ``dp/dt = L p`` (columns
sum to zero; off-diagonal entry ``L[j,i]`` is the rate i->j).  Using the
detailed-balance (finite-volume) discretisation

    L[i+1, i] = (D/dx^2) exp(-(u_{i+1}-u_i)/2)
    L[i-1, i] = (D/dx^2) exp(-(u_{i-1}-u_i)/2)
    L[i, i]   = -sum_{j != i} L[j, i]

guarantees ``pi_i L[j,i] = pi_j L[i,j]`` with ``pi_i propto e^{-u_i}`` -- i.e.
the stationary distribution is Boltzmann for *any* potential.  Reflecting
(no-flux) boundaries: edge states have a single neighbour and columns still sum
to zero (no probability leaks out).

The symmetrised operator ``L_sym = diag(s)^{-1} L diag(s)`` with
``s_i = e^{-u_i/2}`` is symmetric; the tilted operator ``L_sym - diag(mu)`` is
still symmetric (diagonal shift), so it is diagonalisable with ``eigh``.
"""

from __future__ import annotations

import torch

from .config import DTYPE


def min_gauge(u_grid: torch.Tensor) -> torch.Tensor:
    """``u - u.min()``: the overflow guard applied wherever ``u`` feeds ``exp(-u)``.

    This is NOT an identifiability fix and has nothing to do with the gauge *anchor*
    (``objective.gauge_penalty``) or with the grid-mean-zero *reporting* gauge
    (``infer.recovered_potential``).  It exists only so ``exp(-u)`` cannot overflow when
    the grid reaches into a deep well, and it is safe precisely because the likelihood is
    exactly invariant to ``u -> u + const``: subtracting the min changes no observable.
    Needed regardless of whether an anchor is in play.

    The single definition, used by ``stationary`` here, ``forward._BasePotential_on_grid``
    and ``fisher._single_logL`` (which works on a raw ``u`` tensor, not a potential
    object, hence a free function rather than a method).
    """
    return u_grid - u_grid.min()


def smoluchowski(u_grid: torch.Tensor, D: torch.Tensor, dx: float) -> torch.Tensor:
    """Build ``L`` of shape ``[G, G]`` (column-sum-zero generator).

    Hook for position-dependent D: pass a ``[G]`` tensor and use the
    face-averaged value; scalar D is the default.
    """
    G = u_grid.shape[0]
    device = u_grid.device
    if not torch.is_tensor(D):
        D = torch.as_tensor(D, dtype=DTYPE, device=device)

    # Face rates between i and i+1 (there are G-1 faces).
    du = u_grid[1:] - u_grid[:-1]  # u_{i+1} - u_i, shape [G-1]
    pref = D / (dx * dx)
    # rate i -> i+1 (up), and i+1 -> i (down)
    r_up = pref * torch.exp(-du / 2.0)      # L[i+1, i]
    r_down = pref * torch.exp(du / 2.0)     # L[i, i+1]

    L = torch.zeros(G, G, dtype=DTYPE, device=device)
    idx = torch.arange(G - 1, device=device)
    L[idx + 1, idx] = r_up      # from i to i+1
    L[idx, idx + 1] = r_down    # from i+1 to i
    # Diagonal = -(column sum of off-diagonals). Reflecting BCs fall out because
    # edge columns simply have one missing neighbour term.
    col_off = L.sum(dim=0)      # sum over rows for each column i
    L[idx, idx] = 0.0
    L = L - torch.diag(col_off)
    return L


def sqrt_pi(u_grid: torch.Tensor) -> torch.Tensor:
    """``s_i = e^{-u_i/2}`` (unnormalised sqrt of the Boltzmann weight).

    Deliberately does NOT apply ``min_gauge``: its only consumer, ``symmetrize``, forms
    ``s_i / s_j`` ratios, in which any constant shift cancels exactly.
    """
    return torch.exp(-u_grid / 2.0)


def symmetrize(
    L: torch.Tensor, u_grid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(L_sym, s)`` with ``L_sym = diag(s)^{-1} L diag(s)`` symmetric."""
    s = sqrt_pi(u_grid)
    L_sym = (L * s.unsqueeze(0)) / s.unsqueeze(1)  # diag(s)^{-1} L diag(s)
    return L_sym, s


def stationary(u_grid: torch.Tensor) -> torch.Tensor:
    """Normalised Boltzmann stationary distribution ``pi_i propto e^{-u_i}``."""
    w = torch.exp(-min_gauge(u_grid))
    return w / w.sum()


def assert_generator_valid(
    L: torch.Tensor, u_grid: torch.Tensor, atol: float = 1e-8
) -> None:
    """Validate column sums, detailed balance, stationarity, symmetry, spectrum."""
    G = L.shape[0]
    # 1. columns sum to zero (incl. boundaries)
    col = L.sum(dim=0)
    assert torch.allclose(col, torch.zeros_like(col), atol=atol), \
        f"column sums not zero: max |sum|={col.abs().max():.2e}"
    # 2. off-diagonals non-negative
    off = L - torch.diag(torch.diag(L))
    assert (off >= -atol).all(), "negative off-diagonal rate"
    # 3. detailed balance pi_i L[j,i] = pi_j L[i,j]
    pi = stationary(u_grid)
    flux = pi.unsqueeze(0) * L  # flux[j,i] = pi_i L[j,i]  (broadcast over cols)
    assert torch.allclose(flux, flux.T, atol=atol * max(1.0, float(flux.abs().max()))), \
        "detailed balance violated"
    # 4. stationary is a null vector: L pi = 0
    Lpi = L @ pi
    assert torch.allclose(Lpi, torch.zeros_like(Lpi), atol=atol * 10), \
        f"pi not stationary: max|L pi|={Lpi.abs().max():.2e}"
    # 5. L_sym symmetric
    L_sym, _ = symmetrize(L, u_grid)
    assert torch.allclose(L_sym, L_sym.T, atol=atol * max(1.0, float(L_sym.abs().max()))), \
        "L_sym not symmetric"
    # 6. eigenvalues of L_sym <= tol  (NSD generator)
    evals = torch.linalg.eigvalsh(L_sym)
    assert evals.max() <= atol * max(1.0, float(L_sym.abs().max())), \
        f"L_sym has positive eigenvalue {evals.max():.2e}"
