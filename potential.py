from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch import nn

from .config import DTYPE, PotentialConfig


class SplinePotential(nn.Module):
    """Natural-cubic potential spline; free params are knot *heights*.

    ``u(grid) = M_val @ theta`` is linear in ``theta`` (fast, stable).  Off-grid
    evaluation (used by ``force``) rebuilds a differentiable cubic-Hermite
    interpolation of the knot heights so autograd flows into ``theta``.
    """

    def __init__(self, cfg: PotentialConfig, grid: torch.Tensor | None = None):
        super().__init__()
        if cfg.x_center is None:
            raise ValueError("SplinePotential needs the x window in the cfg.")
        x_min = cfg.x_center - cfg.x_scale
        x_max = cfg.x_center + cfg.x_scale
        knots_x = np.linspace(x_min, x_max, cfg.n_knots)
        self.register_buffer("knots_x", torch.tensor(knots_x, dtype=DTYPE))
        self.theta = nn.Parameter(torch.zeros(cfg.n_knots, dtype=DTYPE))
        self._cached_grid = None   # hold a ref (prevents id() recycling)
        self._M_val = None

    def _basis(self, grid: torch.Tensor, deriv: int = 0) -> torch.Tensor:
        """(G, K) natural-cubic value (``deriv=0``) or derivative (``deriv=1``)
        basis so ``u = M_val @ theta`` / ``du/dx = M_der @ theta`` (both linear
        in ``theta``; the basis itself is constant in ``x``)."""
        knots = self.knots_x.detach().cpu().numpy()
        g = grid.detach().cpu().numpy()
        K, G = knots.size, g.size
        M = np.zeros((G, K))
        for k in range(K):
            e = np.zeros(K)
            e[k] = 1.0
            M[:, k] = CubicSpline(knots, e, bc_type="natural")(g, deriv)
        return torch.tensor(M, dtype=DTYPE, device=grid.device)

    def on_grid(self, grid: torch.Tensor) -> torch.Tensor:
        if self._cached_grid is not grid or self._M_val is None:
            self._M_val = self._basis(grid)
            self._cached_grid = grid
        return self._M_val @ self.theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # value basis (constant in x); linear and differentiable in theta.
        M = self._basis(x if x.dim() == 1 else x.reshape(-1), deriv=0)
        out = M @ self.theta
        return out.reshape(x.shape)

    def force(self, x: torch.Tensor) -> torch.Tensor:
        """Analytic ``-du/dx = -(M_der @ theta)`` (differentiable in ``theta``).

        The force is a function of ``theta`` (not of ``x`` in the autograd sense),
        which is what the joint objective's parameter gradients need.  Autograd
        through ``x`` is not an option here anyway: the basis is built in NumPy.
        """
        flat = x if x.dim() == 1 else x.reshape(-1)
        M_der = self._basis(flat, deriv=1)
        return -(M_der @ self.theta).reshape(x.shape)


def build_potential(cfg: PotentialConfig, grid: torch.Tensor) -> SplinePotential:
    """Factory: fill in the knot window from the grid extent."""
    if cfg.x_center is None or cfg.x_scale is None:
        x_min = float(grid.min())
        x_max = float(grid.max())
        cfg = PotentialConfig(
            x_center=0.5 * (x_min + x_max),
            x_scale=0.5 * (x_max - x_min),
            n_knots=cfg.n_knots,
        )
    return SplinePotential(cfg, grid)
