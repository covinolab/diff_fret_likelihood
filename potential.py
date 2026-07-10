"""Neural / spline potential ``u_theta(x)`` in units of k_B T.

The marginal likelihood only needs ``u`` on the grid (the detailed-balance
generator is built from potential *differences*).  The force ``-du/dx`` is
provided via autograd for the secondary joint objective (SPEC section 7.1).

Two interchangeable parameterisations behind one interface:

* ``MLPPotential`` -- smooth-activation MLP (SPEC primary).  Smooth => C^inf
  force; never ReLU.
* ``SplinePotential`` -- natural-cubic potential-knot spline, linear in the
  free knot heights.  Low-dimensional and very stable; used to cross-check the
  MLP and as a robust fallback.

Gauge fixing (subtract the min) is applied wherever ``u_grid`` feeds the
generator/stationary distribution; the additive constant is pure gauge and
must not change any observable (``tests/test_potential.py``).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch import nn

from .config import DTYPE, PotentialConfig

_ACTIVATIONS = {
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
}


class _BasePotential(nn.Module):
    """Common force / on-grid / gauge machinery."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def force(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``-du/dx`` via autograd (create_graph=True for 2nd-order).

        If ``x`` already requires grad, differentiate through it directly (so
        the force is itself differentiable w.r.t. ``x``); otherwise make a
        grad-enabled copy.  Either way the potential parameters stay tracked.
        """
        x_in = x if x.requires_grad else x.detach().requires_grad_(True)
        u = self.forward(x_in)
        (grad,) = torch.autograd.grad(u.sum(), x_in, create_graph=True)
        return -grad

    def on_grid(self, grid: torch.Tensor) -> torch.Tensor:
        return self.forward(grid)

    @staticmethod
    def gauge_fix(u_grid: torch.Tensor) -> torch.Tensor:
        """Pure gauge: pin the minimum to zero."""
        return u_grid - u_grid.min()


class MLPPotential(_BasePotential):
    """MLP ``u_theta`` with a smooth activation (SPEC primary)."""

    def __init__(self, cfg: PotentialConfig):
        super().__init__()
        if cfg.x_center is None or cfg.x_scale is None:
            raise ValueError("MLPPotential needs x_center and x_scale set "
                             "(call PotentialConfig with grid extent).")
        self.x_center = float(cfg.x_center)
        self.x_scale = float(cfg.x_scale)
        act = _ACTIVATIONS[cfg.activation]
        dims = [1, *cfg.hidden, 1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers).to(DTYPE)
        # Initialise near-flat so the fit starts from a featureless landscape.
        with torch.no_grad():
            last = self.net[-1]
            last.weight.mul_(0.01)
            last.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = (x - self.x_center) / self.x_scale
        return self.net(xn.unsqueeze(-1)).squeeze(-1)


class SplinePotential(_BasePotential):
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

        Overrides the autograd-through-x base method, which cannot flow through
        the NumPy-built basis.  Values are exact; the force is a function of
        ``theta`` (not of ``x`` in the autograd sense), which is what the joint
        objective's parameter gradients need.
        """
        flat = x if x.dim() == 1 else x.reshape(-1)
        M_der = self._basis(flat, deriv=1)
        return -(M_der @ self.theta).reshape(x.shape)


def build_potential(cfg: PotentialConfig, grid: torch.Tensor) -> _BasePotential:
    """Factory: fill in the input-normalisation window from the grid extent."""
    if cfg.x_center is None or cfg.x_scale is None:
        x_min = float(grid.min())
        x_max = float(grid.max())
        cfg = PotentialConfig(
            kind=cfg.kind,
            x_center=0.5 * (x_min + x_max),
            x_scale=0.5 * (x_max - x_min),
            hidden=cfg.hidden,
            activation=cfg.activation,
            n_knots=cfg.n_knots,
        )
    if cfg.kind == "mlp":
        return MLPPotential(cfg)
    if cfg.kind == "spline":
        return SplinePotential(cfg, grid)
    raise ValueError(f"unknown potential kind {cfg.kind!r}")
