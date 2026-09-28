"""Free-energy landscape parametrisations.

The likelihood needs exactly one thing from a landscape: ``U`` on its grid,
``potential.on_grid(grid) -> [G]`` (``forward._BasePotential_on_grid``).  Everything
else the fit path asks of a landscape -- the gauge anchor, the curvature prior, the
warm-start projection -- goes through the small contract :class:`Potential` below, so a
new parametrisation is a subclass that implements ``on_grid`` and opts into whichever
hooks make sense for it.

Three implementations ship here:

* :class:`SplinePotential` -- natural-cubic knot heights, the estimator the package was
  built around (linear in its parameters, gauge-free, curvature prior, exact least-squares
  warm start).
* :class:`FixedPotential` -- a landscape given as a *callable* ``u(x)`` and never
  optimised: it emits no parameters, so :func:`infer.fit` moves only ``D`` and the
  photophysics.  Any torch callable works, including a fitted ``SplinePotential``.
* :class:`ParametricPotential` -- a closed form ``u(x, **params)`` whose named
  parameters are all free.

The grid is an *evaluation-time* argument (``on_grid(grid)``), never a construction
parameter: a landscape is a function of ``x``, the grid belongs to the likelihood
discretisation.  The spline needs its knot *window* at construction (where its degrees
of freedom sit), which is what :class:`config.PotentialConfig` carries;
:func:`build_potential` fills an empty window from the grid's extent as a convenience.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch import nn

from .config import DTYPE, PotentialConfig
from .objective import _scalar_zero, curvature_penalty_spline, gauge_offset_from_theta


class Potential(nn.Module):
    """Contract every landscape parametrisation implements.

    Required:

    * ``on_grid(grid) -> [G]`` -- ``U`` on the likelihood grid.  Differentiable in the
      module's parameters (if it has any); called on every likelihood evaluation.

    Optional, with defaults that describe a landscape *without* the corresponding
    structure:

    * ``forward(x)`` -- ``U`` at arbitrary ``x`` (default: not available).
    * ``force(x)`` -- ``-dU/dx``; the default differentiates ``forward`` by autograd.
    * ``gauge_free`` -- ``True`` iff ``U -> U + c`` is a pure convention of the
      parametrisation (an exact flat direction of the likelihood that the fit must anchor
      and that reporting removes, see :func:`infer.recovered_potential`).  Default
      ``False``: the offset is part of the model and left alone.
    * ``gauge_offset(grid)`` -- the offset coordinate the fit anchors toward zero
      (``objective.gauge_penalty``).  Default: an exact zero, i.e. no anchor.
    * ``curvature_penalty(norm)`` -- roughness term for ``PriorConfig.curvature_weight``.
      Default: raises, because "curvature" is defined on spline knot heights; set the
      weight to ``0`` for other landscapes.
    * ``project(grid, u_target)`` -- set the parameters so ``on_grid(grid) ~= u_target``
      (the KDE warm start, :func:`init.warmstart_potential`).  Default: raises.

    Whether a landscape is optimised is decided by what it *emits*: :func:`infer.fit`
    builds its optimiser list from ``potential.parameters()``, so a class with no
    ``nn.Parameter`` is held fixed automatically (:class:`FixedPotential`).
    """

    gauge_free: bool = False

    def on_grid(self, grid: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} must implement on_grid(grid) -> [G]; it is the one "
            "method the likelihood requires of a landscape."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} does not evaluate U off the grid (no forward(x))."
        )

    def force(self, x: torch.Tensor) -> torch.Tensor:
        """``-dU/dx`` by autograd through ``forward`` (differentiable in the parameters
        when grad is enabled).  Subclasses with an analytic derivative override this."""
        keep = torch.is_grad_enabled()
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            u = self.forward(xg)
            du = None
            if u.requires_grad:
                (du,) = torch.autograd.grad(u.sum(), xg, create_graph=keep, allow_unused=True)
            if du is None:
                raise RuntimeError(
                    f"{type(self).__name__}.force: forward(x) is not differentiable in x "
                    "(e.g. it builds its basis outside autograd), so the force cannot be "
                    "taken by autograd; implement force(x) analytically."
                )
        return -du if keep else -du.detach()

    def gauge_offset(self, grid: torch.Tensor) -> torch.Tensor:
        """Offset coordinate for the fit's gauge anchor; exact zero = nothing to anchor."""
        return _scalar_zero(grid)

    def curvature_penalty(self, norm: str = "l2") -> torch.Tensor:
        raise ValueError(
            f"{type(self).__name__} has no curvature prior: the term is defined on spline "
            "knot heights.  Set PriorConfig.curvature_weight=0 for this landscape."
        )

    def project(self, grid: torch.Tensor, u_target) -> "Potential":
        raise NotImplementedError(
            f"{type(self).__name__} has no warm-start projection (project(grid, u_target)); "
            "set its parameters directly instead."
        )


class SplinePotential(Potential):
    """Natural-cubic potential spline; free params are knot *heights*.

    ``u(grid) = M_val @ theta`` is linear in ``theta`` (fast, stable).  Off-grid
    evaluation (used by ``force``) rebuilds a differentiable cubic-Hermite
    interpolation of the knot heights so autograd flows into ``theta``.

    The knot window comes from ``cfg`` (``x_center``, ``x_scale``, ``n_knots``); use
    :func:`build_potential` to derive it from a grid's extent.
    """

    gauge_free = True   # U -> U + c is the direction (1,...,1) in theta: a pure convention

    def __init__(self, cfg: PotentialConfig):
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

    # ---- the Potential contract, each hook delegating to the pre-existing code ---- #
    def gauge_offset(self, grid: torch.Tensor) -> torch.Tensor:
        """``mean(theta)``: the exact flat direction of a partition-of-unity basis."""
        return gauge_offset_from_theta(self.theta)

    def curvature_penalty(self, norm: str = "l2") -> torch.Tensor:
        """Second differences of the knot heights (``objective.curvature_penalty_spline``)."""
        return curvature_penalty_spline(self.theta, self.knots_x, norm=norm)

    def project(self, grid: torch.Tensor, u_target) -> "SplinePotential":
        """Set ``theta`` so ``on_grid(grid) ~= u_target`` (in place).

        One least-squares solve: the spline is linear in its knot heights, so this is
        the exact projection of ``u_target`` onto the knot basis, not an iterative fit.
        """
        u_target = torch.as_tensor(u_target, dtype=DTYPE, device=grid.device).reshape(-1).detach()
        M = self._basis(grid)                                # [G, n_knots]
        sol = torch.linalg.lstsq(M, u_target.unsqueeze(1)).solution.reshape(-1)
        with torch.no_grad():
            self.theta.copy_(sol)
        return self


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
    return SplinePotential(cfg)


class FixedPotential(Potential):
    """A landscape given as a callable ``u(x) -> tensor``; never optimised.

    It has no parameters, so :func:`infer.fit` leaves the landscape alone and fits only
    ``D`` and the photophysics.  Any torch callable works: a lambda, a closed form, or a
    fitted ``SplinePotential`` (an ``nn.Module`` is callable, and its ``forward(grid)``
    equals its ``on_grid(grid)`` bit for bit) -- the idiom for "hold the landscape I
    just fitted fixed and refit ``D``".

    ``on_grid`` evaluates the callable on the grid it is given, detaches the result (a
    wrapped module's own parameters must not enter the graph) and caches it per grid
    object, exactly as the spline caches its basis.  The callable is *not* registered as
    a submodule, so ``.to(device)`` does not move it: wrap a module that already lives on
    the grid's device.  ``gauge_free`` is ``False``: the offset is whatever the callable
    says, and :func:`infer.recovered_potential` reports it unshifted.
    """

    gauge_free = False

    def __init__(self, fn):
        super().__init__()
        if not callable(fn):
            raise TypeError(f"FixedPotential expects a callable u(x) -> tensor, got {type(fn).__name__}")
        # Bypass nn.Module.__setattr__: a wrapped module would otherwise be registered as
        # a submodule and its parameters would leak into ``parameters()`` -- i.e. into fit.
        object.__setattr__(self, "_fn", fn)
        self._cached_grid = None
        self._u_val = None

    def on_grid(self, grid: torch.Tensor) -> torch.Tensor:
        if self._cached_grid is not grid or self._u_val is None:
            u = torch.as_tensor(self._fn(grid), dtype=DTYPE, device=grid.device)
            self._u_val = u.detach().reshape(-1)
            self._cached_grid = grid
        return self._u_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fn(x)

    def curvature_penalty(self, norm: str = "l2") -> torch.Tensor:
        # No parameters: the term would be a constant, and a constant is exactly zero here.
        return torch.zeros((), dtype=DTYPE)

    def extra_repr(self) -> str:
        return getattr(self._fn, "__name__", type(self._fn).__name__)


class ParametricPotential(Potential):
    """A closed-form landscape ``u(x, **params)`` with named, free parameters.

    ``fn(x, **physical) -> tensor`` must use torch operations (autograd carries the
    gradients to the parameters).  Every entry of ``params`` becomes an ``nn.Parameter``,
    registered in dict order so the optimiser's parameter order is deterministic.  Names
    in ``positive`` are stored as ``log_<name>`` and exponentiated in :meth:`physical`, so
    they stay positive through a fit.

    ``gauge_param`` names an additive-constant parameter, if the form has one (``U + c``):
    that direction is exactly flat for the likelihood, so the fit anchors it toward zero
    like the spline's ``mean(theta)`` and reporting removes it (``gauge_free = True``).
    Without it the offset is part of the model and left alone.
    """

    def __init__(self, fn, params: dict, *, positive=(), gauge_param=None, dtype=DTYPE):
        super().__init__()
        if not callable(fn):
            raise TypeError(f"ParametricPotential expects a callable u(x, **params), got {type(fn).__name__}")
        params = dict(params)
        if not params:
            raise ValueError("ParametricPotential needs at least one parameter; use FixedPotential for a fixed form.")
        positive = tuple(positive)
        unknown = sorted(set(positive) - set(params))
        if unknown:
            raise ValueError(f"ParametricPotential: positive names {unknown} are not in params {sorted(params)}")
        if gauge_param is not None:
            if gauge_param not in params:
                raise ValueError(f"ParametricPotential: gauge_param {gauge_param!r} is not in params {sorted(params)}")
            if gauge_param in positive:
                raise ValueError("ParametricPotential: the gauge offset is anchored toward 0 and cannot be log-parametrised (positive)")
        object.__setattr__(self, "_fn", fn)
        self._names = tuple(params)
        self._positive = frozenset(positive)
        self._gauge_param = gauge_param
        self._dtype = dtype
        self.gauge_free = gauge_param is not None
        for name, value in params.items():          # registration order == parameter order
            attr = f"log_{name}" if name in self._positive else name
            if hasattr(self, attr):
                raise ValueError(f"ParametricPotential: parameter name {name!r} clashes with an attribute of nn.Module")
            v = torch.as_tensor(float(value), dtype=dtype)
            if name in self._positive:
                if not v > 0:
                    raise ValueError(f"ParametricPotential: {name} is declared positive but its init is {value}")
                v = v.log()
            setattr(self, attr, nn.Parameter(v))

    def physical(self) -> dict:
        """``{name: 0-d tensor}`` in the units of ``fn``, differentiable in the parameters."""
        return {
            name: (getattr(self, f"log_{name}").exp() if name in self._positive else getattr(self, name))
            for name in self._names
        }

    def on_grid(self, grid: torch.Tensor) -> torch.Tensor:
        return self._fn(grid.to(self._dtype), **self.physical())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fn(x.to(self._dtype), **self.physical())

    def gauge_offset(self, grid: torch.Tensor) -> torch.Tensor:
        if self._gauge_param is None:
            return _scalar_zero(grid)
        return self.physical()[self._gauge_param]

    def extra_repr(self) -> str:
        with torch.no_grad():
            vals = ", ".join(f"{k}={float(v):.4g}" for k, v in self.physical().items())
        gauge = f", gauge_param={self._gauge_param!r}" if self._gauge_param is not None else ""
        return f"{getattr(self._fn, '__name__', type(self._fn).__name__)}: {vals}{gauge}"
