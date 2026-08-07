"""Initializers for the marginal-likelihood fit.

The marginal objective ``-log p`` is non-convex in the landscape parameters, so a
good starting point matters.  Both helpers here are pure, used *before* ``fit``, and
change nothing in the fit path itself:

* ``warmstart_potential`` -- set a potential so ``potential.on_grid(grid) ~= u_target``
  for an externally supplied ``[G]`` profile.  Exact (least squares) for a
  ``SplinePotential``, which is linear in its knots; a short Adam regression for any
  other potential.
* ``estimate_rates`` -- rough initial ``EffectiveRates`` from a photon ``Batch``.  Only
  the per-channel *total* rate is observable from a photon stream, so this is a
  starting point for ``fit(fit_rates=True)``, not a calibration.

Usage::

    from diff_fret_likelihood import init
    init.warmstart_potential(pot, grid, u_target)        # u_target: [G] tensor/array
    rates = init.estimate_rates(batch, bg_frac=0.5, device=device)
    res = dfl.fit(batch, grid, pot, C, R0, D_init=D0, rates_init=rates, prior=prior)
"""

from __future__ import annotations

import torch

from .config import DTYPE
from .potential import SplinePotential
from .photophysics import EffectiveRates


def warmstart_potential(
    potential,
    grid: torch.Tensor,
    u_target,
    *,
    steps: int = 500,
    lr: float = 0.05,
):
    """Set ``potential`` so ``potential.on_grid(grid) ~= u_target`` (in place).

    ``SplinePotential`` is fit exactly by least squares (linear in its knots);
    any other potential (e.g. the MLP) is regressed to the target with a short
    Adam loop.  Returns the (mutated) potential.
    """
    u_target = torch.as_tensor(u_target, dtype=DTYPE, device=grid.device).reshape(-1).detach()

    if isinstance(potential, SplinePotential):
        M = potential._basis(grid)                            # [G, n_knots]
        sol = torch.linalg.lstsq(M, u_target.unsqueeze(1)).solution.reshape(-1)
        with torch.no_grad():
            potential.theta.copy_(sol)
        return potential

    opt = torch.optim.Adam(potential.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((potential.on_grid(grid) - u_target) ** 2).mean()
        loss.backward()
        opt.step()
    return potential


def estimate_rates(batch, *, bg_frac=0.10, bg_g=None, bg_r=None, device=None):
    """Rough data-driven initial EffectiveRates (a_g, a_r, bg_g, bg_r) from a photon Batch.

    Emission model: mu_G(x)=a_g·f_g(x)+bg_g, mu_R(x)=a_r·f_r(x)+bg_r (kHz). Only the per-channel
    TOTAL rate is observable from a photon stream, so this is a STARTING point for fit_rates=True:
      a_g = total green-channel rate, a_r = total red-channel rate (pooled over traces),
      bg_g/bg_r = bg_frac · the corresponding channel rate.
    Pass bg_g/bg_r (kHz) if you have calibrated backgrounds. Returns float64 tensors on `device`.
    """
    dev = device if device is not None else batch.ipt.device
    mask = batch.mask
    n_ph = float(mask.sum())                        # total photons
    total_T = float(batch.T.sum())                  # total observation time (ms)
    rate = n_ph / max(total_T, 1e-9)                # overall photon rate (kHz)
    n_red = float((batch.colors * mask).sum())      # red photons (color == 1)
    frac_red = n_red / max(n_ph, 1.0)
    a_g_val = rate * (1.0 - frac_red)               # total green channel rate
    a_r_val = rate * frac_red                       # total red channel rate
    bg_g_val = bg_frac * a_g_val if bg_g is None else float(bg_g)
    bg_r_val = bg_frac * a_r_val if bg_r is None else float(bg_r)
    t = lambda v: torch.tensor(float(v), dtype=torch.float64, device=dev)
    return EffectiveRates(t(a_g_val), t(a_r_val), t(bg_g_val), t(bg_r_val))