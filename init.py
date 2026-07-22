"""Data-driven and external initializers for the marginal-likelihood fit.

The marginal objective ``-log p`` is non-convex in the landscape parameters, so a
good starting point matters (SPEC section 7.6).  This module provides, as pure
helpers used *before* ``fit`` (no change to the fit path):

* ``occupancy_hist_init`` -- landscape warm-start ``u ~= -log(pi_hist)`` from the
  FRET-efficiency histogram (the fast-photon binned limit): bin photons, read off
  the apparent efficiency per window, map ``E -> x`` through the inverse Foerster
  relation, histogram over the grid, and take the negative-log occupancy.
* ``estimate_D_init`` -- a rough diffusion coefficient from the apparent-``x``
  autocorrelation time via the overdamped (OU) relation ``D = Var(x) / tau_c``.
* ``resolve_u_target`` / ``warmstart_potential`` -- apply *either* a data-driven
  profile *or* an external one (a grid array, an ``(x, u)`` pair on arbitrary
  points, or a callable ``u(x)``) to any potential.

Usage (no fit change; external ``D`` is just the existing ``D_init`` float)::

    from diff_fret_likelihood import init
    init.warmstart_potential(pot, grid, init.occupancy_hist_init(batch, grid, R0))
    D0 = init.estimate_D_init(batch, grid, R0)          # or an external float
    # external profile instead:
    init.warmstart_potential(pot, grid, init.resolve_u_target((x_ext, u_ext), grid, R0))
    res = dfl.fit(batch, grid, pot, C, R0, D_init=D0, rates_init=rates, prior=prior)

These estimates are deliberately *rough* (the apparent efficiency is biased by
crosstalk/background and the binning has its own timescale); they are starting
points refined by the fit.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .config import DTYPE
from .potential import SplinePotential
from .photophysics import EffectiveRates


# ---------------------------------------------------------------------------
# FRET inverse map
# ---------------------------------------------------------------------------
def inverse_fret_x(E, R0: float) -> torch.Tensor:
    """Invert ``E(x) = R0^6 / (R0^6 + x^6)``:  ``x = R0 * ((1-E)/E)^(1/6)``."""
    E = torch.as_tensor(E, dtype=DTYPE)
    return R0 * ((1.0 - E) / E) ** (1.0 / 6.0)


# ---------------------------------------------------------------------------
# Histogram landscape warm-start:  u ~= -log(pi_hist)
# ---------------------------------------------------------------------------
def _gaussian_smooth1d(v: torch.Tensor, sigma: float) -> torch.Tensor:
    if not sigma or sigma <= 0:
        return v
    radius = max(1, int(math.ceil(3.0 * sigma)))
    radius = min(radius, v.numel() - 1)
    k = torch.arange(-radius, radius + 1, dtype=v.dtype, device=v.device)
    ker = torch.exp(-0.5 * (k / sigma) ** 2)
    ker = ker / ker.sum()
    vp = torch.nn.functional.pad(v.view(1, 1, -1), (radius, radius), mode="reflect")
    return torch.nn.functional.conv1d(vp, ker.view(1, 1, -1)).view(-1)


def occupancy_hist_init(
    batch,
    grid: torch.Tensor,
    R0: float,
    *,
    photons_per_window: int = 20,
    e_clip: tuple[float, float] = (0.02, 0.98),
    smooth_sigma_bins: float = 1.0,
    u_max: float = 6.0,
    hist_bins: int | None = None,
) -> torch.Tensor:
    """Landscape warm-start ``u_init [G]`` from the FRET-efficiency histogram.

    Bins each trace's photons into windows of ``photons_per_window``, reads the
    apparent efficiency ``E_win = fraction of red photons`` (fast-photon binned
    limit), maps ``E -> x`` and histograms it; the negative-log occupancy is the
    initial potential.  Poorly sampled / FRET-saturated regions (``E`` near 0 or
    1) collect no mass and revert to a flat high plateau (``u_max``), matching
    SPEC section 6.3.

    ``hist_bins`` sets the **histogram resolution**, which should be *coarse*
    relative to the fitting ``grid``: ``-log(counts)`` on a fine grid is spiky and
    clamps in empty bins, a poor warm-start.  The histogram is built on
    ``hist_bins`` bins over the grid's x-range and then linearly interpolated onto
    ``grid`` (so the returned tensor is always ``[G]``).  ``None`` picks a robust
    coarse default (``max(8, min(G, 30))``); pass an int to control it, or
    ``hist_bins >= G`` to histogram at grid resolution.
    """
    G = grid.shape[0]
    nb = hist_bins if hist_bins is not None else max(8, min(G, 30))
    if nb >= G:
        return _hist_u_on_grid(batch, grid, R0, photons_per_window, e_clip,
                               smooth_sigma_bins, u_max)
    # coarse histogram -> interpolate onto the (finer) fitting grid
    coarse = torch.linspace(float(grid.min()), float(grid.max()), int(nb),
                            dtype=DTYPE, device=grid.device)
    u_c = _hist_u_on_grid(batch, coarse, R0, photons_per_window, e_clip,
                          smooth_sigma_bins, u_max)
    u = _interp_to_grid(grid, coarse, u_c)
    return (u - u.min()).clamp(max=u_max)


def _hist_u_on_grid(batch, grid, R0, photons_per_window, e_clip,
                    smooth_sigma_bins, u_max) -> torch.Tensor:
    """``-log`` occupancy histogrammed directly on ``grid`` (see occupancy_hist_init)."""
    device = grid.device
    G = grid.shape[0]
    dx = float(grid[1] - grid[0]) if G > 1 else 1.0
    min_win = max(3, photons_per_window // 2)

    e_lo, e_hi = e_clip
    x_samples = []
    for b in range(batch.n_traces):
        n = int(batch.lengths[b])
        if n < min_win:
            continue
        cols = batch.colors[b, :n].to(DTYPE)
        for ch in torch.split(cols, photons_per_window):
            if ch.numel() >= min_win:
                x_samples.append(ch.mean())
    if not x_samples:
        return torch.zeros(G, dtype=DTYPE, device=device)

    E = torch.stack(x_samples).clamp(e_lo, e_hi)
    x = inverse_fret_x(E, R0).to(device)

    lo = float(grid.min()) - dx / 2.0
    hi = float(grid.max()) + dx / 2.0
    counts = torch.histc(x, bins=G, min=lo, max=hi)          # out-of-range dropped
    counts = _gaussian_smooth1d(counts, smooth_sigma_bins)

    density = counts + 1e-3 * counts.max().clamp_min(1e-12)
    density = density / density.sum()
    u = -torch.log(density)
    u = u - u.min()
    return u.clamp(max=u_max)


# ---------------------------------------------------------------------------
# Profile resolution (data-driven OR external) + apply to a potential
# ---------------------------------------------------------------------------
def _interp_to_grid(grid: torch.Tensor, x_ext, u_ext) -> torch.Tensor:
    g = grid.detach().cpu().numpy()
    xe = np.asarray(x_ext, dtype=np.float64).reshape(-1)
    ue = np.asarray(u_ext, dtype=np.float64).reshape(-1)
    order = np.argsort(xe)
    u = np.interp(g, xe[order], ue[order])                   # flat extrapolation
    return torch.as_tensor(u, dtype=DTYPE, device=grid.device)


def resolve_u_target(spec, grid: torch.Tensor, R0: float, batch=None) -> torch.Tensor:
    """Normalize any landscape-profile ``spec`` to a gauge-fixed ``[G]`` target.

    ``spec`` may be:
      * ``"histogram"``    -> ``occupancy_hist_init(batch, grid, R0)`` (needs ``batch``);
      * a ``[G]`` tensor/array -> an external profile already on the grid;
      * an ``(x_ext, u_ext)`` pair -> an external profile linearly interpolated
        onto the grid (flat-extrapolated beyond its ends);
      * a callable ``u(x)`` -> evaluated on the grid.
    """
    G = grid.shape[0]
    if isinstance(spec, str):
        if spec == "histogram":
            if batch is None:
                raise ValueError("resolve_u_target('histogram', ...) needs `batch`")
            u = occupancy_hist_init(batch, grid, R0)
        else:
            raise ValueError(f"unknown profile spec {spec!r}")
    elif callable(spec):
        u = torch.as_tensor(spec(grid), dtype=DTYPE, device=grid.device).reshape(-1)
    elif isinstance(spec, (tuple, list)) and len(spec) == 2 and not np.isscalar(spec[0]):
        u = _interp_to_grid(grid, spec[0], spec[1])
    else:
        u = torch.as_tensor(spec, dtype=DTYPE, device=grid.device).reshape(-1)

    if u.shape[0] != G:
        raise ValueError(f"resolved profile has length {u.shape[0]}, expected G={G}")
    return u - u.min()                                       # gauge-fix


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


# ---------------------------------------------------------------------------
# Data-driven D from the apparent-x autocorrelation time  (D = Var / tau_c)
# ---------------------------------------------------------------------------
def estimate_D_from_series(x_series, dt: float, *, max_lag: int | None = None,
                           fallback: float = 1.0) -> float:
    """Estimate ``D = Var(x) / tau_c`` from uniform-time-step apparent-x series.

    ``x_series`` is a 1-D tensor/array or a list of them (pooled).  ``tau_c`` is
    the first ``1/e`` crossing of the normalized autocorrelation of the mean-
    subtracted signal (exact for an OU well: ``tau_c = 1/(D*kappa)``,
    ``Var = 1/kappa``).  Degenerate inputs return ``fallback``.
    """
    if not isinstance(x_series, (list, tuple)):
        x_series = [x_series]
    series = [torch.as_tensor(s, dtype=DTYPE).reshape(-1) for s in x_series]
    series = [s - s.mean() for s in series if s.numel() >= 4]
    if not series:
        return float(fallback)

    var = torch.cat(series).pow(2).mean()
    if float(var) <= 0.0:
        return float(fallback)

    L = min(len(s) for s in series) - 1
    if max_lag is not None:
        L = min(L, max_lag)
    L = min(L, 1000)
    if L < 1:
        return float(fallback)

    acf = torch.zeros(L + 1, dtype=DTYPE)
    norm = torch.zeros(L + 1, dtype=DTYPE)
    for d in series:
        n = d.numel()
        for k in range(min(L, n - 1) + 1):
            acf[k] += (d[: n - k] * d[k:]).sum()
            norm[k] += (n - k)
    acf = acf / norm.clamp_min(1.0)
    if float(acf[0]) <= 0.0:
        return float(fallback)
    acf = acf / acf[0]

    thr = 1.0 / math.e
    cross = None
    for k in range(1, L + 1):
        if float(acf[k]) <= thr:
            a0, a1 = float(acf[k - 1]), float(acf[k])
            frac = (a0 - thr) / (a0 - a1) if a0 != a1 else 0.0
            cross = (k - 1) + frac
            break
    if cross is None:                                         # no crossing: integrated ACT
        s = 1.0
        for k in range(1, L + 1):
            if float(acf[k]) <= 0.0:
                break
            s += 2.0 * float(acf[k])
        cross = max(s, 1.0)

    tau_c = cross * dt
    if not (tau_c > 0.0):
        return float(fallback)
    return float(var / tau_c)


def estimate_D_init(
    batch,
    grid: torch.Tensor,
    R0: float,
    *,
    dt_bin: float | None = None,
    photons_per_window: int = 20,
    e_clip: tuple[float, float] = (0.02, 0.98),
    min_windows: int = 8,
    fallback: float = 1.0,
) -> float:
    """Rough ``D_init`` from the FRET-trace autocorrelation time (OU relation).

    Bins each trace's photons into uniform ``dt_bin`` windows, reads the apparent
    efficiency per window (empty windows linearly interpolated), maps ``E -> x``,
    and pools the series into ``estimate_D_from_series``.  ``dt_bin`` defaults to
    ``photons_per_window * median(inter-photon time)`` so each window holds a few
    photons.  Returns ``fallback`` if the data are too sparse.
    """
    lo, hi = float(grid.min()), float(grid.max())
    e_lo, e_hi = e_clip

    all_gaps = []
    for b in range(batch.n_traces):
        n = int(batch.lengths[b])
        if n > 1:
            all_gaps.append(batch.ipt[b, 1:n])
    if not all_gaps:
        return float(fallback)
    if dt_bin is None:
        med_gap = float(torch.cat(all_gaps).median())
        dt_bin = max(med_gap * photons_per_window, 1e-9)

    series = []
    for b in range(batch.n_traces):
        n = int(batch.lengths[b])
        if n < 2 * min_windows:
            continue
        t = torch.cumsum(batch.ipt[b, :n], 0)                 # ipt[0]=0 -> t[0]=0
        cols = batch.colors[b, :n].to(DTYPE)
        Tb = float(t[-1])
        if Tb <= 0:
            continue
        nb = max(min_windows, int(math.ceil(Tb / dt_bin)))
        idx = torch.clamp((t / dt_bin).long(), 0, nb - 1)
        tot = torch.bincount(idx, minlength=nb).to(DTYPE)
        red = torch.bincount(idx, weights=cols, minlength=nb).to(DTYPE)
        valid = tot > 0
        if int(valid.sum()) < min_windows:
            continue
        E = torch.where(valid, red / tot.clamp_min(1.0), torch.zeros_like(tot))
        E = _fill_invalid(E, valid).clamp(e_lo, e_hi)
        x = inverse_fret_x(E, R0).clamp(lo, hi)
        series.append(x)

    if not series:
        return float(fallback)
    return estimate_D_from_series(series, dt_bin, fallback=fallback)


def _fill_invalid(vals: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate invalid entries from valid ones (flat at the ends)."""
    v = vals.detach().cpu().numpy().astype(np.float64)
    m = valid.detach().cpu().numpy().astype(bool)
    if not m.any():
        return vals
    idx = np.arange(v.size)
    v[~m] = np.interp(idx[~m], idx[m], v[m])
    return torch.as_tensor(v, dtype=DTYPE, device=vals.device)


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