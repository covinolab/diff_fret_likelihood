"""Initializers for the marginal-likelihood fit.

The marginal objective ``-log p`` is non-convex in the landscape parameters, so a
good starting point matters.  Everything here is pure, used *before* ``fit``, and
changes nothing in the fit path itself:

* ``warmstart_potential`` -- set a potential so ``potential.on_grid(grid) == u_target``
  for an externally supplied ``[G]`` profile.  Exact (least squares), because the
  spline is linear in its knot heights.
* ``estimate_rates`` / ``stream_rates`` -- initial ``EffectiveRates`` from a photon
  ``Batch``.  They are NOT interchangeable: ``estimate_rates`` splits the observed rate
  by channel, ``stream_rates`` solves the emission model for the per-dye brightness.
  Anything that reads ``a_g``/``a_r`` as brightnesses wants the latter.
* ``kde_potential_init`` -- the FRET-histogram warm start: bin the photon stream in
  time, read each window's apparent efficiency, invert the model's own ``E_app(x)`` map
  to get a distance, KDE those and take ``U = -ln p``.  The bandwidth is fixed by
  Silverman's rule; the one free knob left, the bin width, is chosen by held-out
  marginal likelihood.  Its four steps (``fret_positions``, ``silverman_bandwidth``,
  ``kde_landscape``, ``select_bin_ms``) are public so each can be inspected on its own.

Usage::

    from diff_fret_likelihood import init
    init.warmstart_potential(pot, grid, u_target)        # u_target: [G] tensor/array
    rates = init.estimate_rates(batch, bg_frac=0.5, device=device)
    res = dfl.fit(batch, grid, pot, C, R0, D_init=D0, rates_init=rates, prior=prior)

    # FRET-histogram warm start (sets pot.theta in place, returns the diagnostics).
    # rates default to stream_rates(batch); D_init comes free with the bin-width scan.
    ini = init.kde_potential_init(pot, batch, grid, physics)
    res = dfl.fit(batch, grid, pot, physics.crosstalk_tensor(dev), physics.R0,
                  D_init=ini.D, rates_init=rates, prior=prior)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import numpy as np

from .config import DTYPE
from .photophysics import EffectiveRates


def warmstart_potential(potential, grid: torch.Tensor, u_target):
    """Set ``potential`` so ``potential.on_grid(grid) ~= u_target`` (in place).

    One least-squares solve: the spline is linear in its knot heights, so this is
    the exact projection of ``u_target`` onto the knot basis, not an iterative fit.
    Returns the (mutated) potential.
    """
    u_target = torch.as_tensor(u_target, dtype=DTYPE, device=grid.device).reshape(-1).detach()
    M = potential._basis(grid)                                # [G, n_knots]
    sol = torch.linalg.lstsq(M, u_target.unsqueeze(1)).solution.reshape(-1)
    with torch.no_grad():
        potential.theta.copy_(sol)
    return potential


def estimate_rates(batch, *, bg_frac=0.10, bg_g=None, bg_r=None, device=None):
    """Rough data-driven initial EffectiveRates (a_g, a_r, bg_g, bg_r) from a photon Batch.

    Emission model: mu_G(x)=a_g·f_g(x)+bg_g, mu_R(x)=a_r·f_r(x)+bg_r (kHz). This is a
    STARTING point for fit_rates=True, nothing more:
      a_g = total green-channel rate, a_r = total red-channel rate (pooled over traces),
      bg_g/bg_r = bg_frac · the corresponding channel rate.
    Pass bg_g/bg_r (kHz) if you have calibrated backgrounds. Returns float64 tensors on `device`.

    NOTE the a_g/a_r here are a per-channel SPLIT of the observed rate, not the model's
    brightnesses -- on case_01 they come out (10.7, 18.9) where the truth is (24.0, 24.0),
    i.e. an implied gamma of 1.78 against a true 1.0.  For anything that reads them as
    brightnesses use ``stream_rates`` instead; this one survives because it is what every
    existing fit was started from.
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


# ---------------------------------------------------------------------------
# FRET-histogram (KDE) warm start for the landscape
# ---------------------------------------------------------------------------
DEFAULT_BIN_MS = (0.8, 1.0, 1.2, 1.5, 1.8, 2.1, 2.5, 3.0, 3.5, 4.2, 5.0, 6.0, 7.2, 8.6)


def stream_rates(batch, *, bg_g=None, bg_r=None, bg_frac=0.2, device=None):
    """``EffectiveRates`` whose brightnesses are calibrated against the measured rate.

    Use this, not ``estimate_rates``, wherever ``a_g``/``a_r`` are read as the emission
    model's brightnesses -- which is what they are (``a_g = eta_g kD``).  ``estimate_rates``
    returns something else: the observed rate *split by channel*, which sums to the total
    instead of being the per-dye brightness.  On case_01 that split is (10.7, 18.9) where
    the truth is (24.0, 24.0).

    Here the total detected rate is measured directly from ``batch`` and the model is
    solved for the brightness.  With ``a_g == a_r == a`` and ``f_g + f_r = 1`` the total
    is ``a + bg_g + bg_r`` regardless of position, so ``a = rate_total - bg_g - bg_r``.
    That recovers 23.64 kHz against a truth of 24.00 on both cases.

    ``a_g == a_r`` means ``gamma = 1``, and gamma is CHECKABLE from the photon stream with
    no calibration: if the two brightnesses are equal the per-window total count rate does
    not depend on position, hence not on the observed red fraction.  Measured at 3 ms bins
    the total rate is flat across the full red-fraction range to 0.9% (case_01) and 0.1%
    (case_02).  If your setup has gamma != 1, pass calibrated ``a_g``/``a_r`` and build
    ``EffectiveRates`` directly instead of using this helper.

    ``bg_g``/``bg_r`` are the one thing the stream cannot supply; give them if you have a
    blank or donor-only measurement, otherwise they fall back to ``bg_frac`` of each
    channel's observed rate.
    """
    dev = device if device is not None else batch.ipt.device
    n_ph = float(batch.mask.sum())
    rate = n_ph / max(float(batch.T.sum()), 1e-9)            # total detected rate (kHz)
    frac_red = float((batch.colors * batch.mask).sum()) / max(n_ph, 1.0)
    bg_g_val = bg_frac * rate * (1.0 - frac_red) if bg_g is None else float(bg_g)
    bg_r_val = bg_frac * rate * frac_red if bg_r is None else float(bg_r)
    a = rate - bg_g_val - bg_r_val
    if a <= 0.0:
        raise ValueError(f"backgrounds ({bg_g_val:.3g}, {bg_r_val:.3g} kHz) exceed the "
                         f"measured rate ({rate:.3g} kHz)")
    t = lambda v: torch.tensor(float(v), dtype=torch.float64, device=dev)
    return EffectiveRates(t(a), t(a), t(bg_g_val), t(bg_r_val))


def fret_positions(batch, physics, rates, *, bin_ms, min_photons=5,
                   x_min=None, x_max=None, n_table=4001):
    """Photon stream -> one estimated inter-dye distance per time window, ``[n]`` nm.

    Bins every trace into ``bin_ms`` windows and reads off each window's *apparent*
    efficiency -- the raw red fraction ``n_red / (n_red + n_green)``, the proximity ratio,
    uncorrected.  That apparent efficiency is a known, monotone function of position,

        E_app(x) = mu_R(x) / (mu_G(x) + mu_R(x))

    with ``mu_G``, ``mu_R`` from ``emission_rates`` -- the same forward model the
    likelihood uses.  So the distance follows by inverting that map numerically on a
    table, rather than by rearranging it into an explicit efficiency correction.  Doing
    it this way is exact for any ``a_g``/``a_r``, needs no ``gamma = 1`` assumption in
    the algebra, and keeps the init consistent with the forward model by construction.

    Windows below ``min_photons`` are dropped, as are windows whose apparent efficiency
    falls outside the range the map can produce -- shot noise puts 4-21% of them there
    (most at short bins) and no position explains them, so they carry no distance.  The
    alternative, clipping them onto the domain edge, manufactures density there.
    ``[x_min, x_max]`` bound the table and hence the returned distances.

    ``rates`` must carry emission-model *brightnesses*; build it with ``stream_rates``,
    not ``estimate_rates`` (see the former's docstring for why they differ).  Using the
    raw apparent efficiency as if it were the true one is much worse and gets worse with
    longer bins -- band-rmse 2.19 vs 0.74 at 3.0 ms on case_01, 2.98 vs 1.02 at 8.6 ms --
    because the uncorrected map compresses E into ~[0.16, 0.85] and shrinks the x-axis.

    Time bins, not fixed photon counts: if brightness varies with position, photon
    binning oversamples the bright states.
    """
    from .photophysics import emission_rates

    ipt = batch.ipt.detach().cpu().numpy()
    colors = batch.colors.detach().cpu().numpy()
    mask = batch.mask.detach().cpu().numpy()

    fracs = []
    for tr_ipt, tr_col, m in zip(ipt, colors, mask):
        idx = (np.cumsum(tr_ipt[m]) / float(bin_ms)).astype(np.int64)
        n = np.bincount(idx)
        red = np.bincount(idx, weights=tr_col[m])
        keep = n >= min_photons
        fracs.append(red[keep] / n[keep])
    p_red = np.concatenate(fracs) if fracs else np.empty(0)
    if not len(p_red):
        raise ValueError(f"no window reached min_photons={min_photons} at bin_ms={bin_ms}")

    # tabulate E_app(x) over the domain, then invert it by interpolation
    lo = physics.R0 * 0.25 if x_min is None else float(x_min)
    hi = physics.R0 * 2.50 if x_max is None else float(x_max)
    xs = torch.linspace(lo, hi, int(n_table), dtype=DTYPE)
    C = physics.crosstalk_tensor(xs.device)
    with torch.no_grad():
        mu_G, mu_R = emission_rates(xs, _as_cpu_rates(rates), C, physics.R0)
        e_app = (mu_R / (mu_G + mu_R)).numpy()
    xs = xs.numpy()
    if e_app[0] > e_app[-1]:                       # E_app falls with x; np.interp needs
        e_app, xs = e_app[::-1], xs[::-1]          # an increasing table
    inside = (p_red > e_app[0]) & (p_red < e_app[-1])
    return np.interp(p_red[inside], e_app, xs)


def _as_cpu_rates(rates):
    """``EffectiveRates`` moved to the CPU in float64, for the numpy-side helpers."""
    t = lambda v: torch.as_tensor(v, dtype=DTYPE, device="cpu").detach()
    return EffectiveRates(t(rates.a_g), t(rates.a_r), t(rates.bg_g), t(rates.bg_r))


def silverman_bandwidth(x):
    """Silverman's rule of thumb, ``1.06 * sd * n**(-1/5)``.

    The textbook normal-reference bandwidth, and the right default here because it needs
    no calibration and, on every case tested, sits well above the quantisation lattice of
    ``E = k/N`` -- so it never aliases, which is the one failure mode that turns this warm
    start into a worse-than-flat landscape.  Note it *grows* with the bin width (longer
    bins -> broader p(x)) while the lattice gets finer, so it errs toward over-smoothing
    at long bins; that costs ~5% landscape error and buys the absence of a tuned constant.
    """
    x = np.asarray(x, float)
    return 1.06 * float(np.std(x)) * len(x) ** (-0.2)


def kde_logpdf(samples, points, *, bandwidth, chunk=20_000):
    """log of the Gaussian KDE built from ``samples``, evaluated at ``points``.

    Floored at what a *single* sample contributes at its own centre,
    ``1 / (n h sqrt(2 pi))``.  Below that the KDE is reporting the absence of data
    rather than a density: a region expected to hold under one sample is
    indistinguishable from an empty one.  The floor keeps held-out scores finite when a
    held-out point lands where no training sample did, and caps ``-log p`` at
    ``ln(samples in the modal bandwidth-window)`` -- a statement about how much dynamic
    range n samples support, with no tuned constant.
    """
    pts = np.asarray(points.detach().cpu() if torch.is_tensor(points) else points,
                     float).reshape(-1)
    s = np.asarray(samples, float)
    dens = np.zeros_like(pts)
    for i in range(0, len(s), chunk):
        block = s[i:i + chunk]
        dens += np.exp(-0.5 * ((pts[:, None] - block[None, :]) / bandwidth) ** 2).sum(1)
    norm = max(len(s), 1) * bandwidth * np.sqrt(2 * np.pi)
    return np.log(np.maximum(dens, 1.0) / norm)


def kde_landscape(x, grid, *, bandwidth, chunk=20_000):
    """Gaussian KDE of the samples ``x`` on ``grid``, returned as ``U = -ln p``.

    Mean-centred, since the offset of U is pure gauge.  The density floor of
    ``kde_logpdf`` binds only when the walker does not explore the whole grid: on
    case_01 (samples spanning the full domain) it changes the band-rmse by 0.000 at
    every bin width, while on a tightly confined well that leaves the grid edges empty
    it cuts the relief from 41 kT to 5 and the error from 7.22 to 1.28.
    """
    U = -kde_logpdf(x, grid, bandwidth=bandwidth, chunk=chunk)
    return U - U.mean()


@dataclass
class KDEInit:
    """What the KDE warm start built, and what it had to choose to build it."""

    u_grid: np.ndarray      # [G] the landscape, U = -ln p, mean-centred
    bin_ms: float           # the time-bin width it was built from
    bandwidth: float        # nm, Silverman at that bin width
    D: float                # the D profiled out at that bin -- reusable as D_init
    table: list = field(default_factory=list)   # (bin_ms, bandwidth, D, holdout_logL/trace)

    def __repr__(self):
        return (f"KDEInit(bin_ms={self.bin_ms:g}, bandwidth={self.bandwidth:.3f} nm, "
                f"D={self.D:.3g}, n_scanned={len(self.table)})")


def select_bin_ms(batch, grid, potential, physics, rates=None, *, bin_ms_grid=DEFAULT_BIN_MS,
                  holdout_frac=0.2, min_photons=5, D_grid=None, seed=0, verbose=False,
                  **loglik_kwargs):
    """Choose the KDE bin width by held-out marginal likelihood.  Returns a `KDEInit`.

    The bin width trades two errors that cannot both be made small: short windows are
    shot-noise dominated (the efficiency is estimated from few photons), long ones
    average over the molecule's motion.  No truth-free formula gives the optimum, so it
    is scored: build U from the training traces, profile D against those same traces,
    then score the held-out traces at that D.

    **Scored on the photons, not on the KDE density.**  That is the whole point and it is
    not negotiable -- scoring the density's own held-out samples (ordinary likelihood
    cross-validation) does NOT work here, and was measured: the score rises monotonically
    with bin width and simply walks off the end of the grid (case_01 picks 7.2 ms against
    a truth-optimal 3.0, +30% error; case_02 picks the last grid point).  The reason is
    that each candidate produces its *own* held-out samples: longer bins give fewer,
    less shot-noise-scattered x values, hence a narrower p(x), hence a higher log-density
    per sample -- mechanically, with nothing penalising the destruction of the dynamics.
    The photons are the one reference that does not change with the bin width, so they
    are the only valid basis for the comparison; an over-smoothed landscape predicts the
    actual photon record worse, which is what produces an interior optimum.

    D is a nuisance parameter here, profiled out per candidate exactly as one would
    profile any nuisance parameter to compare models.  It never escapes the comparison;
    it is returned only because it is free once computed and makes a usable ``D_init``.

    ``rates=None`` builds them with ``stream_rates``, which is what both halves want: the
    inversion needs emission brightnesses, and so does the likelihood.  ``batch`` is split
    by a seeded permutation of its traces -- traces, not windows, because windows inside
    one trace track a continuous trajectory and would leak.  ``potential`` is a template
    only, deep-copied per candidate, never modified.  ``loglik_kwargs`` go through to
    ``marginal_loglik_batch`` (e.g. ``compile_mode="default"``,
    ``propagate_dtype=torch.float32``) and are worth setting: this runs
    ``len(bin_ms_grid) * (len(D_grid) + 1)`` full likelihood passes, ~60 s on a GPU.

    Measured: case_01 picks 3.0 ms, the truth-optimal bin; case_02 picks 4.2 ms against a
    truth-optimal 3.0 ms, costing 0.3% landscape error.
    """
    from copy import deepcopy

    from .forward import marginal_loglik_batch
    from .simulate import Batch

    dev = grid.device
    if rates is None:
        rates = stream_rates(batch, device=dev)
    n = batch.n_traces
    n_ho = max(1, int(round(holdout_frac * n)))
    if n_ho >= n:
        raise ValueError(f"holdout_frac={holdout_frac} leaves no training traces (n={n})")
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed)))
    take = lambda i: Batch(batch.ipt[i], batch.colors[i], batch.mask[i],
                           batch.lengths[i], batch.T[i]).to(dev)
    b_tr, b_ho = take(perm[n_ho:]), take(perm[:n_ho])

    if D_grid is None:
        D_grid = torch.logspace(np.log10(0.15), np.log10(8.0), 17)
    D_grid = torch.as_tensor(D_grid, dtype=DTYPE, device=dev).reshape(-1)
    C = physics.crosstalk_tensor(dev)

    rows, best = [], None
    for T in bin_ms_grid:
        x = fret_positions(b_tr, physics, rates, bin_ms=T, min_photons=min_photons,
                           x_min=float(grid[0]), x_max=float(grid[-1]))
        bw = silverman_bandwidth(x)
        u = kde_landscape(x, grid, bandwidth=bw)

        pot = warmstart_potential(deepcopy(potential).to(dev), grid, u)
        ll = lambda b, D: float(marginal_loglik_batch(
            b.ipt, b.colors, b.mask, pot, D, rates, grid, C, physics.R0, **loglik_kwargs))
        with torch.no_grad():
            D_star = float(max(D_grid, key=lambda D: ll(b_tr, D)))
            score = ll(b_ho, torch.as_tensor(D_star, dtype=DTYPE, device=dev)) / b_ho.n_traces

        rows.append((float(T), bw, D_star, score))
        if verbose:
            print(f"  bin_ms {T:>5.2f}   bw {bw:.3f} nm   D* {D_star:>6.3f}   "
                  f"held-out logL/trace {score:>12,.2f}", flush=True)
        if best is None or score > best[0]:
            best = (score, float(T), bw, D_star, u)

    return KDEInit(u_grid=best[4], bin_ms=best[1], bandwidth=best[2], D=best[3], table=rows)


def kde_potential_init(potential, batch, grid, physics, rates=None, *, bin_ms=None,
                       min_photons=5, **scan_kwargs):
    """FRET-histogram warm start: set ``potential`` to ``U = -ln p(x)`` (in place).

    The whole recipe in one call:

      1. bin the photon stream in time, invert the model's E_app(x) map
         -> one distance sample per window            (``fret_positions``)
      2. bandwidth from Silverman's rule              (``silverman_bandwidth``)
      3. Gaussian KDE -> U = -ln p                    (``kde_landscape``)
      4. bin width by held-out marginal likelihood    (``select_bin_ms``)
      5. least-squares projection onto the potential  (``warmstart_potential``)

    ``rates=None`` builds emission brightnesses with ``stream_rates`` -- the right default
    (``estimate_rates`` returns a per-channel split, not brightnesses).  Pass ``bin_ms`` to
    skip step 4, which is the expensive one (it runs the photon-stream likelihood); extra
    keyword arguments otherwise go to ``select_bin_ms``.  ``result.D`` is the D profiled
    out during that scan, reusable as ``D_init`` -- NaN when you supply ``bin_ms``.

    What this gives you and what it does not: feature *positions* come out accurate
    (well and barrier within ~0.05 nm on the tested cases), but shot noise convolves
    p(x) by ~0.3-0.5 nm, so barriers come out roughly 2x too shallow at every bin
    width.  It is a starting basin, not an estimate.
    """
    if bin_ms is None:
        out = select_bin_ms(batch, grid, potential, physics, rates,
                            min_photons=min_photons, **scan_kwargs)
    else:
        if rates is None:
            rates = stream_rates(batch)
        x = fret_positions(batch, physics, rates, bin_ms=bin_ms, min_photons=min_photons,
                           x_min=float(grid[0]), x_max=float(grid[-1]))
        bw = silverman_bandwidth(x)
        out = KDEInit(u_grid=kde_landscape(x, grid, bandwidth=bw),
                      bin_ms=float(bin_ms), bandwidth=bw, D=float("nan"))

    warmstart_potential(potential, grid, out.u_grid)
    return out
