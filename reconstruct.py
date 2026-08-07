"""Backward filter and reconstruction of the hidden trajectory (manuscript eqs 75-78).

``forward.py`` runs only the forward filter and returns ``log L``.  Adding the backward
filter

    beta(x, t) = E[ Phi_{t->T} | x(t) = x ],   -d_t beta = (L^dag - lambda_tot) beta,
    beta(x, T) = 1,   beta(x, t_k^-) = lambda_{c_k}(x) beta(x, t_k^+)          (eq 76)

buys two things for about the cost of a second forward pass:

* ``L = <beta(.,t), rho(.,t)>`` for ANY ``t`` in [0, T] (eq 77) -- an independent
  consistency check on the whole evaluator, reported as ``loglik`` / ``loglik_spread``;
* the SMOOTHING posterior over the latent coordinate (eq 78),

      p(x(t) = x | D, theta) = beta(x, t) rho(x, t) / L,

  i.e. a calibrated reconstruction of the distance trajectory with error bands, rather
  than the shot-noise-limited binned "apparent E(t)".

Everything collapses in the symmetric basis the forward filter already uses.  With
``rho = s * v`` (``s = e^{-u/2}``, the ``v`` of ``forward.py``) define ``w := s * beta``.
Because ``(L - Lambda)^T = diag(s)^{-1} A diag(s)`` with ``A = L_sym - diag(mu)``
symmetric, the backward filter uses the SAME propagator ``e^{A tau}`` and the SAME
emission multipliers ``mu_c`` as the forward one, with terminal condition ``w(T) = s`` --
precisely the vector the forward pass already dots against.  Hence

    L = <beta, rho> = sum_i (w_i/s_i)(s_i v_i) = w . v ,     gamma_i = w_i v_i / (w . v)

so every log-normaliser cancels in ``gamma``.  ``gamma`` is also continuous across a
photon (``diag(mu)`` simply moves between the two factors), so "gamma at t_k" is
unambiguous.

Sampling.  The same states give EXACT posterior sample paths (forward-filter /
backward-sample).  The backward conditional is
``p(x_j = i | x_{j+1} = m, D_{<=j}) prop rho_j(i) [e^{(L-Lambda)g}]_{mi}``; substituting
``rho = s*v`` and ``e^{(L-Lambda)g} = diag(s) e^{Ag} diag(s)^{-1}`` cancels both ``s``
factors up to a constant in ``i``, leaving ``v_j(i) (e^{Ag})_{mi}`` with ``e^{Ag}``
symmetric -- so the needed columns are one ``Q (exp(lam g) * Q[m,:])`` matmul for all
paths at once, and no dense kernel is ever formed.

This module is a read-only diagnostic: float64, ``torch.no_grad()``, no ``torch.compile``.
The batched forward recursion in ``forward.py`` is the compiled hot path and is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .forward import _BasePotential_on_grid, build_propagator_from_u
from .generator import stationary


@dataclass
class Reconstruction:
    """Smoothing posterior over the latent coordinate, on the requested output times.

    ``gamma`` is the whole answer; the rest are conveniences.  Anything else is one line
    from it, e.g.::

        E_mean = res.gamma @ dfl.fret_efficiency(res.grid, R0)   # vs binned apparent E
        x_mode = res.grid[res.gamma.argmax(-1)]                  # != mean when bimodal
        cdf    = res.gamma.cumsum(-1)                            # -> any quantile

    Note ``E_mean`` is the pushforward mean and is NOT ``E(x_mean)``; it is the quantity
    comparable to a binned apparent efficiency.
    """

    t: torch.Tensor              # [M] output times (ms), ascending, in [0, T]
    gamma: torch.Tensor          # [M, G] posterior mass per grid cell; rows sum to 1
    grid: torch.Tensor           # [G]
    x_mean: torch.Tensor         # [M] posterior mean position (nm)
    x_sd: torch.Tensor           # [M] posterior SD (nm) -- the error band
    paths: torch.Tensor          # [n_paths, M] sampled trajectories (nm); [0, M] if none
    loglik: float                # log L via <beta, rho>            (eq 77)
    loglik_spread: float         # max |deviation| of that over t   (self-check)


# ---------------------------------------------------------------------------
# Event chain
# ---------------------------------------------------------------------------
def _events(times, colors, T, t_out):
    """Merge the photons with the requested output times into one event chain.

    Returns ``(gaps, tail, codes, keep_idx, t)``: ``gaps`` [N] is the gap *before* each
    event (measured from 0 for the first), ``tail`` is ``T - t_last``, ``codes`` is a
    python list of emission codes (0=green, 1=red, 2=none), ``keep_idx`` a python list of
    the event indices to report, and ``t`` [M] the reported times, ascending.

    An output-only event carries emission 1 and has only a gap, so it is EXACTLY the
    identity in both recursions -- the same fact the batched forward relies on for padded
    steps.  Consequences: one code path serves "gamma at the photon times" and "gamma on a
    plotting lattice", inserting lattice points cannot change gamma at the photon times,
    and sampled paths land on the same time axis as ``x_mean``.

    ``t_out=None`` reports at the photon times themselves rather than inserting a second
    event next to every photon (half the chain, same answer).  Ties are broken
    photon-first (a stable sort of ``[photons, outputs]``), so an output time coinciding
    with a photon reports the post-photon state; ``gamma`` is continuous across a photon,
    so either side is the same value.
    """
    K = times.shape[0]
    if t_out is None:
        ev_t, codes, keep = times, colors, None
    else:
        t_out = torch.sort(t_out.reshape(-1))[0]
        ev_t = torch.cat([times, t_out.to(times.dtype)])
        codes = torch.cat([colors, torch.full_like(t_out, 2, dtype=colors.dtype)])
        keep = torch.cat([torch.zeros_like(colors, dtype=torch.bool),
                          torch.ones_like(t_out, dtype=torch.bool)])
        order = torch.argsort(ev_t, stable=True)        # stable => photons first at ties
        ev_t, codes, keep = ev_t[order], codes[order], keep[order]

    gaps = ev_t - torch.cat([ev_t.new_zeros(1), ev_t[:-1]])
    tail = T - ev_t[-1]
    if keep is None:
        return gaps, tail, codes.tolist(), list(range(K)), ev_t
    return (gaps, tail, codes.tolist(),
            keep.nonzero(as_tuple=True)[0].tolist(), ev_t[keep])


# ---------------------------------------------------------------------------
# The two filters.  Deliberately NOT merged behind a `reverse=` flag: they differ in
# the initial vector, in the gap/emission order within a step, and in whether the state
# is recorded before or after the emission -- exactly the three things a reader needs to
# check.  Keeping them as visibly parallel mirror images is the cleaner call.
# ---------------------------------------------------------------------------
def _sweep_forward(prop, expg, emit, codes, slots, n_keep, v0):
    """Forward filter ``v = rho/s`` at the slotted events, plus log-normalisers.

    Mirrors ``forward._forward_recursion_single``, but with the gap exponentials
    pre-computed (``expg[j] = exp(lam * gap_j)``) so each step is two matmuls and a
    multiply, and with no host sync: the normaliser stays a tensor throughout.
    """
    Q = prop.Q
    V = v0.new_empty(n_keep, v0.shape[0])
    logN = v0.new_empty(n_keep)
    v = v0
    c = v.abs().sum(); v = v / c; log_c = c.log()
    for j in range(len(codes)):
        v = Q @ (expg[j] * (Q.T @ v))              # e^{A gap_j}
        v = v * emit[codes[j]]                     # mu_c, or 1 at an output-only event
        c = v.abs().sum(); v = v / c; log_c = log_c + c.log()
        p = slots[j]
        if p >= 0:
            V[p] = v
            logN[p] = log_c
    return V, logN


def _sweep_backward(prop, expg, tail_e, emit, codes, slots, n_keep):
    """Backward filter ``w = s * beta`` at the slotted events, plus log-normalisers.

    Mirror image of ``_sweep_forward``: it starts from ``beta(T) = 1`` i.e. ``w(T) = s``,
    walks the events in reverse, and records the state BEFORE applying that event's
    emission -- so the recorded ``w`` is the ``t+`` state and pairs with the forward's
    post-event ``v`` (eq 77 pairs them at the same instant).  Also returns ``w`` at the
    last event, which the path sampler needs to start from.
    """
    Q, s = prop.Q, prop.s
    W = s.new_empty(n_keep, s.shape[0])
    logN = s.new_empty(n_keep)
    w = Q @ (tail_e * (Q.T @ s))                   # over the trailing gap [t_last, T]
    c = w.abs().sum(); w = w / c; log_c = c.log()
    w_last = w
    for j in range(len(codes) - 1, -1, -1):
        p = slots[j]
        if p >= 0:
            W[p] = w
            logN[p] = log_c
        w = w * emit[codes[j]]                     # -> w(t_j^-)
        w = Q @ (expg[j] * (Q.T @ w))              # -> w(t_{j-1}^+)
        c = w.abs().sum(); w = w / c; log_c = log_c + c.log()
    return W, logN, w_last


# ---------------------------------------------------------------------------
# Backward sampling
# ---------------------------------------------------------------------------
def _draw(wts, u):
    """Inverse-CDF draw of one grid index per column of ``wts`` [G, P], given ``u`` [P]."""
    cdf = wts.cumsum(0)
    cdf = cdf / cdf[-1:].clamp_min(1e-300)
    idx = torch.searchsorted(cdf.T.contiguous(), u.unsqueeze(1)).squeeze(1)
    return idx.clamp_max(wts.shape[0] - 1)


def _sample_paths(prop, expg, V_all, gamma_last, U, slots, n_keep):
    """Exact posterior sample paths (FFBS), vectorised over paths.

    Uses ``p(x_j = i | x_{j+1} = m) prop v_j(i) (e^{A g_{j+1}})_{mi}`` (module docstring).
    ``e^{Ag}`` is symmetric, so the required columns are a row-gather of ``Q`` followed by
    one ``[G, P]`` matmul -- one matmul per event for ALL paths, and never a dense kernel.
    """
    Q = prop.Q
    N, P = U.shape
    out = torch.empty(n_keep, P, dtype=torch.long, device=Q.device)
    cur = _draw(gamma_last.unsqueeze(1).expand(-1, P), U[N - 1])
    if slots[N - 1] >= 0:
        out[slots[N - 1]] = cur
    for j in range(N - 2, -1, -1):
        cols = Q @ (expg[j + 1].unsqueeze(1) * Q[cur, :].T)         # [G, P]
        cur = _draw((V_all[j].unsqueeze(1) * cols).clamp_min(0), U[j])
        p = slots[j]
        if p >= 0:
            out[p] = cur
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@torch.no_grad()
def reconstruct_trace(
    times: torch.Tensor,
    colors: torch.Tensor,
    T: float | None,
    potential,
    D: torch.Tensor,
    rates,
    grid: torch.Tensor,
    C: torch.Tensor,
    R0: float,
    *,
    t_out: torch.Tensor | None = None,
    p0: torch.Tensor | None = None,
    jitter: float = 1e-12,
    n_paths: int = 0,
    seed: int = 0,
) -> Reconstruction:
    """Smoothing posterior over ``x(t)`` for one trace (eqs 77-78).

    Data conventions are exactly those of ``forward.marginal_loglik``:
    ``times`` [K] absolute arrival times (ms, non-decreasing, may start at 0), ``colors``
    [K] int64 in {0=green, 1=red}, ``T`` the window (default ``times[-1]``, i.e. condition
    on the first photon and no trailing gap), ``p0`` the initial law (default the
    Boltzmann ``stationary(u)``).

    ``t_out``  : ``None`` -> report at the photon times; else a 1-D tensor of times in
                 [0, T] (a uniform lattice is ``torch.arange(0, T, dt)`` in the caller).
    ``n_paths``: draw this many exact posterior trajectories (0 = none).
    ``seed``   : seeds the sampler's generator (same seed -> identical paths).

    Cost: ``2(K+M)`` matvecs of size ``G^2``, plus ``(K+M)`` matmuls of width ``n_paths``.
    Memory: ``O(N G)`` for the hoisted gap exponentials (``N = K + M`` events) plus
    ``O(M G)`` for the reported states, and a second ``O(N G)`` when sampling (the sampler
    conditions on the forward state at EVERY event, not just the reported ones).
    """
    if times.shape[0] < 1:
        raise ValueError("reconstruct_trace: need at least one photon")
    # Same tiling guards as `marginal_loglik` (SPEC Remark 1): unsorted times or a window
    # shorter than the last photon would mis-tile the survival integral.
    if times.numel() > 1:
        assert bool((times[1:] - times[:-1] >= -1e-9).all()), \
            "reconstruct_trace: `times` must be non-decreasing"
    if T is None:
        T = float(times[-1])
    assert float(T) >= float(times[-1]) - 1e-9, \
        "reconstruct_trace: T must be >= times[-1]"
    if t_out is not None:
        if t_out.numel() == 0:
            raise ValueError("reconstruct_trace: t_out is empty (use None for the "
                             "photon times)")
        assert bool(((t_out >= -1e-9) & (t_out <= float(T) + 1e-9)).all()), \
            "reconstruct_trace: every t_out must lie in [0, T]"

    u_grid = _BasePotential_on_grid(potential, grid)
    dx = float(grid[1] - grid[0]) if grid.shape[0] > 1 else 1.0
    prop = build_propagator_from_u(u_grid, D, rates, grid, C, R0, dx, jitter)

    gaps, tail, codes, keep_idx, t = _events(times, colors, T, t_out)
    N, M = len(codes), len(keep_idx)

    # Hoisted out of both loops and shared with the sampler: one big exp instead of one
    # per step, and a 3-row emission table instead of a python branch per photon.
    expg = torch.exp(gaps.unsqueeze(1) * prop.lam.unsqueeze(0))          # [N, G]
    tail_e = torch.exp(prop.lam * tail)                                  # [G]
    emit = torch.stack([prop.mu_G, prop.mu_R, torch.ones_like(prop.s)])   # [3, G]

    slots = [-1] * N
    for r, j in enumerate(keep_idx):
        slots[j] = r

    if p0 is None:
        p0 = stationary(u_grid)
    v0 = p0 / prop.s

    # The sampler conditions on the forward states at EVERY event; the backward filter is
    # only ever needed at the reported times.
    f_slots, n_f = (list(range(N)), N) if n_paths else (slots, M)
    V_all, logNf_all = _sweep_forward(prop, expg, emit, codes, f_slots, n_f, v0)
    W, logNb, w_last = _sweep_backward(prop, expg, tail_e, emit, codes, slots, M)
    V = V_all[keep_idx] if n_paths else V_all
    logNf = logNf_all[keep_idx] if n_paths else logNf_all

    VW = (V * W).clamp_min(0)                                            # [M, G]
    tot = VW.sum(-1)
    gamma = VW / tot.unsqueeze(-1)
    logZ = tot.clamp_min(1e-300).log() + logNf + logNb                   # == log L, eq 77
    loglik = logZ.mean()

    x_mean = gamma @ grid
    x_sd = (gamma @ grid.square() - x_mean.square()).clamp_min(0).sqrt()

    if n_paths:
        gen = torch.Generator(device=grid.device).manual_seed(int(seed))
        U = torch.rand(N, int(n_paths), generator=gen,
                       device=grid.device, dtype=grid.dtype)
        gamma_last = (V_all[N - 1] * w_last).clamp_min(0)
        idx = _sample_paths(prop, expg, V_all, gamma_last, U, slots, M)
        paths = grid[idx].T.contiguous()                                 # [n_paths, M]
    else:
        paths = grid.new_zeros(0, M)

    return Reconstruction(
        t=t, gamma=gamma, grid=grid, x_mean=x_mean, x_sd=x_sd, paths=paths,
        loglik=float(loglik), loglik_spread=float((logZ - loglik).abs().max()),
    )


def reconstruct_batch(batch, potential, D, rates, grid, C, R0, *, indices=None, **kw):
    """``reconstruct_trace`` over the traces of a padded ``Batch`` -> list, input order.

    ``indices`` selects a subset (default: all traces).  Each trace gets its own
    propagator build, which is negligible: the ``eigh`` is ``O(G^3)`` once against
    ``O(K G^2)`` for the sweeps, i.e. irrelevant beyond ~G photons per trace.

    Traces have different windows ``T``, so a shared ``t_out`` will fail the ``[0, T]``
    guard on the shorter ones -- leave ``t_out=None`` (report at each trace's photon
    times) or pass a per-trace lattice by calling ``reconstruct_trace`` directly.
    """
    out = []
    for b in (range(batch.n_traces) if indices is None else indices):
        L = int(batch.lengths[b])
        times = torch.cumsum(batch.ipt[b, :L], 0)
        out.append(reconstruct_trace(
            times, batch.colors[b, :L], float(times[-1]),
            potential, D, rates, grid, C, R0, **kw,
        ))
    return out
