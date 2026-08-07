"""Light wrapper around the in-project Cython simulator (``simulator.pyx``).

The notebooks and reliability scripts generate photon streams with the
*in-project* ``smFRET_simulator`` (built via ``build_cython.py``), not the old
external ``smFRET_sbi`` adapter.  This module provides the shared pieces:

* ``Batch`` -- padded batch of photon-stream traces (the data contract the
  marked-point-process likelihood consumes);
* ``_stack`` -- list of ``(ipt, colors)`` numpy arrays -> padded ``Batch``;
* ``simulate_equilibrium`` -- a parallel rejection loop over ``smFRET_simulator``
  (the pattern the notebooks previously duplicated inline), collecting usable
  equilibrium traces into a ``Batch``.

The simulator draws its start position from the Boltzmann equilibrium internally
and parametrises ``U(x)`` directly by the potential *value* knots ``y_knots =
U(x_knots)`` -- there is no ``x0`` argument.  Simulation is CPU-only (Cython);
run it BEFORE any CUDA work.
"""

from __future__ import annotations

import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import torch

from .config import DTYPE

try:
    from .simulator import smFRET_simulator
except ImportError:
    # The compiled `diff_fret_likelihood.simulator` extension may not be built
    # (e.g. GSL missing at install time), so importing `diff_fret_likelihood`
    # must not require it -- only `simulate_equilibrium` does, and its worker
    # re-imports it lazily.
    smFRET_simulator = None


# ---------------------------------------------------------------------------
# Padded batch of traces
# ---------------------------------------------------------------------------
@dataclass
class Batch:
    """Padded batch of photon-stream traces (all on ``device``)."""

    ipt: torch.Tensor       # [B, Kmax] inter-photon gaps (ms)
    colors: torch.Tensor    # [B, Kmax] int64 {0,1}
    mask: torch.Tensor      # [B, Kmax] bool
    lengths: torch.Tensor   # [B] int64
    T: torch.Tensor         # [B] window length (ms) = sum(ipt)

    @property
    def n_traces(self) -> int:
        return self.ipt.shape[0]

    def to(self, device):
        return Batch(
            self.ipt.to(device), self.colors.to(device), self.mask.to(device),
            self.lengths.to(device), self.T.to(device),
        )


def _stack(raw, max_photons, device):
    """List of (ipt[np], colors[np]) -> padded Batch."""
    if max_photons is not None:
        raw = [(i[:max_photons], c[:max_photons]) for i, c in raw]
    Kmax = max(len(i) for i in (r[0] for r in raw))
    B = len(raw)
    ipt = torch.zeros(B, Kmax, dtype=DTYPE)
    colors = torch.zeros(B, Kmax, dtype=torch.int64)
    mask = torch.zeros(B, Kmax, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.int64)
    Tvec = torch.zeros(B, dtype=DTYPE)
    for b, (i, c) in enumerate(raw):
        n = len(i)
        ipt[b, :n] = torch.as_tensor(i, dtype=DTYPE)
        colors[b, :n] = torch.as_tensor(c.astype(np.int64))
        mask[b, :n] = True
        lengths[b] = n
        Tvec[b] = float(np.sum(i))
    return Batch(ipt, colors, mask, lengths, Tvec).to(device)


# ---------------------------------------------------------------------------
# Parallel equilibrium simulation (in-project Cython simulator)
# ---------------------------------------------------------------------------
def _simulate_chunk(share, budget, seed, params):
    """Local rejection loop -> up to ``share`` accepted ``(ipt, cols)`` traces.

    The ONLY rejection is the walker leaving the spline domain.  There is deliberately
    no photon-count floor: the count is *determined* by the photophysics parameters, so
    cutting on it would truncate the count distribution while still returning a
    full-count pool -- a silent bias in anything estimated from the traces (the Fisher
    above all).  The simulator sizes its own buffers, so there is no upper cut either.

    Module-level so it is picklable for ``ProcessPoolExecutor``; imports the
    Cython simulator inside the worker (no need to ship the compiled function).
    """

    sim = smFRET_simulator
    if sim is None:                             # .so wasn't importable at package import
        from .simulator import smFRET_simulator as sim

    np.random.seed(seed)                        # independent stream for this worker
    (D, x_knots, y_knots, R0, kD, k_gb, k_rb,
     eta_g, eta_r, C_gg, C_rr, C_gr, C_rg, T, dt) = params

    out, tries, n_empty = [], 0, 0
    while len(out) < share and tries < budget:
        tries += 1
        G, R = sim(D, x_knots, y_knots, R0, kD, k_gb, k_rb,
                   eta_g, eta_r, C_gg, C_rr, C_gr, C_rg, T, dt)
        if G is None:                            # aborted: walker left the domain
            continue
        G = np.asarray(G, float); R = np.asarray(R, float)
        times = np.concatenate([G, R])
        cols = np.concatenate([np.zeros(G.size, int), np.ones(R.size, int)])
        if times.size == 0:
            # Not a count cut: a trace with no photons has no inter-photon times at
            # all, and would enter the batch as an all-masked row that silently
            # contributes nothing to the likelihood.  Counted and reported upstream.
            n_empty += 1
            continue
        o = np.argsort(times, kind="stable")
        times, cols = times[o], cols[o]
        ipt = np.zeros(times.size); ipt[1:] = np.diff(times)                 # ms
        out.append((ipt, cols))
    return out, n_empty


def simulate_equilibrium(
    x_knots, y_knots, D, R0, kD, k_gb, k_rb, eta_g, eta_r, C_gr, C_rg,
    T, dt, *, n_traces, max_tries_factor=4,
    n_workers=None, seed=None, device="cpu", verbose=True,
) -> Batch:
    """Simulate ``n_traces`` equilibrium photon streams into a padded ``Batch``.

    ``x_knots`` / ``y_knots`` are the potential value knots (``y_knots =
    U(x_knots)``); ``D`` is nm^2/ms.  Crosstalk ``C_gg``/``C_rr`` are derived as
    ``1 - C_gr`` / ``1 - C_rg``.  The start position is drawn from the Boltzmann
    equilibrium inside the simulator -- there is no ``x0`` argument.  Runs a
    fork-pool rejection loop (CPU-only; call BEFORE any CUDA work).

    The photon-count distribution is returned untruncated at both ends: the simulator
    sizes its own arrival-time buffers (no upper cut) and there is no count floor (see
    ``_simulate_chunk``).  Traces are lost only to the walker leaving the spline domain,
    and a short pool warns rather than passing silently.
    """
    x_knots = np.asarray(x_knots, dtype=np.float64)
    y_knots = np.asarray(y_knots, dtype=np.float64)
    C_gg, C_rr = 1.0 - C_gr, 1.0 - C_rg

    n_workers = max(1, min(n_workers or os.cpu_count(), n_traces))
    # split target + budget across workers (keep the serial attempts/trace ratio)
    shares = [n_traces // n_workers + (i < n_traces % n_workers) for i in range(n_workers)]
    budgets = [max_tries_factor * s for s in shares]
    # one independent RNG stream per worker
    seeds = [int(s.generate_state(1)[0])
             for s in np.random.SeedSequence(seed).spawn(n_workers)]

    params = (D, x_knots, y_knots, R0, kD, k_gb, k_rb, eta_g, eta_r,
              C_gg, C_rr, C_gr, C_rg, T, dt)

    t0, raw, n_empty = time.perf_counter(), [], 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_simulate_chunk, sh, bd, sd, params)
                for sh, bd, sd in zip(shares, budgets, seeds)]
        for f in as_completed(futs):
            chunk, n_e = f.result()
            raw.extend(chunk)
            n_empty += n_e

    raw = raw[:n_traces]                         # trim overshoot from remainder rounding
    if not raw:
        raise RuntimeError("no usable equilibrium traces produced")
    if len(raw) < n_traces:
        warnings.warn(
            f"simulate_equilibrium: {len(raw)} of {n_traces} traces produced -- the "
            f"rejection budget ({max_tries_factor} attempts per trace) ran out. Traces "
            f"are rejected only for leaving the spline domain [{x_knots[0]:.3g}, "
            f"{x_knots[-1]:.3g}], so widen the knot domain or shorten T. The returned "
            f"pool is smaller than requested, NOT a biased draw of the requested size.",
            RuntimeWarning, stacklevel=2,
        )
    if n_empty:
        warnings.warn(
            f"simulate_equilibrium: dropped {n_empty} trace(s) with zero photons (no "
            f"inter-photon times to represent). The emission rates are low enough that "
            f"empty traces are likely, so the returned count distribution IS truncated "
            f"at zero -- raise kD, the background rates, or T.",
            RuntimeWarning, stacklevel=2,
        )
    batch = _stack(raw, max_photons=None, device=device)
    if verbose:
        pcts = np.percentile([len(i) for i, _ in raw], [0, 50, 100]).astype(int)
        print(f"{len(raw)}/{n_traces} traces in {time.perf_counter()-t0:.1f}s | "
              f"photons/trace min/med/max = {pcts} | workers={n_workers}")
    return batch
