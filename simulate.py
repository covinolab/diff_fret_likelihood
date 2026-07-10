"""Thin adapter around the ``smFRET_sbi`` Cython simulator (SPEC 7.7 / 13.6).

We do NOT reimplement a simulator.  ``smFRET_sbi`` emits real photon streams
(inter-photon times in ms + channels 0=green/1=red) for both ``equilibrium``
and ``binding`` modes -- exactly the data contract the marked-point-process
likelihood consumes.  This module:

* puts ``smFRET_sbi/src`` first on ``sys.path`` (its ``simulator.py`` must win
  the name clash with ``backup_fret_sbi/simulator.py``);
* assembles the canonical parameter vector
  ``[logD, g_0..g_{K-1}, R0, kD, k_gb, k_rb, eta_g, eta_r, C_gr, C_rg]``;
* runs the requested mode, collecting usable traces into padded batch tensors;
* exposes the ground-truth landscape ``U(x)`` (via the simulator's own
  force-knot integration) and the ground-truth ``EffectiveRates`` / crosstalk,
  so recovery can be scored against the truth.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import OmegaConf

from .config import DTYPE, PhysicsConstants
from .photophysics import EffectiveRates

SMFRET_SRC = "/home/dingeldein/Desktop/fret_sbi/smFRET_sbi/src"


def _ensure_path():
    if sys.path[:1] != [SMFRET_SRC]:
        if SMFRET_SRC in sys.path:
            sys.path.remove(SMFRET_SRC)
        sys.path.insert(0, SMFRET_SRC)


# ---------------------------------------------------------------------------
# Canonical parameter vector
# ---------------------------------------------------------------------------
def assemble_theta(
    knots, logD, R0, kD, k_gb, k_rb, eta_g, eta_r, C_gr, C_rg
) -> np.ndarray:
    """Build the length-``(num_knots+9)`` canonical theta (numpy float64)."""
    knots = np.asarray(knots, dtype=np.float64)
    K = knots.size
    theta = np.zeros(K + 9, dtype=np.float64)
    theta[0] = logD
    theta[1 : K + 1] = knots
    theta[K + 1] = R0
    theta[K + 2] = kD
    theta[K + 3] = k_gb
    theta[K + 4] = k_rb
    theta[K + 5] = eta_g
    theta[K + 6] = eta_r
    theta[K + 7] = C_gr
    theta[K + 8] = C_rg
    return theta


def physics_from_theta(theta, num_knots) -> dict:
    K = num_knots
    return dict(
        logD=float(theta[0]),
        D=float(10.0 ** theta[0]),
        R0=float(theta[K + 1]),
        kD=float(theta[K + 2]),
        k_gb=float(theta[K + 3]),
        k_rb=float(theta[K + 4]),
        eta_g=float(theta[K + 5]),
        eta_r=float(theta[K + 6]),
        C_gr=float(theta[K + 7]),
        C_rg=float(theta[K + 8]),
    )


def constants_from_theta(theta, num_knots, device="cpu") -> tuple[PhysicsConstants, EffectiveRates]:
    p = physics_from_theta(theta, num_knots)
    C_gr, C_rg = p["C_gr"], p["C_rg"]
    consts = PhysicsConstants(
        R0=p["R0"], C_gg=1.0 - C_gr, C_gr=C_gr, C_rg=C_rg, C_rr=1.0 - C_rg
    )
    rates = EffectiveRates.from_physics(
        p["kD"], p["eta_g"], p["eta_r"], p["k_gb"], p["k_rb"], device=device
    )
    return consts, rates


# ---------------------------------------------------------------------------
# Simulation
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


def simulate_traces(
    cfg,
    theta,
    mode: str,
    n_traces: int = 40,
    min_photons: int = 50,
    max_photons: int | None = 1500,
    max_tries_factor: int = 8,
    device="cpu",
    verbose: bool = True,
):
    """Run the simulator and return ``(Batch, meta)``.

    ``cfg``   : path to a ``conf/simulator/*.yaml`` or an OmegaConf object.
    ``theta`` : canonical parameter vector (numpy or 1-D tensor).
    ``mode``  : 'equilibrium' or 'binding'.
    Simulation is CPU-only (Cython); run it BEFORE any CUDA work.
    """
    import time as _time

    _ensure_path()
    from simulator import (
        smFRET_simulator_wrapper_from_config,
        binding_simulator_wrapper_from_config,
    )

    cfg = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg
    assert "smFRET_sbi" in sys.modules["simulator"].__file__, "wrong simulator.py on path"

    if mode == "equilibrium":
        sim = smFRET_simulator_wrapper_from_config(cfg)
    elif mode == "binding":
        sim = binding_simulator_wrapper_from_config(cfg)
    else:
        raise ValueError(f"mode must be 'equilibrium' or 'binding', got {mode!r}")

    theta_t = torch.as_tensor(np.asarray(theta, dtype=np.float64))
    raw, tries = [], 0
    t0 = _time.perf_counter()
    while len(raw) < n_traces and tries < max_tries_factor * n_traces:
        tries += 1
        out = sim(theta_t)
        if out is None:
            continue
        ipt, chans, n = out
        n = int(n)
        if n < min_photons:
            continue
        raw.append((np.asarray(ipt, np.float64)[:n], np.asarray(chans)[:n]))
    dt = _time.perf_counter() - t0
    if verbose:
        print(f"[{mode}] {len(raw)}/{n_traces} usable traces from {tries} tries "
              f"in {dt:.1f}s")
    if not raw:
        raise RuntimeError(f"no usable {mode} traces produced")

    batch = _stack(raw, max_photons, device)
    meta = {
        "mode": mode,
        "tries": tries,
        "seconds": dt,
        "photon_counts": [len(i) for i, _ in raw],
    }
    return batch, meta


# ---------------------------------------------------------------------------
# Ground truth landscape
# ---------------------------------------------------------------------------
def ground_truth_landscape(cfg, theta, x_eval: np.ndarray) -> np.ndarray:
    """True ``U(x)`` (min-zero) via the simulator's own force-knot integration."""
    _ensure_path()
    from utils import eval_free_energy_from_config

    cfg = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg
    fe = eval_free_energy_from_config(cfg)
    _, U = fe(np.asarray(theta, dtype=np.float64), np.asarray(x_eval, dtype=np.float64))
    return U - U.min()
