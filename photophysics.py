"""FRET map and per-channel emission intensities (SPEC section 4.2).

Matches the ``smFRET_sbi`` / ``diff_fret_photon`` forward model exactly so the
likelihood is evaluated with the *same* photon model that generated the data:

    E(x)  = R0^6 / (R0^6 + x^6)
    phi_g = C_gg (1-E) + C_rg E,   phi_r = C_gr (1-E) + C_rr E
    mu_G  = eta_g kD phi_g + beta_g = a_g phi_g + bg_g
    mu_R  = eta_r kD phi_r + beta_r = a_r phi_r + bg_r

matching ``simulator.pyx`` lines 380-381 exactly.  The backgrounds are DETECTED
rates and are added *outside* the detector efficiency, on both sides -- the
simulator takes ``beta_g``/``beta_r`` directly, so no conversion exists anywhere.
The identifiable brightnesses are ``a_g = eta_g kD``, ``a_r = eta_r kD`` and the
backgrounds are ``bg_g = beta_g``, ``bg_r = beta_r``.  The crosstalk matrix ``C``
and ``R0`` are fixed calibration.  These four positive rates (a_g, a_r, bg_g,
bg_r) are the emission parameters exposed to the optimiser; the mapping
``mu = a phi + bg`` is fully general regardless of the eta/kD decomposition.

``eta_c`` and ``kD`` never appear separately -- only the product ``a_c = eta_c kD``
is identifiable, which is why ``from_physics`` takes both but stores one number
per channel.

Everything is vectorised over ``x`` (shape preserving); no per-photon loops.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import DTYPE


@dataclass
class EffectiveRates:
    """Identifiable emission parameters (all positive scalars, kHz).

    a_g, a_r : donor-excitation brightness reaching green / red detectors
               (= eta_g kD, eta_r kD before crosstalk mixing).
    bg_g, bg_r : detected green / red background rates (= the simulator's
               beta_g / beta_r verbatim -- background sits outside the eta
               factor on both sides, see module docstring).
    """

    a_g: torch.Tensor
    a_r: torch.Tensor
    bg_g: torch.Tensor
    bg_r: torch.Tensor

    @staticmethod
    def from_physics(kD, eta_g, eta_r, beta_g, beta_r, device="cpu") -> "EffectiveRates":
        # mu = eta*kD*phi + beta: the background is already a detected rate and
        # passes through untouched (simulator.pyx:380-381).
        f = lambda v: torch.as_tensor(float(v), dtype=DTYPE, device=device)
        return EffectiveRates(
            f(eta_g) * f(kD), f(eta_r) * f(kD),
            f(beta_g), f(beta_r),
        )

    def as_dict(self) -> dict:
        return {
            "a_g": float(self.a_g),
            "a_r": float(self.a_r),
            "bg_g": float(self.bg_g),
            "bg_r": float(self.bg_r),
        }


def fret_efficiency(x: torch.Tensor, R0: float) -> torch.Tensor:
    R0_6 = R0 ** 6
    return R0_6 / (R0_6 + x ** 6)


def channel_fractions(
    x: torch.Tensor, C: torch.Tensor, R0: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crosstalk-mixed source fractions reaching (green, red) detectors.

    f_g = C_gg (1-E) + C_rg E,   f_r = C_gr (1-E) + C_rr E.
    """
    E = fret_efficiency(x, R0)
    C_gg, C_gr = C[0, 0], C[0, 1]
    C_rg, C_rr = C[1, 0], C[1, 1]
    f_g = C_gg * (1.0 - E) + C_rg * E
    f_r = C_gr * (1.0 - E) + C_rr * E
    return f_g, f_r


def emission_rates(
    x: torch.Tensor, rates: EffectiveRates, C: torch.Tensor, R0: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel intensities mu_G(x), mu_R(x) in kHz (shape of ``x``)."""
    f_g, f_r = channel_fractions(x, C, R0)
    mu_G = rates.a_g * f_g + rates.bg_g
    mu_R = rates.a_r * f_r + rates.bg_r
    return mu_G, mu_R
