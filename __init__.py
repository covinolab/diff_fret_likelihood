"""Differentiable smFRET marked-point-process likelihood.

A fully differentiable likelihood for continuous-illumination smFRET photon
streams: it uses inter-photon times via a marked (coloured) inhomogeneous
Poisson observation model and parameterises the free-energy landscape with a
neural network, so the potential shape, the diffusion coefficient ``D`` and the
photophysics can all be estimated by gradient descent.

See ``SPEC`` (module docstrings carry the section references) and the notebook
``diff_fret_likelihood_recovery.ipynb``.
"""

from __future__ import annotations

from ._version import __version__
from .config import (
    DTYPE, GridConfig, PotentialConfig, PhysicsConstants, PriorConfig, OptimConfig,
    use_float64,
)
from .potential import build_potential, MLPPotential, SplinePotential
from .photophysics import EffectiveRates, fret_efficiency, emission_rates
from .generator import (
    smoluchowski, symmetrize, stationary, assert_generator_valid,
)
from .forward import (
    marginal_loglik, marginal_loglik_batch, reference_loglik,
    build_propagator_from_u,
)
from .objective import curvature_penalty, prior_penalty, neg_log_posterior
from .dynamics import em_transition_logp
from .infer import fit, fit_multi, FitResult, FreeRates, recovered_potential
from .fisher import cramer_rao_bound, CRBResult
from .reconstruct import reconstruct_trace, reconstruct_batch, Reconstruction
from . import simulate
from . import init
from . import sample

# NOTE: importing this package no longer mutates global torch state. The
# likelihood is designed for float64 (SPEC section 8); call ``use_float64()`` to
# make it the global default, or pass ``dtype=DTYPE`` explicitly.

__all__ = [
    "__version__", "use_float64",
    "DTYPE", "GridConfig", "PotentialConfig", "PhysicsConstants", "PriorConfig",
    "OptimConfig", "build_potential", "MLPPotential", "SplinePotential",
    "EffectiveRates", "fret_efficiency", "emission_rates", "smoluchowski",
    "symmetrize", "stationary", "assert_generator_valid", "marginal_loglik",
    "marginal_loglik_batch", "reference_loglik", "build_propagator_from_u",
    "curvature_penalty", "prior_penalty", "neg_log_posterior",
    "em_transition_logp", "fit", "fit_multi", "FitResult", "FreeRates",
    "recovered_potential", "cramer_rao_bound", "CRBResult",
    "reconstruct_trace", "reconstruct_batch", "Reconstruction",
    "simulate", "init", "sample",
]
