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

import torch

from .config import (
    DTYPE, GridConfig, PotentialConfig, PhysicsConstants, PriorConfig, OptimConfig,
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
from .objective import curvature_penalty, neg_log_posterior, complete_data_loglik
from .dynamics import em_transition_logp
from .infer import fit, FitResult, FreeRates, recovered_potential, posterior_occupancy
from . import simulate
from . import init
from . import sample

# float64 in the likelihood path is non-negotiable (SPEC section 8).
torch.set_default_dtype(DTYPE)

__all__ = [
    "DTYPE", "GridConfig", "PotentialConfig", "PhysicsConstants", "PriorConfig",
    "OptimConfig", "build_potential", "MLPPotential", "SplinePotential",
    "EffectiveRates", "fret_efficiency", "emission_rates", "smoluchowski",
    "symmetrize", "stationary", "assert_generator_valid", "marginal_loglik",
    "marginal_loglik_batch", "reference_loglik", "build_propagator_from_u",
    "curvature_penalty", "neg_log_posterior", "complete_data_loglik",
    "em_transition_logp", "fit", "FitResult", "FreeRates",
    "recovered_potential", "posterior_occupancy", "simulate", "init", "sample",
]
