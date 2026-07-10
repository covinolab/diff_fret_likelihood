"""Euler-Maruyama transition log-density for the joint objective (SPEC 4.4).

    p(x_{m+1} | x_m) = N( x_{m+1}; x_m - D u'(x_m) h, 2 D h ).

Used only by the secondary complete-data objective (``objective.py``); the
primary marginal likelihood never instantiates a path.
"""

from __future__ import annotations

import torch

from .utils import LOG2PI


def em_transition_logp(x_path: torch.Tensor, D: torch.Tensor, potential, h: float) -> torch.Tensor:
    """Per-step log-density ``[M]`` for a path ``x_path`` ``[M+1]`` (step ``h``)."""
    x_prev = x_path[:-1]
    x_next = x_path[1:]
    force = potential.force(x_prev)                 # -u'(x_prev)
    mean = x_prev + D * force * h                   # drift = -D u' h
    var = 2.0 * D * h
    return -0.5 * ((x_next - mean) ** 2 / var + torch.log(var) + LOG2PI)
