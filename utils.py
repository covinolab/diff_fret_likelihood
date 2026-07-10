"""Small numerical helpers: seeding, positivity transforms, log-space ops.

Positivity transforms live here so the optimiser can work in an unconstrained
space (SPEC section 7.6): optimise ``log D``, ``log rate`` freely and map back
with ``exp``.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch

from .config import DTYPE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- positivity transforms (unconstrained <-> positive) ---------------------
def softplus(raw: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(raw)


def inv_softplus(value: float | torch.Tensor) -> torch.Tensor:
    v = torch.as_tensor(value, dtype=DTYPE)
    # numerically-stable inverse of softplus: log(exp(v)-1)
    return v + torch.log(-torch.expm1(-v))


def to_log(value: float | torch.Tensor) -> torch.Tensor:
    return torch.log(torch.as_tensor(value, dtype=DTYPE))


# --- log-space contraction helper -------------------------------------------
def log_dot(log_vec: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """log(sum_i exp(log_vec_i) * vec_i) for vec>=0, done stably."""
    m = log_vec.max()
    return m + torch.log((log_vec - m).exp() @ vec)


def as_tensor(x, device="cpu") -> torch.Tensor:
    """Coerce to a float64 tensor on ``device`` (no copy if already right)."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=DTYPE, device=device)
    return torch.as_tensor(x, dtype=DTYPE, device=device)


LOG2PI = math.log(2.0 * math.pi)
