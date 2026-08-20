"""Seeding.

Positivity transforms and log-space helpers lived here too, but the fit does its
own ``log``/``exp`` inline (``infer.FreeRates``, ``infer.fit``) and nothing ever
imported them; they were removed in 0.2.0 along with ``LOG2PI``, whose only
consumer was the retired ``dynamics`` module.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
