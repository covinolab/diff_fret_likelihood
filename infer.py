"""Gradient-based estimation of ``(theta, D, photophysics)`` (SPEC 7.6).

Unconstrained parameterisation: optimise ``log D`` and (optionally)
``log a_g, log a_r, log bg_g, log bg_r`` freely; map back with ``exp``.  The MLP
potential parameters are optimised directly.  Adam warmup (stable) then LBFGS
with strong-Wolfe line search (smooth MLE endgame).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import DTYPE, OptimConfig, PriorConfig
from .objective import neg_log_posterior, gauge_penalty, prior_penalty
from .forward import marginal_loglik_batch
from .photophysics import EffectiveRates


class FreeRates(torch.nn.Module):
    """Optimisable positive emission rates via ``log``-parameterisation.

    ``fit_bg=False`` freezes the two background rates at their ``init`` values (they
    become buffers, not parameters) while the brightnesses ``a_g``/``a_r`` stay free.
    Because the frozen pair is registered as buffers, ``parameters()`` -- which the
    optimiser and the homotopy blur list are both built from -- excludes them
    automatically, and ``.to(device)`` still moves them.
    """

    def __init__(self, init: EffectiveRates, fit_bg: bool = True):
        super().__init__()
        self.log_a_g = torch.nn.Parameter(torch.log(init.a_g.clone()))
        self.log_a_r = torch.nn.Parameter(torch.log(init.a_r.clone()))
        log_bg_g = torch.log(init.bg_g.clamp_min(1e-6)).clone()
        log_bg_r = torch.log(init.bg_r.clamp_min(1e-6)).clone()
        if fit_bg:
            self.log_bg_g = torch.nn.Parameter(log_bg_g)
            self.log_bg_r = torch.nn.Parameter(log_bg_r)
        else:
            self.register_buffer("log_bg_g", log_bg_g)
            self.register_buffer("log_bg_r", log_bg_r)

    def build(self) -> EffectiveRates:
        return EffectiveRates(
            self.log_a_g.exp(), self.log_a_r.exp(),
            self.log_bg_g.exp(), self.log_bg_r.exp(),
        )


@dataclass
class FitResult:
    potential: object
    D: float
    rates: EffectiveRates
    best_loss: float
    history: list = field(default_factory=list)
    log_D_param: object = None
    free_rates: object = None


def _sigma_schedule(n, steps, n_polish, sigma0, noise_tau, shape="exp"):
    """Annealed noise scale at step n; sigma=0 for the last n_polish (polish) steps."""
    if n >= steps - n_polish:
        return 0.0
    if shape == "exp":
        return sigma0 * float(np.exp(-n / noise_tau))
    n_anneal = max(1, steps - n_polish)
    frac = n / n_anneal
    if shape == "linear":
        return sigma0 * max(0.0, 1.0 - frac)
    if shape == "cosine":
        return sigma0 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, frac)))
    if shape == "sqrt":
        return sigma0 * max(0.0, 1.0 - math.sqrt(min(1.0, frac)))
    if shape == "const":
        return sigma0
    raise ValueError(f"unknown noise shape {shape!r}")


def fit(
    batch,
    grid: torch.Tensor,
    potential,
    C: torch.Tensor,
    R0: float,
    D_init: float,
    rates_init: EffectiveRates,
    prior: PriorConfig | None,
    optim: OptimConfig | None = None,
    fit_D: bool = True,
    fit_rates: bool = False,
    fit_bg: bool = True,
    verbose: bool = True,
    gauge_sd: float = 1.0,
    # ---- Gaussian-homotopy knobs (graduated non-convexity; ON by default) ----
    sigma0: float = 1.0,
    noise_tau: float | None = None,
    polish_frac: float = 0.15,
    blur: str = "all",            # "all" -> U+D+rates ; "ud" -> U+D only ; "none" -> plain Adam
    noise_shape: str = "exp",
    seed: int = 0,
) -> FitResult:
    """MAP fit by Gaussian homotopy (graduated non-convexity).

    Identical interface/return to the plain-Adam ``fit``, but the Adam warmup runs a
    homotopy schedule: annealed gradient noise (sigma0 -> 0) on the ``blur`` target,
    with a noise-free polish tail (last ``polish_frac`` of ``optim.adam_steps``).  This
    escapes the high-barrier basin trap that plain Adam collapses into.  Set
    ``blur="none"`` (or ``sigma0=0``) to recover the original plain-Adam behaviour.

    ``fit_bg=False`` (only meaningful with ``fit_rates=True``) holds ``bg_g``/``bg_r``
    at their ``rates_init`` values while ``a_g``/``a_r`` stay free -- for calibrating
    out a known background instead of inferring it jointly.
    """
    optim = optim or OptimConfig()
    device = grid.device
    ipt, colors, mask = batch.ipt, batch.colors, batch.mask
    gen = torch.Generator(device=device).manual_seed(int(seed))

    log_D = torch.tensor(
        float(torch.log(torch.as_tensor(D_init, dtype=DTYPE))),
        dtype=DTYPE, device=device, requires_grad=fit_D,
    )
    free_rates = FreeRates(rates_init, fit_bg=fit_bg).to(device) if fit_rates else None

    def current_rates():
        return free_rates.build() if fit_rates else rates_init

    params = list(potential.parameters())
    if fit_D:
        params.append(log_D)
    if fit_rates:
        params += list(free_rates.parameters())

    # which of the optimised params receive the annealed noise
    if blur == "all":
        blurred = (list(potential.parameters())
                   + ([log_D] if fit_D else [])
                   + (list(free_rates.parameters()) if fit_rates else []))
    elif blur == "ud":
        blurred = list(potential.parameters()) + ([log_D] if fit_D else [])
    elif blur == "none":
        blurred = []
    else:
        raise ValueError(f"unknown blur {blur!r}")

    steps = optim.adam_steps
    n_polish = int(round(polish_frac * steps))
    if noise_tau is None:
        noise_tau = max(1.0, steps / 4.0)

    history = []
    compile_mode = optim.compile_mode if optim.compile else None

    def closure_value():
        D = log_D.exp() if fit_D else torch.as_tensor(D_init, dtype=DTYPE, device=device)
        nlp = neg_log_posterior(
            ipt, colors, mask, potential, D, current_rates(),
            grid, C, R0, prior, p0=None, compile_mode=compile_mode,
            propagate_dtype=optim.propagate_dtype,
        )
        return nlp + gauge_penalty(potential, grid, gauge_sd)

    def _mark_step():
        if optim.compile and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

    adam = torch.optim.Adam(params, lr=optim.adam_lr)
    best_loss = float("inf")
    best_state = None

    for step in range(steps):
        _mark_step()
        adam.zero_grad()
        loss = closure_value()
        if not torch.isfinite(loss):
            break
        loss.backward()

        # --- graduated non-convexity: annealed gradient noise on the blur target ---
        sigma_n = _sigma_schedule(step, steps, n_polish, sigma0, noise_tau, noise_shape)
        if sigma_n > 0 and blurred:
            with torch.no_grad():
                for p in blurred:
                    if p.grad is None:
                        continue
                    scale = p.grad.abs().mean() + 1e-12
                    p.grad.add_(sigma_n * scale * torch.randn(
                        p.grad.shape, generator=gen, device=device, dtype=p.grad.dtype))

        torch.nn.utils.clip_grad_norm_(params, optim.grad_clip)
        adam.step()

        if loss < best_loss:
            best_loss = loss
            best_state = [p.detach().clone() for p in params]

        if verbose and (step % optim.log_every == 0 or step == steps - 1):
            D_now = float(log_D.exp()) if fit_D else float(D_init)
            history.append({"phase": "homotopy", "step": step, "sigma": sigma_n,
                            "loss": float(loss), "D": D_now})
            print(f"  [gh {step:4d}] sigma={sigma_n:.3f} loss={float(loss):.3f}  "
                  f"D={D_now:.3f} best_loss={float(best_loss):.3f}")

    if best_state is not None:
        with torch.no_grad():
            for p, p_best in zip(params, best_state):
                p.copy_(p_best)

    with torch.no_grad():
        best_loss = float(best_loss)
        D_final = float(log_D.exp()) if fit_D else float(D_init)
        rates_final = current_rates()

    return FitResult(
        potential=potential,
        D=D_final,
        rates=EffectiveRates(*(r.detach() for r in (rates_final.a_g, rates_final.a_r, rates_final.bg_g, rates_final.bg_r))),
        best_loss=best_loss,
        history=history,
        log_D_param=log_D if fit_D else None,
        free_rates=free_rates,
    )


def fit_multi(
    batches,
    grid: torch.Tensor,
    potential,
    C_list,
    R0_list,
    D_init: float,
    rates_init_list,
    prior: PriorConfig | None,
    optim: OptimConfig | None = None,
    fit_D: bool = True,
    fit_rates: bool = True,
    verbose: bool = True,
    gauge_sd: float = 1.0,
    # ---- Gaussian-homotopy knobs (identical semantics to ``fit``) ----
    sigma0: float = 1.0,
    noise_tau: float | None = None,
    polish_frac: float = 0.15,
    blur: str = "all",
    noise_shape: str = "exp",
    seed: int = 0,
) -> FitResult:
    """Joint MAP fit of ONE shared ``(U, D)`` across several datasets.

    The datasets share the free-energy landscape ``potential`` and the diffusion
    coefficient ``D`` but each carries its own *fixed* calibration ``(C_d, R0_d)`` and
    its own *free* emission rates ``(a_g, a_r, bg_g, bg_r)_d``.  This is the estimator
    for the same molecule acquired under different photophysics conditions (e.g. varying
    ``kD`` / Förster radius): the correct thing is to maximise the *joint* likelihood
    with ``(U, D)`` tied, not to fit each dataset separately and average.

    Because independent datasets have additive log-likelihoods, the objective is

        loss = - sum_d loglik_d(U, D, rates_d ; R0_d, C_d)  +  prior(U, D)  +  gauge(U)

    with the prior and gauge anchor added exactly ONCE on the shared ``(U, D)`` (looping
    ``neg_log_posterior`` per dataset would triple-count the smoothness / ``logD``
    priors).  The datasets *cannot* be pooled into one padded batch: each ``(kD, R0)``
    builds a different propagator, so it is one ``marginal_loglik_batch`` call per
    dataset, summed.

    Interface mirrors :func:`fit` -- ``batch``/``C``/``R0``/``rates_init`` are simply
    pluralised into equal-length lists; everything shared (``grid``, ``potential``,
    ``D_init``, ``prior``, optimiser knobs) stays scalar.  ``fit_rates`` defaults to
    ``True`` here (it is ``False`` in :func:`fit`): per-dataset photophysics is the whole
    reason to call this.  With a single-element list ``fit_multi`` is bit-identical to
    :func:`fit` (same parameter order, same RNG draw).

    Returns a :class:`FitResult` whose ``rates`` is a ``list[EffectiveRates]`` (one per
    dataset, input order) and whose ``free_rates`` is a ``list[FreeRates]`` (or ``None``
    when ``fit_rates=False``) -- the only fields that go plural.
    """
    n = len(batches)
    if not (len(C_list) == len(R0_list) == len(rates_init_list) == n):
        raise ValueError(
            "fit_multi: batches, C_list, R0_list and rates_init_list must have the "
            f"same length; got {n}, {len(C_list)}, {len(R0_list)}, "
            f"{len(rates_init_list)}."
        )
    if n == 0:
        raise ValueError("fit_multi: need at least one dataset.")

    optim = optim or OptimConfig()
    device = grid.device
    batches = [b.to(device) for b in batches]
    gen = torch.Generator(device=device).manual_seed(int(seed))

    log_D = torch.tensor(
        float(torch.log(torch.as_tensor(D_init, dtype=DTYPE))),
        dtype=DTYPE, device=device, requires_grad=fit_D,
    )
    free_rates_list = (
        [FreeRates(r).to(device) for r in rates_init_list] if fit_rates else None
    )

    def current_rates(i):
        return free_rates_list[i].build() if fit_rates else rates_init_list[i]

    params = list(potential.parameters())
    if fit_D:
        params.append(log_D)
    if fit_rates:
        for fr in free_rates_list:
            params += list(fr.parameters())

    # which optimised params receive the annealed noise (same rule as ``fit``,
    # extended over every dataset's rate block)
    if blur == "all":
        blurred = list(potential.parameters()) + ([log_D] if fit_D else [])
        if fit_rates:
            for fr in free_rates_list:
                blurred += list(fr.parameters())
    elif blur == "ud":
        blurred = list(potential.parameters()) + ([log_D] if fit_D else [])
    elif blur == "none":
        blurred = []
    else:
        raise ValueError(f"unknown blur {blur!r}")

    steps = optim.adam_steps
    n_polish = int(round(polish_frac * steps))
    if noise_tau is None:
        noise_tau = max(1.0, steps / 4.0)

    history = []
    compile_mode = optim.compile_mode if optim.compile else None

    def closure_value():
        D = log_D.exp() if fit_D else torch.as_tensor(D_init, dtype=DTYPE, device=device)
        ll = None
        for i, b in enumerate(batches):
            ll_i = marginal_loglik_batch(
                b.ipt, b.colors, b.mask, potential, D, current_rates(i),
                grid, C_list[i], R0_list[i], p0=None, compile_mode=compile_mode,
                propagate_dtype=optim.propagate_dtype,
            )
            ll = ll_i if ll is None else ll + ll_i
        return -ll + prior_penalty(potential, D, grid, prior) \
            + gauge_penalty(potential, grid, gauge_sd)

    def _mark_step():
        if optim.compile and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

    adam = torch.optim.Adam(params, lr=optim.adam_lr)
    best_loss = float("inf")
    best_state = None

    for step in range(steps):
        _mark_step()
        adam.zero_grad()
        loss = closure_value()
        if not torch.isfinite(loss):
            break
        loss.backward()

        # graduated non-convexity: annealed gradient noise on the blur target
        sigma_n = _sigma_schedule(step, steps, n_polish, sigma0, noise_tau, noise_shape)
        if sigma_n > 0 and blurred:
            with torch.no_grad():
                for p in blurred:
                    if p.grad is None:
                        continue
                    scale = p.grad.abs().mean() + 1e-12
                    p.grad.add_(sigma_n * scale * torch.randn(
                        p.grad.shape, generator=gen, device=device, dtype=p.grad.dtype))

        torch.nn.utils.clip_grad_norm_(params, optim.grad_clip)
        adam.step()

        if loss < best_loss:
            best_loss = loss
            best_state = [p.detach().clone() for p in params]

        if verbose and (step % optim.log_every == 0 or step == steps - 1):
            D_now = float(log_D.exp()) if fit_D else float(D_init)
            history.append({"phase": "homotopy", "step": step, "sigma": sigma_n,
                            "loss": float(loss), "D": D_now})
            print(f"  [gh {step:4d}] sigma={sigma_n:.3f} loss={float(loss):.3f}  "
                  f"D={D_now:.3f} best_loss={float(best_loss):.3f}")

    if best_state is not None:
        with torch.no_grad():
            for p, p_best in zip(params, best_state):
                p.copy_(p_best)

    with torch.no_grad():
        best_loss = float(best_loss)
        D_final = float(log_D.exp()) if fit_D else float(D_init)
        rates_final = [current_rates(i) for i in range(n)]
        rates_out = [
            EffectiveRates(*(x.detach() for x in (r.a_g, r.a_r, r.bg_g, r.bg_r)))
            for r in rates_final
        ]

    return FitResult(
        potential=potential,
        D=D_final,
        rates=rates_out,
        best_loss=best_loss,
        history=history,
        log_D_param=log_D if fit_D else None,
        free_rates=free_rates_list,
    )


@torch.no_grad()
def recovered_potential(potential, grid) -> torch.Tensor:
    """Gauge-fixed recovered ``U(x)`` on the grid (grid-mean = 0).

    The single reporting entry point: all downstream consumers (plots, RMSE harness,
    figure scripts) should call this so every landscape is in one convention.  This is
    the grid-mean-zero gauge -- uniform for spline and MLP alike and robust to compare.
    It differs from the fit/CRB enforcement gauge (``mean(theta)=0``) only by a
    constant.  (The min-subtraction in ``_BasePotential_on_grid`` is an *internal*
    exp-overflow safeguard for the likelihood, deliberately not used for reporting.)
    """
    u = potential.on_grid(grid)
    return u - u.mean()




