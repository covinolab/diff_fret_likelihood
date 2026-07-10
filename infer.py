"""Gradient-based estimation of ``(theta, D, photophysics)`` (SPEC 7.6).

Unconstrained parameterisation: optimise ``log D`` and (optionally)
``log a_g, log a_r, log bg_g, log bg_r`` freely; map back with ``exp``.  The MLP
potential parameters are optimised directly.  Adam warmup (stable) then LBFGS
with strong-Wolfe line search (smooth MLE endgame).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import DTYPE, OptimConfig, PriorConfig
from .generator import stationary
from .forward import _BasePotential_on_grid, marginal_loglik_batch
from .objective import neg_log_posterior
from .photophysics import EffectiveRates


class FreeRates(torch.nn.Module):
    """Optimisable positive emission rates via ``log``-parameterisation."""

    def __init__(self, init: EffectiveRates):
        super().__init__()
        self.log_a_g = torch.nn.Parameter(torch.log(init.a_g.clone()))
        self.log_a_r = torch.nn.Parameter(torch.log(init.a_r.clone()))
        self.log_bg_g = torch.nn.Parameter(torch.log(init.bg_g.clamp_min(1e-6)))
        self.log_bg_r = torch.nn.Parameter(torch.log(init.bg_r.clamp_min(1e-6)))

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
    final_loss: float
    history: list = field(default_factory=list)
    log_D_param: object = None
    free_rates: object = None


def fit(
    batch,
    grid: torch.Tensor,
    potential,
    C: torch.Tensor,
    R0: float,
    D_init: float,
    rates_init: EffectiveRates,
    prior: PriorConfig,
    optim: OptimConfig | None = None,
    fit_D: bool = True,
    fit_rates: bool = False,
    p0: torch.Tensor | None = None,
    verbose: bool = True,
) -> FitResult:
    """Fit the potential (+ D, + rates) to a ``Batch`` by MAP.

    All machinery (Adam->LBFGS, ``fit_D``, ``fit_rates``, priors) is shared.

    Returns a ``FitResult`` with point estimates, the fitted potential, and a
    per-log-step history.
    """
    optim = optim or OptimConfig()
    device = grid.device
    ipt, colors, mask = batch.ipt, batch.colors, batch.mask

    log_D = torch.tensor(
        float(torch.log(torch.as_tensor(D_init, dtype=DTYPE))),
        dtype=DTYPE, device=device, requires_grad=fit_D,
    )
    free_rates = FreeRates(rates_init).to(device) if fit_rates else None

    def current_rates():
        return free_rates.build() if fit_rates else rates_init

    params = list(potential.parameters())
    if fit_D:
        params.append(log_D)
    if fit_rates:
        params += list(free_rates.parameters())

    history = []

    compile_mode = optim.compile_mode if optim.compile else None

    def closure_value():
        D = log_D.exp() if fit_D else torch.as_tensor(D_init, dtype=DTYPE, device=device)
        return neg_log_posterior(
            ipt, colors, mask, potential, D, current_rates(),
            grid, C, R0, prior, p0=p0, compile_mode=compile_mode,
            propagate_dtype=optim.propagate_dtype,
        )

    def _mark_step():
        # CUDA-graph trees need a step boundary between optimiser iterations so
        # replays don't alias across steps (no-op when not using cudagraphs).
        if optim.compile and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

    # --- Adam warmup ---
    adam = torch.optim.Adam(params, lr=optim.adam_lr)
    for step in range(optim.adam_steps):
        _mark_step()
        adam.zero_grad()
        loss = closure_value()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, optim.grad_clip)
        adam.step()
        if verbose and (step % optim.log_every == 0 or step == optim.adam_steps - 1):
            D_now = float(log_D.exp()) if fit_D else D_init
            history.append({"phase": "adam", "step": step,
                            "loss": float(loss), "D": D_now})
            print(f"  [adam {step:4d}] loss={float(loss):.3f}  D={D_now:.3f}")

    # --- LBFGS polish ---
    if optim.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            params, lr=optim.lbfgs_lr, max_iter=1,          # one LBFGS iter per .step()
            line_search_fn="strong_wolfe", history_size=50,
        )

        def closure():
            _mark_step()
            lbfgs.zero_grad()
            loss = closure_value()
            loss.backward()
            return loss

        for step in range(optim.lbfgs_steps):
            loss = lbfgs.step(closure)
            if verbose and (step % optim.log_every == 0 or step == optim.lbfgs_steps - 1):
                D_now = float(log_D.exp()) if fit_D else D_init
                history.append({"phase": "lbfgs", "step": step,
                                "loss": float(loss), "D": D_now})
                print(f"  [lbfgs {step:4d}] loss={float(loss):.3f}  D={D_now:.3f}")

    with torch.no_grad():
        final_loss = float(closure_value())
        D_final = float(log_D.exp()) if fit_D else float(D_init)
        rates_final = current_rates()

    return FitResult(
        potential=potential, D=D_final,
        rates=EffectiveRates(*(r.detach() for r in
              (rates_final.a_g, rates_final.a_r, rates_final.bg_g, rates_final.bg_r))),
        final_loss=final_loss, history=history,
        log_D_param=log_D if fit_D else None, free_rates=free_rates,
    )


# ---------------------------------------------------------------------------
# Diagnostics used by the notebook
# ---------------------------------------------------------------------------
@torch.no_grad()
def recovered_potential(potential, grid) -> torch.Tensor:
    """Gauge-fixed recovered ``U(x)`` on the grid (min = 0)."""
    return _BasePotential_on_grid(potential, grid)


@torch.no_grad()
def posterior_occupancy(batch, potential, D, rates, grid, C, R0, p0=None):
    """Forward-backward smoothed state occupancy summed over traces ``[G]``.

    A proper latent-state posterior (NOT a naive FRET->x inversion), analogous
    to the reference notebook's ``posterior_occupancy``.  Uses the same
    symmetric-basis propagator; runs a forward and a backward sweep and sums the
    normalised products over photons and traces.
    """
    from .forward import build_propagator_from_u

    dx = float(grid[1] - grid[0])
    u_grid = _BasePotential_on_grid(potential, grid)
    prop = build_propagator_from_u(u_grid, torch.as_tensor(D, dtype=DTYPE, device=grid.device),
                                   rates, grid, C, R0, dx)
    s = prop.s
    G = grid.shape[0]
    if p0 is None:
        p0 = stationary(u_grid)

    occ = torch.zeros(G, dtype=DTYPE, device=grid.device)
    ipt, colors, mask, lengths = batch.ipt, batch.colors, batch.mask, batch.lengths
    for b in range(batch.n_traces):
        n = int(lengths[b])
        if n == 0:
            continue
        gaps = ipt[b, :n]
        cols = colors[b, :n]
        # Symmetric-basis forward-backward. Forward a~_k = V_{c_k} e^{A tau_k} a~_{k-1},
        # a~_0 = p0/s. Backward b~_{k-1} = e^{A tau_k} (V_{c_k} b~_k), seeded b~_{n-1}=s
        # (trailing gap 0). Smoothed occupancy (probability space) = a~_k (elementwise) b~_k
        # -- the sqrt(pi) factors of the two messages already combine, so NO extra *s.
        alphas = []
        v = p0 / s
        v = v / v.abs().sum()
        for k in range(n):
            v = prop.propagate(v, gaps[k])
            emit = prop.mu_G if int(cols[k]) == 0 else prop.mu_R
            v = v * emit
            v = v / v.abs().sum()
            alphas.append(v)
        betas = [None] * n
        beta = s / s.abs().sum()                     # b~_{n-1} = s (symmetric-basis seed)
        betas[n - 1] = beta
        for k in range(n - 1, 0, -1):
            emit = prop.mu_G if int(cols[k]) == 0 else prop.mu_R
            b_em = betas[k] * emit                   # V_{c_k} b~_k
            b_prop = prop.propagate(b_em, gaps[k])   # e^{A tau_k} (.) ; A symmetric => self-adjoint
            b_prop = b_prop / b_prop.abs().sum()
            betas[k - 1] = b_prop
        for k in range(n):
            g = alphas[k] * betas[k]                 # a~_k * b~_k  == probability-space occupancy
            g = g.clamp_min(0)
            tot = g.sum()
            if tot > 0:
                occ += g / tot
    return occ
