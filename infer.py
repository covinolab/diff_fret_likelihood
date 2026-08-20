"""Gradient-based estimation of ``(theta, D, photophysics)`` (SPEC 7.6).

Unconstrained parameterisation: optimise ``log D`` and (optionally)
``log a_g, log a_r, log bg_g, log bg_r`` freely; map back with ``exp``.  The spline
knot heights are optimised directly.  Optimiser: guarded LBFGS with NO
line search (strong-Wolfe returns a zero step on this objective and strands the
fit ~8 nats short) -- one quasi-Newton step per outer iteration (``max_iter=1``;
the curvature history persists across calls), so a non-finite guard with
best-snapshot restore and a plateau stop act between every step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .config import DTYPE, OptimConfig, PriorConfig
from .objective import neg_log_posterior, gauge_penalty, prior_penalty, bg_penalty
from .forward import marginal_loglik_batch
from .photophysics import EffectiveRates


class FreeRates(torch.nn.Module):
    """Optimisable positive emission rates via ``log``-parameterisation.

    ``fit_bg=False`` freezes the two background rates at their ``init`` values (they
    become buffers, not parameters) while the brightnesses ``a_g``/``a_r`` stay free.
    Because the frozen pair is registered as buffers, ``parameters()`` -- which the
    optimiser param list is built from -- excludes them automatically, and
    ``.to(device)`` still moves them.
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
    stop_reason: str = ""
    log_D_param: object = None
    free_rates: object = None


def _lbfgs_fit(params, closure_value, optim, verbose, d_of):
    """Guarded LBFGS driver shared by ``fit``/``fit_multi``.

    ``max_iter=1`` inside the Python loop: LBFGS keeps its curvature history across
    ``step()`` calls (numerically equivalent to a single long call), while the
    non-finite guard, the best snapshot and the plateau stop act between every
    quasi-Newton step.  NEVER pass a line search: strong-Wolfe returns a zero step
    on this objective and strands the fit ~8 nats short.  No gradient clipping
    either -- clipping inside the closure would corrupt the curvature pairs.

    Returns ``(best_loss, history, stop_reason)``.  On EVERY exit path the params
    hold the best-observed state (the untouched entry params if the very first
    loss is already non-finite).  ``d_of`` is a zero-arg callable returning the
    current D as float (logging only).
    """
    opt = torch.optim.LBFGS(params, lr=optim.lbfgs_lr, max_iter=1,
                            history_size=optim.history_size, line_search_fn=None)

    def closure():
        if optim.compile and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        opt.zero_grad()
        loss = closure_value()
        loss.backward()
        return loss

    best_loss, best_state, best_step = float("inf"), None, -1
    last_reset = -1     # patience clock; only > stop_min_delta improvements reset it
    history, stop_reason, step = [], "max_steps", -1
    for step in range(optim.steps):
        # INVARIANT: best_loss is the objective evaluated AT best_state.  With
        # max_iter=1, opt.step returns the loss at the params it was entered with
        # and leaves them one quasi-Newton step further -- so snapshot first.
        prev = [p.detach().clone() for p in params]
        D_at_loss = d_of()           # history pairs (loss, D) at the SAME state
        loss = float(opt.step(closure))

        # Bookkeep BEFORE the guard: a finite loss was measured at `prev`, which
        # predates whatever the step just did to the params, so it is safe to
        # keep even when that step exploded.  Any gain updates the snapshot; only
        # a gain > stop_min_delta extends the run (otherwise sub-threshold
        # dribble keeps the fit alive forever).
        if math.isfinite(loss) and loss < best_loss:
            if loss < best_loss - optim.stop_min_delta:
                last_reset = step
            best_loss, best_state, best_step = loss, prev, step

        # GUARD: fixed-step LBFGS has no descent guarantee; observed divergences
        # are late and not bit-reproducible, so check every step.
        if not (math.isfinite(loss)
                and all(bool(torch.isfinite(p).all()) for p in params)):
            if best_state is None:   # first loss already non-finite: keep the
                best_state = prev    # entry (init) params, not the blown-up ones
            stop_reason = f"recovered@{best_step}"
            break

        if step - last_reset >= optim.stop_patience:
            stop_reason = f"plateau@{step}"
            break

        if all(torch.equal(p, q) for p, q in zip(params, prev)):
            # LBFGS hit an internal tolerance and stopped moving the params; every
            # further step() would re-evaluate the objective for nothing.
            stop_reason = f"converged@{step}"
            break

        if step % optim.log_every == 0:
            history.append({"step": step, "loss": loss, "D": D_at_loss})
            if verbose:
                print(f"  [lbfgs {step:4d}] loss={loss:.3f}  D={D_at_loss:.3f}  "
                      f"best={best_loss:.3f}")

    if best_state is not None:
        with torch.no_grad():
            for p, p_best in zip(params, best_state):
                p.copy_(p_best)

    # final entry describes the RETURNED (restored-best) state.
    history.append({"step": step, "loss": best_loss, "D": d_of()})
    if verbose:
        print(f"  [lbfgs] stop: {stop_reason}  best={best_loss:.3f}  D={d_of():.3f}")
    return best_loss, history, stop_reason


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
) -> FitResult:
    """MAP fit by guarded LBFGS (no line search).

    Each step is one quasi-Newton step (see :func:`_lbfgs_fit`), with two
    safeguards acting between steps:

    - **guard**: a best-loss snapshot is kept and restored on any non-finite loss
      or parameter;
    - **plateau stop**: the fit ends once the best loss has not improved by more
      than ``optim.stop_min_delta`` nats within ``optim.stop_patience`` steps
      (smaller gains still update the snapshot but do not extend the run).

    ``FitResult.stop_reason`` records the exit path: ``"plateau@{step}"``,
    ``"recovered@{best_step}"`` (non-finite hit; best snapshot restored --
    ``@-1`` means the very first loss was non-finite and the init was kept),
    ``"converged@{step}"`` (LBFGS hit an internal tolerance and stopped moving)
    or ``"max_steps"`` (``optim.steps`` reached).

    ``fit_bg=False`` (only meaningful with ``fit_rates=True``) holds ``bg_g``/``bg_r``
    at their ``rates_init`` values while ``a_g``/``a_r`` stay free -- for calibrating
    out a known background instead of inferring it jointly.
    """
    optim = optim or OptimConfig()
    device = grid.device
    ipt, colors, mask = batch.ipt, batch.colors, batch.mask

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

    compile_mode = optim.compile_mode if optim.compile else None

    def closure_value():
        D = log_D.exp() if fit_D else torch.as_tensor(D_init, dtype=DTYPE, device=device)
        nlp = neg_log_posterior(
            ipt, colors, mask, potential, D, current_rates(),
            grid, C, R0, prior, p0=None, compile_mode=compile_mode,
            propagate_dtype=optim.propagate_dtype,
        )
        return nlp + gauge_penalty(potential, grid, gauge_sd)

    d_of = (lambda: float(log_D.exp())) if fit_D else (lambda: float(D_init))
    best_loss, history, stop_reason = _lbfgs_fit(
        params, closure_value, optim, verbose, d_of)

    with torch.no_grad():
        D_final = float(log_D.exp()) if fit_D else float(D_init)
        rates_final = current_rates()

    return FitResult(
        potential=potential,
        D=D_final,
        rates=EffectiveRates(*(r.detach() for r in (rates_final.a_g, rates_final.a_r, rates_final.bg_g, rates_final.bg_r))),
        best_loss=best_loss,
        history=history,
        stop_reason=stop_reason,
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
    ``D_init``, ``prior``, optimiser knobs) stays scalar.  The optimiser is the same
    guarded LBFGS (see :func:`fit` for the safeguards and ``stop_reason`` values).
    ``fit_rates`` defaults to ``True`` here (it is ``False`` in :func:`fit`):
    per-dataset photophysics is the whole reason to call this.  With a single-element
    list ``fit_multi`` is bit-identical to :func:`fit` (same parameter order; the
    guarded LBFGS loop is deterministic).

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
        # The shared priors (curvature / logD / GP / l2) are applied ONCE -- see the
        # note above about not triple-counting them.  The background prior is NOT
        # shared: every dataset carries its own bg_g/bg_r, so it gets its own term.
        reg = prior_penalty(potential, D, grid, prior)
        if prior is not None and (prior.bg_g_mean is not None
                                  or prior.bg_r_mean is not None):
            for i in range(len(batches)):
                reg = reg + bg_penalty(current_rates(i), prior)
        return -ll + reg + gauge_penalty(potential, grid, gauge_sd)

    d_of = (lambda: float(log_D.exp())) if fit_D else (lambda: float(D_init))
    best_loss, history, stop_reason = _lbfgs_fit(
        params, closure_value, optim, verbose, d_of)

    with torch.no_grad():
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
        stop_reason=stop_reason,
        log_D_param=log_D if fit_D else None,
        free_rates=free_rates_list,
    )


@torch.no_grad()
def recovered_potential(potential, grid) -> torch.Tensor:
    """Gauge-fixed recovered ``U(x)`` on the grid (grid-mean = 0).

    The single reporting entry point: all downstream consumers (plots, RMSE harness,
    figure scripts) should call this so every landscape is in one convention.  This is
    the grid-mean-zero gauge, which is robust to compare across fits.
    It differs from the fit/CRB enforcement gauge (``mean(theta)=0``) only by a
    constant.  (The min-subtraction in ``_BasePotential_on_grid`` is an *internal*
    exp-overflow safeguard for the likelihood, deliberately not used for reporting.)
    """
    u = potential.on_grid(grid)
    return u - u.mean()
