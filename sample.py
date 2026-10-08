"""HMC/NUTS posterior sampling of the landscape U(x), D and the emission rates.

Where ``infer.fit`` returns a MAP point estimate, this module draws from the full
posterior with **pyro** NUTS/HMC.

The one thing to know about this module: **its target density is exactly the objective
``infer.fit`` minimises, negated.**  Term for term,

    log_prob(z) = -( neg_log_posterior(..., prior) + gauge_penalty(..., gauge_sd) )

which is ``infer.fit``'s ``closure_value()`` with a minus sign.  So the sampler adds no
prior of its own: everything that shapes the posterior comes from the ``PriorConfig`` you
pass (the curvature roughness prior and the background Gamma prior), exactly as it does
for the fit.  ``prior=None`` means pure MLE here just as it does there.  The single term
outside the config is the **gauge anchor** -- a Gaussian on ``mean(theta)``, which is a
choice of coordinates rather than a belief: the likelihood is invariant to ``U -> U + c``,
so that direction has to be pinned before it can be sampled.  ``infer.fit`` adds the same
anchor with the same ``gauge_sd``, which is why the two agree.

Caveat worth knowing before trusting a band: the curvature prior is the improper
thin-plate limit, so it constrains roughness but not the constant or linear directions of
``U``.  The gauge anchor handles the constant; the tilt is left to the likelihood.  The
practical remedy is to narrow the grid to the region the photons actually inform, which is
what the published fits do.

pyro's NUTS is real NUTS: it chooses each trajectory's length dynamically via the
No-U-Turn criterion (bounded by ``max_tree_depth``) and adapts both the step size and the
mass matrix during warmup.  We hand it ``build_log_prob``'s scalar as a ``potential_fn``
and let it do the preconditioning -- no hand-rolled metric.  (The one sign convention to
remember: pyro minimises *energy*, so ``potential_fn = -log_prob``.)

Flat parameter layout (unconstrained; positives are sampled in log space):

    z = [ theta | logD | log_a_g, log_a_r (, log_bg_g, log_bg_r) ]

The two background entries are present only when ``sample_bg`` is true; with a calibrated
background held fixed they drop out of ``z`` and are carried through from ``rates_init``.

Usage::

    post = sample_posterior(batch, grid, pot, C, R0, prior)   # KDE + MAP warm start
    U_mean = post.U_mean()
    lo, hi = post.U_band((0.05, 0.95))
"""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass

import torch
from torch.func import functional_call

from .config import DTYPE, PhysicsConstants, PriorConfig
from .objective import (neg_log_posterior, gauge_offset_from_theta,
                        gauge_penalty_from_offset)
from .photophysics import EffectiveRates

N_RATES = 4                                        # a_g, a_r, bg_g, bg_r
RATE_NAMES = ("a_g", "a_r", "bg_g", "bg_r")


# --------------------------------------------------------------------------- #
# flat <-> structured parameter helpers
# --------------------------------------------------------------------------- #
def _require_spline(potential, where):
    """The flat vector is ``[theta | logD | log rates]`` and the gauge anchor is
    ``mean(theta)``: spline-only.  Refuse anything else up front -- before the warm
    start can touch the potential and before ``torch.cat`` trips over an empty
    parameter list."""
    if not hasattr(potential, "_basis"):
        raise NotImplementedError(
            f"{where} supports SplinePotential only (its parameter vector is the knot "
            f"heights and its gauge anchor mean(theta)); got {type(potential).__name__}. "
            "FixedPotential / ParametricPotential landscapes have no sampler yet."
        )


def _param_specs(potential):
    """[(name, shape, numel)] for the potential's learnable parameters."""
    return [(n, tuple(p.shape), p.numel()) for n, p in potential.named_parameters()]


def _unflatten(flat, specs, prefix=""):
    """Map a flat 1-D tensor to a {name: view} dict (views keep the graph)."""
    out, ptr = {}, 0
    for name, shape, numel in specs:
        out[prefix + name] = flat[ptr:ptr + numel].view(shape)
        ptr += numel
    return out


def _flatten_init(potential, D_init, rates_init, *, device=None, sample_bg=True):
    """Build ``z0`` from the potential's current params, ``D_init`` and ``rates_init``.

    ``device`` defaults to the potential's own; ``sample_posterior`` passes the grid's so
    the whole vector and the target live on one device.  With ``sample_bg=False`` the two
    background entries are omitted -- they are held at their ``rates_init`` values.
    """
    pot_flat = torch.cat(
        [p.detach().reshape(-1) for _, p in potential.named_parameters()]
    ).to(DTYPE)
    dev = pot_flat.device if device is None else device
    pot_flat = pot_flat.to(dev)
    logD = torch.log(torch.as_tensor(float(D_init), dtype=DTYPE, device=dev)).reshape(1)
    values = [rates_init.a_g, rates_init.a_r]
    if sample_bg:
        values += [rates_init.bg_g, rates_init.bg_r]
    log_rates = torch.log(torch.tensor(
        [float(v) for v in values], dtype=DTYPE, device=dev))
    return torch.cat([pot_flat, logD, log_rates])


# --------------------------------------------------------------------------- #
# mass matrix: a starting matrix and the adaptation schedule
# --------------------------------------------------------------------------- #
MASS_SCHEDULES = ("stan", "single")


def _single_window_schedule(warmup_steps):
    """pyro's warmup schedule with ONE mass-matrix window (``mass_schedule="single"``).

    Same start and end buffers as pyro's default (75 and 50 steps, or 15 % / 10 % of a
    short warmup), but the doubling slow windows in between are merged into one, so the
    matrix is replaced exactly once, by the covariance of everything drawn in between.
    ``None`` below 20 warmup steps, where pyro adapts no mass matrix at all.
    """
    from pyro.infer.mcmc.adaptation import adapt_window

    W = int(warmup_steps)
    if W < 20:
        return None
    start, end = 75, 50
    if start + end + 25 > W:                     # pyro's own rule for short warmups
        start, end = int(0.15 * W), int(0.1 * W)
    return [adapt_window(0, start - 1), adapt_window(start, W - end - 1),
            adapt_window(W - end, W - 1)]


def _prepare_inverse_mass_matrix(M, dim, full_mass, device):
    """Validate a starting inverse mass matrix and bring it to the kernel's layout.

    Accepts ``[dim, dim]`` or a ``[dim]`` vector of variances, in z order. A dense kernel
    gets the symmetrised matrix (a vector becomes its diagonal), a diagonal kernel gets the
    diagonal. pyro takes a Cholesky factor, so a matrix that fails only by round-off gets a
    relative diagonal jitter of at most 1e-8; anything worse raises.
    """
    M = torch.as_tensor(M, dtype=DTYPE, device=device)
    if tuple(M.shape) not in ((dim,), (dim, dim)):
        raise ValueError(f"inverse_mass_matrix must have shape ({dim},) or ({dim}, {dim}), in "
                         f"z order [theta | logD | log rates]; got {tuple(M.shape)}")
    if not full_mass:
        diag = (M if M.ndim == 1 else torch.diagonal(M)).clone()
        if not (diag > 0).all():
            raise ValueError("inverse_mass_matrix: the variances must be positive")
        return diag
    M = torch.diag(M) if M.ndim == 1 else 0.5 * (M + M.T)
    eye = torch.eye(dim, dtype=DTYPE, device=device)
    scale = float(torch.diagonal(M).abs().mean())
    for rel in (0.0, 1e-12, 1e-10, 1e-8):
        Mj = M + rel * scale * eye
        if int(torch.linalg.cholesky_ex(Mj).info) == 0:
            return Mj
    raise ValueError("inverse_mass_matrix is not positive definite")


def _kernel_class(base, inverse_mass_matrix, mass_schedule):
    """``base`` (pyro's NUTS or HMC) itself, or a subclass that installs the starting
    inverse mass matrix and/or the single-window schedule right after pyro's ``setup``.

    pyro builds the identity matrix and its windowed schedule inside ``setup``, so anything
    set earlier is overwritten. After the swap the step size is searched again, under the
    new metric. With the defaults this returns ``base`` unchanged.
    """
    if inverse_mass_matrix is None and mass_schedule == "stan":
        return base

    class _Seeded(base):
        def setup(self, warmup_steps, *args, **kwargs):
            super().setup(warmup_steps, *args, **kwargs)
            adapter = self._adapter
            if inverse_mass_matrix is not None:
                (key,) = tuple(self.mass_matrix_adapter.inverse_mass_matrix)   # ("z",)
                self.mass_matrix_adapter.inverse_mass_matrix = {key: inverse_mass_matrix}
            if mass_schedule == "single" and adapter.adapt_mass_matrix:
                schedule = _single_window_schedule(warmup_steps)
                if schedule is not None:
                    adapter._adaptation_schedule = schedule
                    adapter._current_window = 0
            if adapter.adapt_step_size:
                adapter.reset_step_size_adaptation(self._initial_params)

    return _Seeded


def _jittered_start(z0, init_jitter, seed, scale=None):
    """``z0`` plus ``init_jitter`` times standard-normal noise from a CPU generator seeded
    with ``seed``: per coordinate in units of ``scale`` (the starting matrix's
    sqrt-diagonal) when given, isotropic in z otherwise (the historical behaviour)."""
    if not init_jitter or init_jitter <= 0:
        return z0
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(z0.numel(), generator=g, dtype=DTYPE).to(z0.device)
    if scale is not None:
        noise = noise * scale.to(z0.device)
    return z0 + init_jitter * noise


# --------------------------------------------------------------------------- #
# log-posterior as a function of the flat vector
# --------------------------------------------------------------------------- #
class _NLPModule(torch.nn.Module):
    """Wraps the potential so ``functional_call`` can swap its params for the whole
    ``neg_log_posterior`` evaluation.

    The swap is what lets the target be a function of a flat ``z`` while the likelihood
    keeps calling ``potential`` as an ordinary module; the original parameters are
    restored automatically afterwards.
    """

    def __init__(self, potential):
        super().__init__()
        self.potential = potential

    def forward(self, D, rates, ipt, colors, mask, grid, C, R0, prior, p0,
                compile_mode=None, propagate_dtype=None):
        return neg_log_posterior(
            ipt, colors, mask, self.potential, D, rates, grid, C, R0, prior, p0=p0,
            compile_mode=compile_mode, propagate_dtype=propagate_dtype,
        )


def _warn_about_prior(prior):
    """Diagnostics only -- these never change the target density."""
    if prior is None or not prior.active_terms():
        warnings.warn(
            "sampling with no prior at all: the target is the bare marginal likelihood "
            "plus the gauge anchor. U's roughness directions are then nearly "
            "unconstrained, so the chain random-walks there and the posterior band on U "
            "comes out far wider than the data warrant. This is almost never intended -- "
            "pass a PriorConfig.",
            RuntimeWarning, stacklevel=3,
        )
    if prior is not None and prior.curvature_weight and prior.curvature_norm == "l1":
        warnings.warn(
            "curvature_norm='l1' is not differentiable at d2=0; leapfrog cannot integrate "
            "the kink, so acceptance and the shape of the posterior are both biased. Use "
            "'l2' for sampling.",
            RuntimeWarning, stacklevel=3,
        )


def build_log_prob(
    batch, grid, potential, C, R0, prior: PriorConfig | None, rates_init, *,
    D_init, gauge_sd: float = 1.0, sample_bg: bool = True,
    p0=None, compile_mode=None, propagate_dtype=None,
):
    """Return ``(log_prob_func, z0, info)`` for the sampler.

    ``log_prob_func(z) -> scalar`` is ``-(neg_log_posterior + gauge)``, differentiable
    w.r.t. the flat vector ``z``.  That expression is ``infer.fit``'s objective negated;
    see the module docstring.  ``prior`` is used exactly as ``infer.fit`` uses it -- passed
    through untouched, ``None`` meaning pure MLE -- and no prior term is added here.

    ``sample_bg=False`` freezes ``bg_g``/``bg_r`` at their ``rates_init`` values and drops
    them from ``z``, mirroring ``infer.fit(fit_bg=False)``.
    """
    _require_spline(potential, "build_log_prob")
    _warn_about_prior(prior)

    specs = _param_specs(potential)
    npot = sum(numel for _, _, numel in specs)
    n_rates = N_RATES if sample_bg else 2

    module = _NLPModule(potential)
    ipt, colors, mask = batch.ipt, batch.colors, batch.mask

    # frozen backgrounds: detached, so no gradient flows into them
    bg_fixed = None if sample_bg else (
        torch.as_tensor(rates_init.bg_g, dtype=DTYPE, device=grid.device).detach(),
        torch.as_tensor(rates_init.bg_r, dtype=DTYPE, device=grid.device).detach(),
    )

    def log_prob_func(z):
        flat_pot = z[:npot]
        logD = z[npot]
        log_rates = z[npot + 1:npot + 1 + n_rates]

        pdict = _unflatten(flat_pot, specs, prefix="potential.")
        D = logD.exp()
        r = log_rates.exp()
        rates = (EffectiveRates(r[0], r[1], r[2], r[3]) if sample_bg
                 else EffectiveRates(r[0], r[1], bg_fixed[0], bg_fixed[1]))
        nlp = functional_call(
            module, pdict,
            args=(D, rates, ipt, colors, mask, grid, C, R0, prior, p0,
                  compile_mode, propagate_dtype),
        )
        # The gauge offset must be read off the swapped z-slice, not potential.theta:
        # the module's params only exist substituted inside the functional_call above.
        gauge = gauge_penalty_from_offset(gauge_offset_from_theta(flat_pot), gauge_sd)
        return -(nlp + gauge)          # == -(infer.fit's closure_value)

    z0 = _flatten_init(potential, D_init, rates_init,
                       device=grid.device, sample_bg=sample_bg)
    info = dict(specs=specs, npot=npot, dim=z0.numel(), n_sampled_rates=n_rates,
                rate_names=RATE_NAMES[:n_rates], sample_bg=sample_bg)
    return log_prob_func, z0, info


# --------------------------------------------------------------------------- #
# posterior draws container
# --------------------------------------------------------------------------- #
@dataclass
class PosteriorSamples:
    U: torch.Tensor       # [S, G] gauge-fixed landscapes (grid-mean = 0)
    D: torch.Tensor       # [S]
    rates: torch.Tensor   # [S, 4]  (a_g, a_r, bg_g, bg_r); frozen entries are constant
    theta: torch.Tensor   # [S, npot] knot heights
    z: torch.Tensor       # [S, dim] raw unconstrained draws
    grid: torch.Tensor    # [G]
    n_sampled_rates: int = N_RATES   # 2 when the backgrounds were held fixed
    # sampler diagnostics; None where unavailable (the counts with jit_compile=True)
    n_grad_warmup: int | None = None    # potential evaluations in warmup, incl. setup
    n_grad_sample: int | None = None    # ... while drawing: / S = the cost of one draw
    accept_rate: float | None = None    # mean acceptance probability of the draws
    n_divergences: int | None = None
    step_size: float | None = None
    inverse_mass_matrix: torch.Tensor | None = None   # final; [dim, dim], [dim] if diagonal

    def U_mean(self) -> torch.Tensor:
        return self.U.mean(0)

    def U_band(self, q=(0.05, 0.95)) -> torch.Tensor:
        """[len(q), G] posterior quantile band of U(x)."""
        qs = torch.tensor(q, dtype=self.U.dtype, device=self.U.device)
        return torch.quantile(self.U, qs, dim=0)

    def to_arviz(self):
        """Build an ``arviz.InferenceData`` (single chain) for R-hat / ESS."""
        import arviz as az
        return az.from_dict(posterior=_posterior_dict([self]))


def _posterior_dict(chains):
    """``{name: [chain, draw]}`` for arviz, truncated to the shortest chain.

    Backgrounds held fixed are constant by construction, and arviz reports a NaN R-hat for
    a zero-variance variable, so they are left out. A *sampled* rate that happens not to
    move is kept: that NaN is a real diagnostic about a stuck chain.
    """
    import numpy as np

    n = min(int(c.D.shape[0]) for c in chains)
    post = {"logD": np.stack([np.log(c.D[:n].cpu().numpy()) for c in chains])}
    for k in range(chains[0].theta.shape[1]):
        post[f"theta_{k}"] = np.stack([c.theta[:n, k].cpu().numpy() for c in chains])
    for k, nm in enumerate(RATE_NAMES[:chains[0].n_sampled_rates]):
        post[f"log_{nm}"] = np.stack([np.log(c.rates[:n, k].cpu().numpy()) for c in chains])
    return post


# --------------------------------------------------------------------------- #
# warm start
# --------------------------------------------------------------------------- #
def _physics_from(C, R0):
    """A ``PhysicsConstants`` rebuilt from the crosstalk matrix the caller already has.

    ``C[0,0]=C_gg`` (D->G), ``C[0,1]=C_gr`` (D->R), ``C[1,0]=C_rg`` (A->G),
    ``C[1,1]=C_rr`` (A->R) -- the layout ``PhysicsConstants.crosstalk_tensor`` produces.
    """
    return PhysicsConstants(R0=float(R0), C_gg=float(C[0, 0]), C_gr=float(C[0, 1]),
                            C_rg=float(C[1, 0]), C_rr=float(C[1, 1]))


def _warm_start(batch, grid, potential, C, R0, prior, rates_init, *, physics,
                kde_warmstart, kde_bin_ms, kde_kwargs, compile_mode, propagate_dtype,
                verbose):
    """Resolve ``rates_init`` and set ``potential.theta`` from the FRET histogram.

    Returns ``(rates_init, D_kde)``; ``D_kde`` is ``None`` unless the bin-width scan ran
    and profiled one out.  ``potential`` is modified in place, as ``init`` and ``infer``
    both do.
    """
    from . import init

    if rates_init is None:
        # stream_rates, not estimate_rates: the KDE inversion and the likelihood both read
        # a_g/a_r as emission brightnesses, which is what stream_rates solves for. A
        # calibrated background is the better starting point than a fraction of the
        # observed rate, so use the prior's when there is one.
        rates_init = init.stream_rates(
            batch,
            bg_g=None if prior is None else prior.bg_g_mean,
            bg_r=None if prior is None else prior.bg_r_mean,
            device=grid.device,
        )
    if not kde_warmstart:
        return rates_init, None

    kde_kw = dict(kde_kwargs or {})
    if kde_bin_ms is not None:
        kde_kw["bin_ms"] = float(kde_bin_ms)      # pinned: no scan, so no loglik kwargs
    else:
        kde_kw.setdefault("compile_mode", compile_mode)
        kde_kw.setdefault("propagate_dtype", propagate_dtype)
        kde_kw.setdefault("verbose", verbose)

    physics = physics or _physics_from(C, R0)
    try:
        out = init.kde_potential_init(potential, batch, grid, physics, rates_init,
                                      **kde_kw)
    except ValueError as e:
        # Raised when no time window holds enough photons to estimate an efficiency.
        # The landscape init is a convenience, not a correctness requirement, so keep
        # the potential the caller supplied rather than failing the whole run.
        warnings.warn(f"KDE warm start skipped ({e}); sampling from the potential as "
                      f"supplied", RuntimeWarning)
        return rates_init, None

    D_kde = float(out.D)
    return rates_init, (D_kde if math.isfinite(D_kde) else None)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def sample_posterior(
    batch, grid, potential, C, R0, prior: PriorConfig | None, rates_init=None, *,
    physics=None, kde_warmstart=True, kde_bin_ms=None, kde_kwargs=None,
    map_warmstart=True, map_optim=None, fit_rates=True, fit_bg=True, D_init=None,
    sampler="nuts", num_samples=1000, warmup=200, step_size=0.01, target_accept=0.8,
    full_mass=True, max_tree_depth=10, num_steps_per_sample=20, seed=0, init_jitter=0.0,
    gauge_sd=1.0, p0=None, inverse_mass_matrix=None, mass_schedule="stan", adapt_mass=True,
    adapt_step_size=True,
    compile_mode=None, propagate_dtype=None, jit_compile=False, verbose=True,
) -> PosteriorSamples:
    """Draw from the posterior over U(x), D and the emission rates with pyro HMC/NUTS.

    The target is ``infer.fit``'s objective negated (see the module docstring), so a chain
    and a fit run on the same ``prior`` and ``gauge_sd`` describe the same posterior.

    **Warm start.**  By default the landscape is initialised from the FRET histogram
    (``init.kde_potential_init``) and the chain then starts at the MAP (a quick
    ``infer.fit``), which cuts burn-in sharply.  ``rates_init=None`` builds emission
    brightnesses with ``init.stream_rates``, seeding the backgrounds from the prior's
    calibration when it has one.  ``D_init`` falls back to the D profiled out by the KDE
    bin-width scan, then to ``1.0``.  Set ``kde_bin_ms`` to pin the bin width and skip that
    scan, which is the expensive part; ``kde_kwargs`` goes to ``init.select_bin_ms``.
    ``physics`` is rebuilt from ``C``/``R0`` when not given.  Both warm starts modify
    ``potential`` in place.

    **Sampler.**  ``sampler="nuts"`` (default) or ``"hmc"``.  ``full_mass=True`` adapts a
    dense mass matrix, which captures the strong knot-to-knot correlations and the very
    different parameter scales; keep ``warmup`` comfortably above the parameter dimension
    for it to be well estimated (pyro's windowed adaptation reserves a 75-step start and a
    50-step end buffer).  ``max_tree_depth`` caps the NUTS doubling depth.
    ``num_steps_per_sample`` sets the trajectory length for ``"hmc"`` only -- NUTS chooses
    its own.  ``init_jitter`` over-disperses the start in z-space and is what
    ``sample_posterior_multi`` uses to make R-hat honest.

    **Objective.**  ``gauge_sd`` is the width of the anchor on ``mean(theta)``; it is
    forwarded to the MAP fit too, so the start and the chain share one gauge.

    ``fit_rates`` and ``fit_bg`` are not symmetric here.  ``fit_rates`` is passed to the
    MAP warm start and stops there: the chain samples ``a_g``/``a_r`` either way, since
    marginalising over the photophysics is exactly what a posterior is for (and what
    ``cramer_rao_bound`` scores).  ``fit_bg=False`` reaches both -- it holds
    ``bg_g``/``bg_r`` at ``rates_init`` in the fit *and* drops them from ``z`` -- because a
    separately calibrated background is a measurement to keep, not a parameter to infer.

    **Mass matrix.**  By default pyro starts from the identity and learns a dense matrix in
    doubling windows (25, 50, 100, ... steps after a 75-step buffer), each replacing the
    last; with posterior widths that differ by orders of magnitude it learns slowly and
    trees run into ``max_tree_depth``.  ``inverse_mass_matrix`` (``[dim, dim]`` or
    ``[dim]``, z order) starts it from a known covariance instead; the posterior CRB
    (``cramer_rao_bound(..., prior=prior)``, same prior, same gauge anchor, same order) is
    the natural choice.  ``mass_schedule="single"`` merges the slow windows into one, so a
    good start is not thrown away after 25 draws but replaced once, by the covariance of
    everything sampled until the 50-step end buffer.  ``adapt_mass=False`` keeps the
    starting matrix and adapts only the step size.  A slow direction is under-estimated by
    correlated warmup draws, so compare the final ``inverse_mass_matrix`` with the start.
    With a starting matrix, ``init_jitter`` is in units of its sqrt-diagonal.

    **Continuing a chain.**  ``adapt_step_size=False`` keeps ``step_size`` as given (no
    search, no dual averaging).  With ``warmup=0``, ``init_jitter=0``, ``adapt_mass=False``,
    a chain's final ``step_size`` and ``inverse_mass_matrix`` and its last draw as the start
    (``potential``'s knots, ``D_init``, ``rates_init``), this continues that chain with the
    same kernel; only the random-number stream is new.

    **Diagnostics.**  The draws carry the number of potential evaluations in warmup and in
    sampling (each one value + gradient, bar a few step-size searches), the mean acceptance
    probability of the draws (the statistic ``target_accept`` tunes; pyro's own "acceptance
    rate" counts the NUTS iterations that moved, which is ~1 whatever the step size), the
    number of divergences, the final step size and the final inverse mass matrix.

    Returns ``PosteriorSamples`` (S = number of post-warmup draws).
    """
    _require_spline(potential, "sample_posterior")
    if mass_schedule not in MASS_SCHEDULES:
        raise ValueError(f"mass_schedule must be one of {MASS_SCHEDULES}, got {mass_schedule!r}")
    import pyro
    from pyro.infer import HMC, MCMC, NUTS

    device = grid.device
    # the Cython simulator emits CPU batches; move the batch (and the small crosstalk
    # matrix) onto the grid's device so the likelihood, the photon gaps and the crosstalk
    # all agree.
    batch = batch.to(device)
    C = C.to(device)
    M0 = None
    if inverse_mass_matrix is not None:      # checked before the (expensive) warm starts
        dim = sum(p.numel() for p in potential.parameters()) + 1 + (N_RATES if fit_bg else 2)
        M0 = _prepare_inverse_mass_matrix(inverse_mass_matrix, dim, full_mass, device)

    rates_init, D_kde = _warm_start(
        batch, grid, potential, C, R0, prior, rates_init, physics=physics,
        kde_warmstart=kde_warmstart, kde_bin_ms=kde_bin_ms, kde_kwargs=kde_kwargs,
        compile_mode=compile_mode, propagate_dtype=propagate_dtype, verbose=verbose,
    )
    if D_init is None:
        D_init = D_kde if D_kde is not None else 1.0

    if map_warmstart:
        from .infer import fit
        res = fit(batch, grid, potential, C, R0, D_init=D_init, rates_init=rates_init,
                  prior=prior, optim=map_optim, fit_D=True, fit_rates=fit_rates,
                  fit_bg=fit_bg, verbose=verbose, gauge_sd=gauge_sd)
        if res.stop_reason.startswith("recovered"):
            warnings.warn(f"MAP warm start ended in {res.stop_reason!r} (a non-finite "
                          f"loss was hit and the best snapshot restored); the chain is "
                          f"starting from a poor point", RuntimeWarning)
        D_init = float(res.D)
        rates_init = res.rates

    # fit_rates governs the MAP warm start only -- the chain always samples the
    # brightnesses, because marginalising over the photophysics is the point of a
    # posterior. fit_bg is the one that also reaches the chain: a background you
    # calibrated separately should be held, not re-inferred.
    sample_bg = bool(fit_bg)
    log_prob_func, z0, info = build_log_prob(
        batch, grid, potential, C, R0, prior, rates_init, D_init=D_init,
        gauge_sd=gauge_sd, sample_bg=sample_bg, p0=p0,
        compile_mode=compile_mode, propagate_dtype=propagate_dtype,
    )

    pyro.set_rng_seed(int(seed))

    jitter_scale = None if M0 is None else (M0 if M0.ndim == 1 else torch.diagonal(M0)).sqrt()
    z0 = _jittered_start(z0, init_jitter, seed, jitter_scale)

    # pyro minimises the potential ENERGY, so the potential_fn is -log_prob. The whole
    # flat vector lives under one site name ("z"); draws come back keyed by it.
    # Every evaluation is counted (each one value + gradient, bar the few step-size
    # searches) and split at the warmup/sampling boundary by pyro's hook, which also reads
    # the kernel's diagnostics at the last draw, before pyro's cleanup resets them.
    stats = {"n": 0, "n_warmup": 0, "diag": None, "accept": None}

    def potential_fn(params):
        stats["n"] += 1
        return -log_prob_func(params["z"])

    def hook(kernel, params, stage, i):
        if stage.startswith("Warmup"):
            stats["n_warmup"] = stats["n"]
        elif i == int(num_samples) - 1:
            stats["diag"] = kernel.diagnostics()
            stats["accept"] = kernel._mean_accept_prob     # running mean over the draws

    kernel_kw = dict(
        potential_fn=potential_fn, step_size=step_size, adapt_step_size=bool(adapt_step_size),
        adapt_mass_matrix=bool(adapt_mass), full_mass=full_mass,
        target_accept_prob=target_accept, jit_compile=jit_compile,
    )
    base = NUTS if sampler.lower() == "nuts" else HMC
    Kernel = _kernel_class(base, M0, mass_schedule)
    if base is NUTS:
        kernel = Kernel(max_tree_depth=int(max_tree_depth), **kernel_kw)
    else:
        kernel = Kernel(num_steps=int(num_steps_per_sample), **kernel_kw)

    mcmc = MCMC(
        kernel, num_samples=int(num_samples), warmup_steps=int(warmup),
        num_chains=1, initial_params={"z": z0}, disable_progbar=not verbose,
        hook_fn=hook,
    )
    mcmc.run()

    # get_samples returns only the post-warmup draws: [S, dim].
    Z = mcmc.get_samples()["z"].detach().to(device)
    npot, n_rates = info["npot"], info["n_sampled_rates"]

    theta = Z[:, :npot]
    # u = M @ theta is linear in theta, so one basis build covers every draw. (Calling the
    # potential per draw would rebuild the spline basis each time -- same numbers, orders
    # of magnitude slower.)
    M = potential._basis(grid)                              # [G, K]
    U = theta @ M.T                                         # [S, G]
    U = U - U.mean(dim=1, keepdim=True)      # grid-mean-zero reporting gauge
    D = Z[:, npot].exp()
    rates = Z[:, npot + 1:npot + 1 + n_rates].exp()
    if not sample_bg:                        # re-attach the frozen pair so rates is [S, 4]
        bg = torch.tensor([float(rates_init.bg_g), float(rates_init.bg_r)],
                          dtype=rates.dtype, device=rates.device)
        rates = torch.cat([rates, bg.expand(rates.shape[0], 2)], dim=1)

    diag = stats["diag"] or {}
    M_final = kernel.inverse_mass_matrix               # the adapter survives pyro's cleanup
    M_final = next(iter(M_final.values())).detach().to(device) if M_final else None
    counted = not jit_compile                          # a traced potential is called once
    return PosteriorSamples(
        U=U, D=D, rates=rates, theta=theta, z=Z, grid=grid, n_sampled_rates=n_rates,
        n_grad_warmup=stats["n_warmup"] if counted else None,
        n_grad_sample=stats["n"] - stats["n_warmup"] if counted else None,
        accept_rate=None if stats["accept"] is None else float(stats["accept"]),
        n_divergences=len(diag["divergences"]) if "divergences" in diag else None,
        step_size=float(kernel.step_size), inverse_mass_matrix=M_final,
    )


# --------------------------------------------------------------------------- #
# multi-chain (R-hat / ESS across chains)
# --------------------------------------------------------------------------- #
@dataclass
class MultiChainPosterior:
    """Several chains plus an arviz ``InferenceData`` with a real chain dimension.

    ``.U/.D/.rates/.theta`` concatenate every chain's draws (for pooled plotting);
    ``.summary()`` returns ``arviz.summary`` (R-hat / ESS)."""

    chains: list                 # list[PosteriorSamples]
    idata: object = None         # arviz.InferenceData (chain dim = num_chains)

    @property
    def U(self):
        return torch.cat([c.U for c in self.chains], 0)

    @property
    def D(self):
        return torch.cat([c.D for c in self.chains], 0)

    @property
    def rates(self):
        return torch.cat([c.rates for c in self.chains], 0)

    @property
    def theta(self):
        return torch.cat([c.theta for c in self.chains], 0)

    @property
    def grid(self):
        return self.chains[0].grid

    def summary(self):
        import arviz as az
        return az.summary(self.idata)


def _chains_to_arviz(chains):
    """Stack ``PosteriorSamples`` into an arviz ``InferenceData`` (chain dim = len)."""
    import arviz as az
    return az.from_dict(posterior=_posterior_dict(chains))


def _run_one_chain(payload):
    """One chain, from a picklable payload -- the unit ``n_parallel`` fans out over.

    Module level so ``spawn`` can pickle it by reference; the whole payload (a CUDA
    ``potential``, ``grid``, ``C`` and the returned draws) survives the round trip in both
    directions on the same device, so nothing needs marshalling to the CPU and back.
    """
    batch, grid, potential, C, R0, prior, rates_init, seed, jitter, verbose, kwargs = payload
    return sample_posterior(batch, grid, potential, C, R0, prior, rates_init,
                            kde_warmstart=False, seed=seed, init_jitter=jitter,
                            verbose=verbose, **kwargs)


def sample_posterior_multi(
    batch, grid, potential, C, R0, prior: PriorConfig | None, rates_init=None, *,
    num_chains=4, overdisperse=0.3, base_seed=0, n_parallel=1, verbose=True, **kwargs,
) -> MultiChainPosterior:
    """Run ``num_chains`` chains from over-dispersed starts for R-hat / ESS.

    A single chain gives R-hat = NaN (arviz needs >= 2).  Each chain gets a deep-copied
    ``potential``, a distinct ``seed = base_seed + c``, and an ``init_jitter =
    overdisperse`` perturbation of its start (in units of the starting matrix's
    sqrt-diagonal when ``inverse_mass_matrix`` is passed), so R-hat is not flattered by
    every chain sharing one starting point.  Extra ``kwargs`` pass through to
    ``sample_posterior``.

    The KDE warm start runs **once**, here, on the template potential -- the per-chain
    copies inherit it.  Left to ``sample_posterior`` it would repeat its bin-width scan
    (a full likelihood pass per candidate) for every chain.

    ``n_parallel`` (default 1, i.e. sequential) runs that many chains at once in ``spawn``
    worker processes.  On one GPU this is worth roughly **1.4x**, not ``n_parallel``-fold:
    measured on an A6000 the device sits at 64-70% utilisation, and the idle time is
    Python-side work between kernel launches.  Threads therefore do not help at all (0.9x,
    the GIL serialises exactly the gap being targeted) while separate interpreters do
    (1.42x at two workers), but 64% utilisation caps the total near 1.56x.

    The cost to weigh against it: every worker is a fresh interpreter and **re-compiles the
    likelihood**, measured at 12-23 s, so a short chain finishes *slower* in parallel than
    it would sequentially.  Point ``TORCHINDUCTOR_CACHE_DIR`` at a shared directory and
    later workers hit the on-disk cache instead of compiling again.
    """
    _require_spline(potential, "sample_posterior_multi")
    kwargs.pop("init_jitter", None)   # controlled here via ``overdisperse``
    kwargs.pop("seed", None)

    rates_init, D_kde = _warm_start(
        batch.to(grid.device), grid, potential, C.to(grid.device), R0, prior, rates_init,
        physics=kwargs.pop("physics", None),
        kde_warmstart=kwargs.pop("kde_warmstart", True),
        kde_bin_ms=kwargs.pop("kde_bin_ms", None),
        kde_kwargs=kwargs.pop("kde_kwargs", None),
        compile_mode=kwargs.get("compile_mode"),
        propagate_dtype=kwargs.get("propagate_dtype"),
        verbose=verbose,
    )
    if kwargs.get("D_init") is None and D_kde is not None:
        kwargs["D_init"] = D_kde

    n_parallel = max(1, min(int(n_parallel), num_chains))
    # Interleaved pyro progress bars are unreadable, so the workers stay quiet and the
    # parent reports one line per chain instead.
    payloads = [(batch, grid, copy.deepcopy(potential), C, R0, prior, rates_init,
                 base_seed + c, overdisperse, verbose and n_parallel == 1, kwargs)
                for c in range(num_chains)]

    if n_parallel > 1:
        import multiprocessing as mp
        if verbose:
            print(f"[multi-chain] {num_chains} chains on {n_parallel} workers")
        # spawn, not fork: a forked process cannot inherit a CUDA context.  Plain
        # multiprocessing rather than torch.multiprocessing, whose reductions hand tensors
        # over by CUDA IPC -- that would tie the returned draws to a worker that has exited.
        chains = []
        with mp.get_context("spawn").Pool(n_parallel) as pool:
            for c, ps in enumerate(pool.imap(_run_one_chain, payloads)):  # order preserved
                if verbose:
                    print(f"[multi-chain] chain {c + 1}/{num_chains} done "
                          f"(seed={base_seed + c})")
                chains.append(ps)
    else:
        chains = []
        for c, p in enumerate(payloads):
            if verbose:
                print(f"[multi-chain] chain {c + 1}/{num_chains} (seed={base_seed + c})")
            chains.append(_run_one_chain(p))

    idata = None
    try:
        idata = _chains_to_arviz(chains)
    except Exception as e:  # noqa: BLE001 - arviz optional / stacking edge cases
        warnings.warn(f"could not build arviz InferenceData: {e!r}", RuntimeWarning)
    return MultiChainPosterior(chains=chains, idata=idata)
