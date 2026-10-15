"""NUTS sampling of the posterior over the landscape U(x), D, the emission rates and the
weight of the curvature prior.

``infer.fit`` returns the MAP at a fixed curvature weight.  This module draws from the full
posterior with pyro's NUTS and samples that weight as well, so the data decide how smooth
the landscape is.

The chain moves in one unconstrained vector, with the positive quantities in log space,

    z = [ theta | ln D | ln a_g, ln a_r (, ln bg_g, ln bg_r) | ln w ]

where ``theta`` are the K knot heights of the spline landscape.  Its target is

    log p(z) =  log L(theta, D, rates)          photon-by-photon likelihood
              - bg_penalty(rates)                Gamma calibration prior (if configured)
              - w R(theta) + (K-2)/2 ln w        curvature prior with its normalising term
              - mean(theta)^2 / (2 gauge_sd^2)   gauge anchor (U -> U + c)

with flat priors on ln D, ln a_g, ln a_r and ln w.

* ``R(theta)`` is ``objective.curvature_penalty_spline``: the sum of squared second
  differences of the knot heights, the roughness ``infer.fit`` penalises.
* ``(K-2)/2 ln w`` normalises the curvature prior, a Gaussian of rank K-2 whose precision
  is proportional to ``w``.  Without it the chain would send ``w`` to zero.
* The gauge anchor pins the one direction the likelihood cannot see, a constant shift of
  U.  ``infer.fit`` adds the same anchor.
* With ``sample_bg=False`` the two backgrounds are held at given values and drop out of z.

At a fixed ``w`` the target is ``infer.fit``'s objective at that weight, negated, plus the
constant ``(K-2)/2 ln w``: a fit at weight ``w`` finds the mode of the chain's posterior
given that ``w``.

pyro's NUTS chooses each trajectory's length itself (up to ``max_tree_depth`` doublings)
and adapts the step size and a dense mass matrix during warmup.  It minimises an energy,
so it is handed ``-log p``.

Usage::

    post = sample_posterior(batch, grid, pot, C, R0, prior)   # KDE + MAP warm start
    post.U, post.D, post.curvature_weight                     # one row per draw
"""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass

import torch

from .config import DTYPE, PhysicsConstants, PriorConfig
from .forward import marginal_loglik_batch
from .objective import (bg_penalty, curvature_penalty_spline, gauge_offset_from_theta,
                        gauge_penalty_from_offset)
from .photophysics import EffectiveRates

N_RATES = 4                                        # a_g, a_r, bg_g, bg_r
RATE_NAMES = ("a_g", "a_r", "bg_g", "bg_r")


# --------------------------------------------------------------------------- #
# the target
# --------------------------------------------------------------------------- #
def _require_spline(potential, where):
    """The knot heights are what the chain samples: spline landscapes only.  Refused up
    front, before a warm start can touch the potential."""
    if not hasattr(potential, "_basis"):
        raise NotImplementedError(
            f"{where} supports SplinePotential only (it samples the knot heights); got "
            f"{type(potential).__name__}. FixedPotential / ParametricPotential landscapes "
            "have no sampler."
        )


def _require_l2_curvature_prior(prior):
    """The weight of the curvature prior is sampled, so there must be one: L2, because
    ``(K-2)/2 ln w`` is the normalising term of the Gaussian form, and with a positive
    weight, which is where the MAP warm start (and by default the chain) puts it."""
    if prior is None or prior.curvature_norm != "l2" or not prior.curvature_weight > 0:
        raise ValueError(
            "the sampler needs a PriorConfig with an L2 curvature prior "
            f"(curvature_norm='l2', curvature_weight > 0); got {prior!r}"
        )


class _Landscape:
    """A landscape given by its values on the grid: ``on_grid`` is the one thing the
    likelihood asks of a potential."""

    def __init__(self, U):
        self.U = U

    def on_grid(self, grid):
        return self.U


def build_log_prob(
    batch, grid, potential, C, R0, prior: PriorConfig, rates_init, *, D_init,
    curvature_weight_init=None, gauge_sd: float = 1.0, sample_bg: bool = True, p0=None,
    compile_mode=None, propagate_dtype=None,
):
    """Return ``(log_prob, z0, info)``: the target of the module docstring as a function of
    the flat vector ``z``, the starting point, and the layout of ``z``.

    ``prior`` must have an L2 curvature prior with a positive ``curvature_weight``; its
    background terms, if any, enter as they do in ``infer.fit``.  The start is the knot
    heights in ``potential``, ``D_init``, ``rates_init`` and the weight
    ``curvature_weight_init`` (default ``prior.curvature_weight``, the weight the MAP is
    fitted with).  ``sample_bg=False`` holds ``bg_g``/``bg_r`` at their ``rates_init``
    values and drops them from ``z``, as ``infer.fit(fit_bg=False)`` does.

    ``info``: ``npot`` (K), ``n_sampled_rates`` (4, or 2 with fixed backgrounds),
    ``rate_names``, ``i_log_weight`` (the last entry of ``z``) and ``dim``.
    """
    _require_spline(potential, "build_log_prob")
    _require_l2_curvature_prior(prior)
    w0 = float(prior.curvature_weight if curvature_weight_init is None
               else curvature_weight_init)
    if not w0 > 0:
        raise ValueError(f"curvature_weight_init must be positive, got {w0}")

    K = potential.theta.numel()
    n_rates = N_RATES if sample_bg else 2
    M = potential._basis(grid)                     # [G, K]: U on the grid = M @ theta
    knots_x = potential.knots_x
    ipt, colors, mask = batch.ipt, batch.colors, batch.mask
    with_bg_prior = prior.bg_g_mean is not None or prior.bg_r_mean is not None
    if not sample_bg:                              # held fixed: constants, no gradient
        bg_g, bg_r = (torch.as_tensor(float(v), dtype=DTYPE, device=grid.device)
                      for v in (rates_init.bg_g, rates_init.bg_r))

    def log_prob(z):
        theta, logD, u = z[:K], z[K], z[-1]        # u = ln w
        r = z[K + 1:K + 1 + n_rates].exp()
        rates = (EffectiveRates(r[0], r[1], r[2], r[3]) if sample_bg
                 else EffectiveRates(r[0], r[1], bg_g, bg_r))
        logp = marginal_loglik_batch(
            ipt, colors, mask, _Landscape(M @ theta), logD.exp(), rates, grid, C, R0,
            p0=p0, compile_mode=compile_mode, propagate_dtype=propagate_dtype)
        if with_bg_prior:
            logp = logp - bg_penalty(rates, prior)
        logp = logp - u.exp() * curvature_penalty_spline(theta, knots_x) + 0.5 * (K - 2) * u
        return logp - gauge_penalty_from_offset(gauge_offset_from_theta(theta), gauge_sd)

    rates0 = [rates_init.a_g, rates_init.a_r]
    if sample_bg:
        rates0 += [rates_init.bg_g, rates_init.bg_r]
    logs = torch.log(torch.tensor([float(D_init), *(float(v) for v in rates0), w0],
                                  dtype=DTYPE, device=grid.device))
    z0 = torch.cat([potential.theta.detach().to(device=grid.device, dtype=DTYPE), logs])
    info = dict(npot=K, n_sampled_rates=n_rates, rate_names=RATE_NAMES[:n_rates],
                i_log_weight=z0.numel() - 1, dim=z0.numel())
    return log_prob, z0, info


# --------------------------------------------------------------------------- #
# posterior draws container
# --------------------------------------------------------------------------- #
@dataclass
class PosteriorSamples:
    U: torch.Tensor                  # [S, G] landscapes, grid-mean zero
    D: torch.Tensor                  # [S]
    rates: torch.Tensor              # [S, 4] (a_g, a_r, bg_g, bg_r); held-fixed ones constant
    theta: torch.Tensor              # [S, K] knot heights
    curvature_weight: torch.Tensor   # [S]
    z: torch.Tensor                  # [S, dim] the raw draws
    grid: torch.Tensor               # [G]
    n_sampled_rates: int = N_RATES   # 2 when the backgrounds were held fixed
    # the kernel at the end of the run -- what a later call needs to continue this chain
    step_size: float | None = None
    inverse_mass_matrix: torch.Tensor | None = None   # [dim, dim]; [dim] if diagonal
    n_divergences: int | None = None                  # divergent transitions among the draws

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
    post["log_curvature_weight"] = np.stack(
        [np.log(c.curvature_weight[:n].cpu().numpy()) for c in chains])
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
# one chain
# --------------------------------------------------------------------------- #
def _nuts_starting_from(inverse_mass_matrix):
    """pyro's NUTS, starting from ``inverse_mass_matrix`` instead of the identity.

    pyro builds the identity inside ``setup``, so the matrix is installed right after it.
    If the step size is still adapted, its initial search is redone under the new matrix.
    """
    from pyro.infer import NUTS

    class _NUTS(NUTS):
        def setup(self, warmup_steps, *args, **kwargs):
            super().setup(warmup_steps, *args, **kwargs)
            (key,) = tuple(self.mass_matrix_adapter.inverse_mass_matrix)   # ("z",)
            self.mass_matrix_adapter.inverse_mass_matrix = {key: inverse_mass_matrix}
            if self._adapter.adapt_step_size:
                self._adapter.reset_step_size_adaptation(self._initial_params)

    return _NUTS


def sample_posterior(
    batch, grid, potential, C, R0, prior: PriorConfig, rates_init=None, *,
    physics=None, kde_warmstart=True, kde_bin_ms=None, kde_kwargs=None,
    map_warmstart=True, map_optim=None, fit_rates=True, fit_bg=True, D_init=None,
    curvature_weight_init=None, num_samples=1000, warmup=200, step_size=0.01,
    target_accept=0.8, full_mass=True, max_tree_depth=10, seed=0, init_jitter=0.0,
    gauge_sd=1.0, p0=None, inverse_mass_matrix=None, adapt_step_size=True,
    compile_mode=None, propagate_dtype=None, jit_compile=False, verbose=True,
) -> PosteriorSamples:
    """One NUTS chain over the posterior of the module docstring: U(x), D, the rates and
    the curvature weight.

    **Warm start.**  By default the landscape is initialised from the FRET histogram
    (``init.kde_potential_init``) and the chain then starts at the MAP (a quick
    ``infer.fit`` at ``prior.curvature_weight``), which cuts burn-in sharply.
    ``rates_init=None`` builds emission brightnesses with ``init.stream_rates``, seeding the
    backgrounds from the prior's calibration when it has one.  ``D_init`` falls back to the
    D profiled out by the KDE bin-width scan, then to ``1.0``.  ``kde_bin_ms`` pins the bin
    width and skips that scan, which is the expensive part; ``kde_kwargs`` goes to
    ``init.select_bin_ms``; ``physics`` is rebuilt from ``C``/``R0`` when not given.  Both
    warm starts modify ``potential`` in place.  The weight starts at
    ``curvature_weight_init``, by default ``prior.curvature_weight``.

    **Sampler.**  ``full_mass=True`` adapts a dense mass matrix during warmup, which
    captures the strong knot-to-knot correlations and the very different parameter scales;
    keep ``warmup`` well above the dimension of ``z`` (pyro's windowed adaptation reserves a
    75-step start and a 50-step end buffer).  ``max_tree_depth`` caps the NUTS doublings.
    ``init_jitter`` over-disperses the start in z-space, which ``sample_posterior_multi``
    uses to make R-hat honest.  ``gauge_sd`` reaches the MAP fit too, so the start and the
    chain share one gauge.

    **Rates.**  ``fit_rates`` reaches the MAP warm start only: the chain always samples
    ``a_g``/``a_r``.  ``fit_bg=False`` reaches both -- it holds ``bg_g``/``bg_r`` at
    ``rates_init`` in the fit and drops them from ``z`` -- because a separately calibrated
    background is a measurement to keep, not a parameter to infer.

    **Continuing a chain.**  The draws carry the kernel's final ``step_size`` and
    ``inverse_mass_matrix``.  Pass them back with ``warmup=0``, ``init_jitter=0``,
    ``adapt_step_size=False`` and both warm starts off, starting from the chain's last draw
    (the potential's knots, ``D_init``, ``rates_init`` and ``curvature_weight_init``), and
    the chain continues with the same kernel; only the random-number stream is new.

    Returns ``PosteriorSamples`` with the post-warmup draws.
    """
    _require_spline(potential, "sample_posterior")
    _require_l2_curvature_prior(prior)
    import pyro
    from pyro.infer import MCMC, NUTS

    device = grid.device
    # the Cython simulator emits CPU batches; move the batch (and the small crosstalk
    # matrix) onto the grid's device so the likelihood, the photon gaps and the crosstalk
    # all agree.
    batch = batch.to(device)
    C = C.to(device)

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

    sample_bg = bool(fit_bg)
    log_prob, z0, info = build_log_prob(
        batch, grid, potential, C, R0, prior, rates_init, D_init=D_init,
        curvature_weight_init=curvature_weight_init, gauge_sd=gauge_sd, sample_bg=sample_bg,
        p0=p0, compile_mode=compile_mode, propagate_dtype=propagate_dtype,
    )

    pyro.set_rng_seed(int(seed))
    if init_jitter and init_jitter > 0:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(z0.numel(), generator=g, dtype=DTYPE).to(device)
        z0 = z0 + init_jitter * noise

    # pyro minimises the potential ENERGY, so it gets -log p. The whole flat vector lives
    # under one site name ("z"); the draws come back keyed by it.
    Kernel = (NUTS if inverse_mass_matrix is None else
              _nuts_starting_from(torch.as_tensor(inverse_mass_matrix, dtype=DTYPE,
                                                  device=device)))
    kernel = Kernel(potential_fn=lambda params: -log_prob(params["z"]),
                    step_size=step_size, adapt_step_size=bool(adapt_step_size),
                    adapt_mass_matrix=True, full_mass=full_mass,
                    target_accept_prob=target_accept, max_tree_depth=int(max_tree_depth),
                    jit_compile=jit_compile)
    mcmc = MCMC(kernel, num_samples=int(num_samples), warmup_steps=int(warmup),
                num_chains=1, initial_params={"z": z0}, disable_progbar=not verbose)
    mcmc.run()

    Z = mcmc.get_samples()["z"].detach().to(device)      # post-warmup draws only: [S, dim]
    K, n_rates = info["npot"], info["n_sampled_rates"]
    theta = Z[:, :K]
    # U = M @ theta is linear in theta, so one basis build covers every draw.
    U = theta @ potential._basis(grid).T
    U = U - U.mean(dim=1, keepdim=True)                  # grid-mean-zero reporting gauge
    rates = Z[:, K + 1:K + 1 + n_rates].exp()
    if not sample_bg:                                    # re-attach the frozen pair: [S, 4]
        bg = torch.tensor([float(rates_init.bg_g), float(rates_init.bg_r)],
                          dtype=rates.dtype, device=rates.device)
        rates = torch.cat([rates, bg.expand(rates.shape[0], 2)], dim=1)

    # The kernel's final state survives pyro's cleanup. Divergences come from pyro's own
    # per-chain record; mcmc.diagnostics() would also compute ESS / R-hat, which needs >= 4 draws.
    M_final = next(iter(kernel.inverse_mass_matrix.values())).detach().to(device)
    return PosteriorSamples(U=U, D=Z[:, K].exp(), rates=rates, theta=theta,
                            curvature_weight=Z[:, -1].exp(), z=Z, grid=grid,
                            n_sampled_rates=n_rates, step_size=float(kernel.step_size),
                            inverse_mass_matrix=M_final,
                            n_divergences=len(mcmc._diagnostics[0]["divergences"]))


# --------------------------------------------------------------------------- #
# multi-chain (R-hat / ESS across chains)
# --------------------------------------------------------------------------- #
@dataclass
class MultiChainPosterior:
    """Several chains plus an arviz ``InferenceData`` with a real chain dimension.

    ``.U/.D/.rates/.theta/.curvature_weight`` concatenate every chain's draws (for pooled
    plotting); ``.summary()`` returns ``arviz.summary`` (R-hat / ESS)."""

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
    def curvature_weight(self):
        return torch.cat([c.curvature_weight for c in self.chains], 0)

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
    batch, grid, potential, C, R0, prior: PriorConfig, rates_init=None, *,
    num_chains=4, overdisperse=0.3, base_seed=0, n_parallel=1, verbose=True, **kwargs,
) -> MultiChainPosterior:
    """Run ``num_chains`` chains from over-dispersed starts for R-hat / ESS.

    A single chain gives R-hat = NaN (arviz needs >= 2).  Each chain gets a deep-copied
    ``potential``, a distinct ``seed = base_seed + c``, and an ``init_jitter =
    overdisperse`` perturbation of its start, so R-hat is not flattered by every chain
    sharing one starting point.  Extra ``kwargs`` pass through to ``sample_posterior``.

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
    _require_l2_curvature_prior(prior)
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
