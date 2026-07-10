"""HMC posterior sampling of the landscape U(x), D and photophysics (SPEC 9).

Where ``infer.fit`` returns a MAP point estimate, this module draws from the full
posterior with Hamiltonian Monte Carlo (**pyro** NUTS/HMC).  It reuses
``objective.neg_log_posterior`` verbatim: that scalar *is* the potential energy the
sampler needs, and the whole model is autograd-differentiable end to end, so HMC is
a thin driver rather than a re-implementation.

pyro's NUTS is *real* NUTS: it chooses each trajectory's length dynamically via the
No-U-Turn criterion (bounded by ``max_tree_depth``) and adapts the step size AND the
mass matrix (``adapt_mass_matrix``, dense with ``full_mass=True``) during warmup.
So we hand it the ``build_log_prob`` scalar as a ``potential_fn`` and let pyro do the
preconditioning -- no hand-rolled Laplace/SoftAbs metric is needed.  (The single
sign convention to remember: pyro minimises *energy*, so ``potential_fn = -log_prob``.)

Two requirements make sampling well-posed (both handled here):

* A **proper prior** on the landscape.  The default ``curvature_penalty`` is the
  *improper* thin-plate limit (constant+linear unpenalised) -- fine for MAP, fatal
  for HMC (those directions random-walk and never mix).  So a proper GP prior is
  REQUIRED: ``prior.gp_sigma`` must be set (else we raise).  See ``objective.gp_penalty``.
* A **gauge anchor**.  Both the likelihood and the mean-centered GP prior are
  invariant to an additive constant in U, so ``mean(theta)`` is unidentified.  A
  tight Gaussian anchor (``gauge_sd``) pins that pure-gauge direction without
  touching the identifiable landscape shape.

Target the **spline** potential: low-dimensional (n_knots + logD + 4 log-rates),
with the GP prior acting directly on ``theta``.  The MLP is supported but a poor HMC
target (thousands of weights, mode/scaling symmetries -> poor mixing).

Flat parameter layout (all unconstrained, matching ``infer.fit`` / ``FreeRates``):
    z = [ potential params | logD | log_a_g, log_a_r, log_bg_g, log_bg_r ]
positives are sampled in log space (D = exp(logD), rate = exp(log_rate)).
"""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass, replace

import torch
from torch.func import functional_call

from .config import DTYPE, PriorConfig
from .objective import neg_log_posterior
from .photophysics import EffectiveRates

N_RATES = 4  # a_g, a_r, bg_g, bg_r


# --------------------------------------------------------------------------- #
# flat <-> structured parameter helpers
# --------------------------------------------------------------------------- #
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


def _flatten_init(potential, D_init, rates_init):
    """Build z0 from the potential's current params + D + rates (all unconstrained).

    Everything is placed on the potential's device so the concatenation (and the
    downstream sampler) stays on a single device.
    """
    pot_flat = torch.cat(
        [p.detach().reshape(-1) for _, p in potential.named_parameters()]
    ).to(DTYPE)
    dev = pot_flat.device
    logD = torch.log(torch.as_tensor(float(D_init), dtype=DTYPE, device=dev)).reshape(1)
    log_rates = torch.log(torch.tensor(
        [float(rates_init.a_g), float(rates_init.a_r),
         float(rates_init.bg_g), float(rates_init.bg_r)], dtype=DTYPE, device=dev
    ))
    return torch.cat([pot_flat, logD, log_rates])


# --------------------------------------------------------------------------- #
# log-posterior as a function of the flat vector
# --------------------------------------------------------------------------- #
class _NLPModule(torch.nn.Module):
    """Wraps the potential so ``functional_call`` can swap its params for the
    whole ``neg_log_posterior`` evaluation (all internal potential calls see the
    swapped params, then they are restored automatically)."""

    def __init__(self, potential):
        super().__init__()
        self.potential = potential

    def forward(self, D, rates, ipt, colors, mask, grid, C, R0, prior, p0,
                compile_mode=None, propagate_dtype=None):
        return neg_log_posterior(
            ipt, colors, mask, self.potential, D, rates, grid, C, R0, prior, p0=p0,
            compile_mode=compile_mode, propagate_dtype=propagate_dtype,
        )


def build_log_prob(
    batch, grid, potential, C, R0, prior: PriorConfig, rates_init, *,
    D_init, gauge_sd: float = 1.0, rate_sd: float = 1.0, logD_sd: float = 1.0,
    p0=None, compile_mode=None, propagate_dtype=None,
):
    """Return ``(log_prob_func, z0, info)`` for the sampler.

    ``log_prob_func(z) -> scalar`` is ``-(neg_log_posterior + rate_prior + gauge)``
    and is differentiable w.r.t. the flat vector ``z``.  ``sample_posterior`` wraps it
    into a pyro ``potential_fn`` (energy ``= -log_prob_func(z)``); pyro takes the grad.
    """
    if prior.gp_sigma is None:
        raise ValueError(
            "Posterior sampling needs a PROPER landscape prior: set prior.gp_sigma "
            "(the improper curvature penalty alone leaves the landscape posterior "
            "improper and HMC will not mix). See objective.gp_penalty / sample.py."
        )

    specs = _param_specs(potential)
    npot = sum(numel for _, _, numel in specs)
    is_spline = (len(specs) == 1 and specs[0][0] == "theta")

    module = _NLPModule(potential)
    ipt, colors, mask = batch.ipt, batch.colors, batch.mask

    # weak proper prior on logD (identifies + helps mixing); respect a user-set mean
    logD_mean = prior.logD_mean if prior.logD_mean is not None else math.log(float(D_init))
    prior_s = replace(prior, logD_mean=logD_mean, logD_std=logD_sd)

    log_rates0 = torch.log(torch.tensor(
        [float(rates_init.a_g), float(rates_init.a_r),
         float(rates_init.bg_g), float(rates_init.bg_r)],
        dtype=DTYPE, device=grid.device,
    ))

    def log_prob_func(z):
        flat_pot = z[:npot]
        logD = z[npot]
        log_rates = z[npot + 1:npot + 1 + N_RATES]

        pdict = _unflatten(flat_pot, specs, prefix="potential.")
        D = logD.exp()
        rates = EffectiveRates(*log_rates.exp())
        nlp = functional_call(
            module, pdict,
            args=(D, rates, ipt, colors, mask, grid, C, R0, prior_s, p0,
                  compile_mode, propagate_dtype),
        )
        rate_prior = 0.5 * (((log_rates - log_rates0) / rate_sd) ** 2).sum()
        gauge = (0.5 * (flat_pot.mean() / gauge_sd) ** 2) if is_spline \
            else torch.zeros((), dtype=z.dtype, device=z.device)
        return -(nlp + rate_prior + gauge)

    z0 = _flatten_init(potential, D_init, rates_init)
    info = dict(specs=specs, npot=npot, is_spline=is_spline, dim=z0.numel(),
                log_rates0=log_rates0, prior_sample=prior_s)
    return log_prob_func, z0, info


# --------------------------------------------------------------------------- #
# posterior draws container
# --------------------------------------------------------------------------- #
@dataclass
class PosteriorSamples:
    U: torch.Tensor       # [S, G] gauge-fixed landscapes (min = 0)
    D: torch.Tensor       # [S]
    rates: torch.Tensor   # [S, 4]  (a_g, a_r, bg_g, bg_r)
    theta: torch.Tensor   # [S, npot] raw potential params
    z: torch.Tensor       # [S, dim] raw unconstrained draws
    grid: torch.Tensor    # [G]

    def U_mean(self) -> torch.Tensor:
        return self.U.mean(0)

    def U_band(self, q=(0.05, 0.95)) -> torch.Tensor:
        """[len(q), G] posterior quantile band of U(x)."""
        qs = torch.tensor(q, dtype=self.U.dtype, device=self.U.device)
        return torch.quantile(self.U, qs, dim=0)

    def to_arviz(self):
        """Build an ``arviz.InferenceData`` (single chain) for R-hat / ESS."""
        import arviz as az
        import numpy as np
        post = {"logD": np.log(self.D.cpu().numpy())[None, :]}
        for k in range(self.theta.shape[1]):
            post[f"theta_{k}"] = self.theta[:, k].cpu().numpy()[None, :]
        names = ["log_a_g", "log_a_r", "log_bg_g", "log_bg_r"]
        for k, nm in enumerate(names):
            post[nm] = np.log(self.rates[:, k].cpu().numpy())[None, :]
        return az.from_dict(posterior=post)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def sample_posterior(
    batch, grid, potential, C, R0, prior: PriorConfig, rates_init, *,
    D_init=None, num_samples=1000, warmup=200, num_steps_per_sample=20,
    step_size=0.01, target_accept=0.8, sampler="nuts", seed=0,
    gauge_sd=1.0, rate_sd=1.0, logD_sd=1.0, p0=None,
    map_warmstart=True, map_optim=None, fit_rates=True,
    full_mass=True, max_tree_depth=10, jit_compile=False, init_jitter=0.0,
    compile_mode=None, propagate_dtype=None, verbose=True,
) -> PosteriorSamples:
    """Draw from the posterior over U(x), D and the 4 emission rates with pyro HMC.

    ``prior.gp_sigma`` MUST be set (proper landscape prior).  With
    ``map_warmstart`` the chain is initialised at the MAP (a quick ``infer.fit``),
    which slashes burn-in.  ``sampler``: "nuts" (default) or "hmc".

    The ``build_log_prob`` scalar is handed to pyro as a ``potential_fn`` (energy
    ``= -log_prob``); pyro adapts the step size and the mass matrix during warmup and
    -- for NUTS -- picks each trajectory's length dynamically (No-U-Turn), so no
    hand-built preconditioner is needed.

    Mixing knobs:

    * ``full_mass`` -- ``True`` (default) adapts a *dense* mass matrix (captures the
      strong GP-knot correlations + very different parameter scales, i.e. the pyro
      equivalent of the old dense Laplace covariance).  ``False`` adapts a diagonal
      metric (cheaper; use for high-dimensional MLP potentials).  For the dense metric
      to be well-estimated, keep ``warmup`` comfortably larger than the parameter
      dimension (pyro's Stan-style windowed adaptation reserves a 75-step start + 50-step
      end buffer, so very short warmups barely adapt it).
    * ``max_tree_depth`` -- upper bound on the NUTS doubling depth, i.e. a *cap* on the
      (dynamically chosen) trajectory length (only used for ``sampler="nuts"``).
    * ``num_steps_per_sample`` -- sets the HMC trajectory *length* to
      ``step_size * num_steps_per_sample`` using the INITIAL step size; pyro then
      dual-averages the step size, so the realised leapfrog count per proposal floats
      around this value rather than being fixed.  Only used for ``sampler="hmc"``
      (NUTS chooses its own trajectory length and ignores this).
    * ``jit_compile`` -- pyro's TorchScript JIT of the potential (off by default).
    * ``init_jitter`` -- std of Gaussian noise added to the (MAP) start, in
      z-space; used by ``sample_posterior_multi`` to over-disperse chains for R-hat.

    Returns ``PosteriorSamples`` (S = number of post-warmup draws).
    """
    import pyro
    from pyro.infer import HMC, MCMC, NUTS

    device = grid.device
    # the Cython simulator emits CPU batches; move batch (+ the small crosstalk
    # matrix) onto the grid's device so the likelihood (grid-derived
    # eigendecomposition) and the photon gaps / crosstalk all match.
    batch = batch.to(device)
    C = C.to(device)
    if map_warmstart:
        from .infer import fit
        if D_init is None:
            D_init = 1.0
        res = fit(batch, grid, potential, C, R0, D_init=D_init,
                  rates_init=rates_init, prior=prior, optim=map_optim,
                  fit_D=True, fit_rates=fit_rates, verbose=verbose)
        D_init = float(res.D)
        rates_init = res.rates  # MAP rates
    elif D_init is None:
        D_init = 1.0

    log_prob_func, z0, info = build_log_prob(
        batch, grid, potential, C, R0, prior, rates_init,
        D_init=D_init, gauge_sd=gauge_sd, rate_sd=rate_sd, logD_sd=logD_sd, p0=p0,
        compile_mode=compile_mode, propagate_dtype=propagate_dtype,
    )
    z0 = z0.to(device)

    pyro.set_rng_seed(int(seed))

    # over-dispersed start for multi-chain R-hat
    if init_jitter and init_jitter > 0:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(z0.numel(), generator=g, dtype=DTYPE).to(device)
        z0 = z0 + init_jitter * noise

    # pyro minimises the potential ENERGY, so the potential_fn is -log_prob.  The
    # single flat parameter lives under one site name ("z"); the returned draws come
    # back keyed by that name.
    def potential_fn(params):
        return -log_prob_func(params["z"])

    kernel_kw = dict(
        potential_fn=potential_fn, step_size=step_size, adapt_step_size=True,
        adapt_mass_matrix=True, full_mass=full_mass,
        target_accept_prob=target_accept, jit_compile=jit_compile,
    )
    if sampler.lower() == "nuts":
        kernel = NUTS(max_tree_depth=int(max_tree_depth), **kernel_kw)
    else:
        kernel = HMC(num_steps=int(num_steps_per_sample), **kernel_kw)

    mcmc = MCMC(
        kernel, num_samples=int(num_samples), warmup_steps=int(warmup),
        num_chains=1, initial_params={"z": z0}, disable_progbar=not verbose,
    )
    mcmc.run()

    # get_samples returns only the post-warmup draws, keyed by site name: [S, dim].
    Z = mcmc.get_samples()["z"].detach().to(device)

    npot = info["npot"]
    specs = info["specs"]
    U_list, D_list, rate_list, theta_list = [], [], [], []
    for z in Z:
        flat_pot = z[:npot]
        pdict = _unflatten(flat_pot, specs)  # bare potential keys
        with torch.no_grad():
            u = functional_call(potential, pdict, args=(grid,))
            u = u - u.min()
        U_list.append(u)
        theta_list.append(flat_pot.clone())
        D_list.append(z[npot].exp())
        rate_list.append(z[npot + 1:npot + 1 + N_RATES].exp())

    return PosteriorSamples(
        U=torch.stack(U_list),
        D=torch.stack(D_list),
        rates=torch.stack(rate_list),
        theta=torch.stack(theta_list),
        z=Z,
        grid=grid,
    )


# --------------------------------------------------------------------------- #
# multi-chain (R-hat / ESS across chains)
# --------------------------------------------------------------------------- #
@dataclass
class MultiChainPosterior:
    """Several chains + an arviz ``InferenceData`` with a real chain dimension.

    ``.U/.D/.rates/.theta`` concatenate all chains' draws (for pooled plotting);
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
    """Stack a list of ``PosteriorSamples`` into an arviz ``InferenceData``
    (chain dim = len(chains)); draws are truncated to the shortest chain."""
    import arviz as az
    import numpy as np

    n = min(int(c.D.shape[0]) for c in chains)
    post = {"logD": np.stack([np.log(c.D[:n].cpu().numpy()) for c in chains])}
    npot = chains[0].theta.shape[1]
    for k in range(npot):
        post[f"theta_{k}"] = np.stack([c.theta[:n, k].cpu().numpy() for c in chains])
    for k, nm in enumerate(["log_a_g", "log_a_r", "log_bg_g", "log_bg_r"]):
        post[nm] = np.stack([np.log(c.rates[:n, k].cpu().numpy()) for c in chains])
    return az.from_dict(posterior=post)


def sample_posterior_multi(
    batch, grid, potential, C, R0, prior: PriorConfig, rates_init, *,
    num_chains=4, overdisperse=0.3, base_seed=0, verbose=True, **kwargs,
) -> MultiChainPosterior:
    """Run ``num_chains`` HMC chains from over-dispersed starts for R-hat / ESS.

    A single chain gives R-hat = NaN (arviz needs >= 2 chains).  Each chain gets a
    fresh (deep-copied) ``potential`` -- ``sample_posterior`` warm-starts it to the
    MAP in place -- a distinct ``seed = base_seed + c``, and an
    ``init_jitter = overdisperse`` perturbation of the MAP start (so R-hat is not
    optimistic from all chains sharing one start).  Extra ``kwargs`` (``num_samples``,
    ``warmup``, ``sampler``, ``full_mass``, ``max_tree_depth``, ``map_optim``, ...)
    pass through to ``sample_posterior``.  Chains run sequentially (per-GPU
    parallelism is a separate concern).
    """
    kwargs.pop("init_jitter", None)   # controlled here via ``overdisperse``
    kwargs.pop("seed", None)
    chains = []
    for c in range(num_chains):
        if verbose:
            print(f"[multi-chain] chain {c + 1}/{num_chains} (seed={base_seed + c})")
        pot_c = copy.deepcopy(potential)
        ps = sample_posterior(
            batch, grid, pot_c, C, R0, prior, rates_init,
            seed=base_seed + c, init_jitter=overdisperse, verbose=verbose, **kwargs,
        )
        chains.append(ps)

    idata = None
    try:
        idata = _chains_to_arviz(chains)
    except Exception as e:  # noqa: BLE001 - arviz optional / stacking edge cases
        warnings.warn(f"could not build arviz InferenceData: {e!r}", RuntimeWarning)
    return MultiChainPosterior(chains=chains, idata=idata)
