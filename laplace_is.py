"""laplace_is.py -- error bars via Laplace-proposal importance sampling.

``infer.fit`` returns only a MAP *point* estimate.  This module turns that point
into posterior *samples* (and hence error bars) cheaply, without a full HMC run:

1. Reconstruct the exact log-posterior ``log_prob_func(z)`` used by the sampler via
   ``sample.build_log_prob`` -- so the target here is *identical* to the one
   ``sample.sample_posterior`` (pyro HMC/NUTS) draws from.
2. Build a Laplace (Gaussian / Student-t) proposal centred at the MAP from the
   Hessian of the negative log-posterior at that point.
3. Importance-sample the proposal, self-normalise the weights, and SIR-resample down
   to equally-weighted draws.

The returned object is the same :class:`sample.PosteriorSamples` container HMC
returns, so all downstream tooling (``U_mean``/``U_band``/``to_arviz``) works
unchanged.  Import quality diagnostics (Kish ESS, PSIS Pareto-k) are attached as
attributes and printed when ``verbose``.

Everything happens in the *unconstrained* flat vector

    z = [ theta (npot) | logD | log_a_g, log_a_r, log_bg_g, log_bg_r ]

exactly as in ``sample.build_log_prob``.  The target ``pi(z) prop exp(log_prob_func(z))``
and the proposal ``q(z)`` are BOTH densities on this z-space w.r.t. Lebesgue measure,
and the ``exp`` maps (``D = exp(logD)``, ``rate = exp(log_rate)``) are applied ONLY to
the drawn samples as a deterministic push-forward.  Consequently the importance weights
``w = pi / q`` carry **no** log/exp Jacobian term -- adding one would be a bug.  (This is
the same reason HMC samples ``z`` with ``potential_fn = -log_prob_func`` and never applies
a Jacobian either.)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from torch.func import functional_call

from .config import DTYPE
from .sample import build_log_prob, PosteriorSamples, N_RATES

__all__ = ["laplace_importance_samples", "LaplaceProposal"]


# --------------------------------------------------------------------------- #
# Laplace proposal (Gaussian / multivariate Student-t) built from the Hessian
# --------------------------------------------------------------------------- #
@dataclass
class LaplaceProposal:
    """A Laplace proposal ``q(z)`` centred at ``z_map``.

    The proposal *scale* matrix is ``Sigma_scale = L @ L.T``; for a Gaussian this is
    also its covariance, for a Student-t the covariance is ``df/(df-2) * Sigma_scale``.
    We use the covariance-matched convention (see :func:`_build_proposal`) so that
    ``cov_scale`` inflates the proposal covariance identically for both families.
    """

    z_map: torch.Tensor        # [dim] proposal centre (the MAP vector)
    L: torch.Tensor            # [dim, dim] scale factor, L @ L.T = Sigma_scale
    log_det_scale: float       # log|Sigma_scale|
    dist: str                  # "gaussian" | "student_t"
    df: float                  # Student-t d.o.f. (ignored for Gaussian)
    cov: torch.Tensor          # [dim, dim] Laplace covariance (pre cov_scale, = H^-1 floored)

    @property
    def dim(self) -> int:
        return int(self.z_map.numel())

    def sample(self, n, gen, rng):
        """Draw ``n`` points and their log-proposal-density.

        ``gen`` is a torch Generator (standard normals), ``rng`` a numpy Generator
        (chi-square draws for the Student-t; torch's Chi2 does not take a generator).
        Returns ``(z [n, dim], log_q [n])``.
        """
        dim = self.dim
        g = torch.randn(n, dim, generator=gen, dtype=DTYPE, device=self.z_map.device)
        step = g @ self.L.T                       # [n, dim], z - mu = L g  (row form)
        if self.dist == "gaussian":
            z = self.z_map + step
            maha = (g * g).sum(1)                  # (z-mu)^T Sigma_scale^-1 (z-mu) = ||g||^2
            log_q = (-0.5 * dim * math.log(2.0 * math.pi)
                     - 0.5 * self.log_det_scale
                     - 0.5 * maha)
        elif self.dist == "student_t":
            df = float(self.df)
            u = torch.as_tensor(rng.chisquare(df, size=n), dtype=DTYPE,
                                device=self.z_map.device)
            scale = torch.sqrt(df / u)             # [n]
            z = self.z_map + step * scale[:, None]
            maha = df * (g * g).sum(1) / u         # Mahalanobis^2 w.r.t. Sigma_scale
            p = dim
            log_q = (math.lgamma(0.5 * (df + p)) - math.lgamma(0.5 * df)
                     - 0.5 * p * math.log(df * math.pi)
                     - 0.5 * self.log_det_scale
                     - 0.5 * (df + p) * torch.log1p(maha / df))
        else:
            raise ValueError(f"unknown proposal {self.dist!r} (use 'gaussian'|'student_t')")
        return z, log_q


def _hessian_at(log_prob_func, z_map):
    """H = Hessian of ``-log_prob_func`` at ``z_map`` (the observed information),
    symmetrised.  Eager float64.  Raises on a non-finite Hessian (grid extends into
    a high-potential tail where the generator's eigendecomposition gradient blows up).
    """
    z0 = z_map.detach().clone()

    def neg_lp(z):
        return -log_prob_func(z)

    try:
        # vectorize=False: a plain vjp loop (NO vmap) -- log_prob_func wraps
        # functional_call + a numpy-built spline basis and is not functorch-traceable,
        # so we must not vmap it. dim is tiny (n_knots + 5), so the loop is cheap.
        H = torch.autograd.functional.hessian(neg_lp, z0, vectorize=False)
    except (RuntimeError, TypeError):
        # fall back to the explicit row-by-row double backward (fisher._mean_hessian)
        H = _hessian_double_backward(neg_lp, z0)
    H = 0.5 * (H + H.T)
    if not torch.isfinite(H).all():
        raise ValueError(
            "Non-finite Hessian at the MAP: the landscape likely spans a very high "
            "potential (U >> 10 kT) somewhere on `grid`, where the generator's "
            "eigendecomposition gradient is unstable. Narrow `grid` to the "
            "data-visited region and refit."
        )
    return H


def _hessian_double_backward(f, z0):
    """Dense Hessian of scalar ``f`` at ``z0`` via row-by-row double backward."""
    z = z0.detach().clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(f(z), z, create_graph=True)
    rows = []
    for i in range(z.numel()):
        (row,) = torch.autograd.grad(grad[i], z, retain_graph=True)
        rows.append(row.detach())
    return torch.stack(rows)


def _build_proposal(z_map, H, *, dist, df, cov_scale, eig_rtol):
    """Turn the Hessian into a :class:`LaplaceProposal`.

    Eigen-decompose ``H = V diag(lam) V^T`` and floor the eigenvalues at
    ``eig_rtol * lam_max``.  This (a) inverts the near-flat gauge direction stably and
    (b) repairs tiny negative eigenvalues from an approximate (Adam-fit) MAP -- both of
    which a plain ``cholesky(H^-1)`` cannot handle.  Then

        Laplace covariance  Sigma = V diag(1/lam_floor) V^T
        scale factor        c     = cov_scale^2 * (gaussian: 1 ; student_t: (df-2)/df)
        L                   = sqrt(c) * V diag(lam_floor^{-1/2})     (L L^T = c * Sigma)
        log|Sigma_scale|    = dim*log(c) - sum(log lam_floor)

    The Student-t ``c`` uses the covariance-matched convention so that the proposal
    covariance equals ``cov_scale^2 * Sigma`` for both families.
    """
    if dist == "student_t" and not (df > 2.0):
        raise ValueError(
            f"Student-t proposal needs df > 2 for the covariance-matched convention "
            f"(got df={df}); use df>=3, or proposal='gaussian'."
        )
    lam, V = torch.linalg.eigh(H)
    lam_max = lam.max().clamp_min(torch.finfo(lam.dtype).tiny)
    lam_floor = torch.clamp(lam, min=float(eig_rtol) * lam_max)

    c = float(cov_scale) ** 2
    if dist == "student_t":
        c *= (df - 2.0) / df

    inv = 1.0 / lam_floor
    cov = (V * inv) @ V.T                                   # Laplace covariance (pre cov_scale)
    cov = 0.5 * (cov + cov.T)
    scale_vec = math.sqrt(c) * lam_floor.rsqrt()            # [dim]
    L = V * scale_vec                                       # V @ diag(scale_vec)
    log_det_scale = V.shape[0] * math.log(c) - float(torch.log(lam_floor).sum())
    return LaplaceProposal(z_map=z_map.detach(), L=L, log_det_scale=log_det_scale,
                           dist=dist, df=float(df), cov=cov)


# --------------------------------------------------------------------------- #
# importance sampling in z-space (target-agnostic core; used by the public fn
# and directly unit-testable with a synthetic log_prob_func)
# --------------------------------------------------------------------------- #
def _importance_sample(log_prob_func, z_map, *, n_samples, oversample, dist, df,
                       cov_scale, eig_rtol, seed, verbose):
    """Core: build proposal, draw ``M = oversample*n_samples`` points, weight them,
    diagnose, and SIR-resample to ``n_samples`` equally-weighted indices.

    Returns a dict with the raw proposal draws, (smoothed) normalised weights, the
    resample indices, and diagnostics.  No push-forward to physical units here.
    """
    device = z_map.device
    H = _hessian_at(log_prob_func, z_map)
    prop = _build_proposal(z_map, H, dist=dist, df=df, cov_scale=cov_scale,
                           eig_rtol=eig_rtol)

    M = int(oversample) * int(n_samples)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    z, log_q = prop.sample(M, gen, rng)

    # score the target at each draw (no vmap: functional_call + numpy-built spline
    # basis are not functorch-traceable). weights need no gradient.
    log_p = torch.full((M,), float("-inf"), dtype=DTYPE, device=device)
    n_bad = 0
    with torch.no_grad():
        for i in range(M):
            try:
                val = log_prob_func(z[i])
            except Exception:  # noqa: BLE001 -- extreme tail draw broke the likelihood
                n_bad += 1
                continue
            if torch.isfinite(val):
                log_p[i] = val
            else:
                n_bad += 1

    log_w_raw = torch.where(torch.isfinite(log_p), log_p - log_q,
                            torch.full_like(log_p, float("-inf")))
    if not torch.isfinite(log_w_raw).any():
        raise RuntimeError(
            "All importance weights are non-finite: the proposal never landed in a "
            "region of finite posterior density. The MAP/Hessian is likely off; try "
            "proposal='gaussian', a smaller df, or larger cov_scale."
        )

    # up-to-constant log marginal-likelihood estimate (proposal q is normalised;
    # neg_log_posterior drops additive constants, so this is comparable across runs
    # sharing the same data + prior, not an absolute evidence).
    log_evidence = float(torch.logsumexp(log_w_raw, 0) - math.log(M))

    # PSIS-smoothed weights + Pareto-k (falls back to raw self-normalised weights).
    pareto_k = None
    log_w_use = log_w_raw
    try:
        import arviz as az
        lw = log_w_raw.detach().cpu().numpy().astype(float).copy()
        smoothed, khat = az.stats.psislw(lw)
        log_w_use = torch.as_tensor(np.asarray(smoothed), dtype=DTYPE, device=device)
        pareto_k = float(np.asarray(khat).reshape(-1)[0])
    except Exception:  # noqa: BLE001 -- arviz missing / psis edge case
        pass

    W = torch.softmax(log_w_use, 0)
    ess = float(1.0 / (W * W).sum())
    ess_frac = ess / M

    idx = _systematic_resample(W, int(n_samples), rng)

    if verbose:
        kstr = "n/a" if pareto_k is None else f"{pareto_k:.2f}"
        print(f"[laplace_is] proposal={dist} df={df} cov_scale={cov_scale} "
              f"M={M} -> n={n_samples} | ESS={ess:.0f} ({100*ess_frac:.1f}%) "
              f"pareto_k={kstr} logZ_hat={log_evidence:.2f}"
              + (f" | {n_bad} non-finite draws" if n_bad else ""))
    if ess_frac < 0.1:
        warnings.warn(
            f"Low importance-sampling ESS ({100*ess_frac:.1f}% of {M} draws): the "
            f"Laplace proposal is a poor match to the posterior. SIR draws will contain "
            f"many duplicates. Try larger cov_scale, smaller df, or sample.sample_posterior "
            f"(HMC).", RuntimeWarning)
    if pareto_k is not None and pareto_k > 0.7:
        warnings.warn(
            f"PSIS Pareto-k = {pareto_k:.2f} > 0.7: importance-weight variance is "
            f"unreliable; treat these error bars with caution and prefer HMC.",
            RuntimeWarning)

    return dict(z=z, weights=W, log_weights=log_w_raw, idx=idx, ess=ess,
                ess_frac=ess_frac, pareto_k=pareto_k, log_evidence=log_evidence,
                n_nonfinite=n_bad, cov=prop.cov, z_map=z_map.detach())


def _systematic_resample(W, n, rng):
    """Systematic (low-variance) resampling: ``n`` indices drawn prop to ``W``."""
    cdf = torch.cumsum(W, 0).detach().cpu().numpy()
    cdf[-1] = 1.0
    u0 = float(rng.random())
    positions = (np.arange(n) + u0) / n
    idx = np.searchsorted(cdf, positions, side="left")
    idx = np.clip(idx, 0, W.numel() - 1)
    return torch.as_tensor(idx, dtype=torch.long, device=W.device)


# --------------------------------------------------------------------------- #
# z -> physical push-forward (matches sample.sample_posterior's reporting gauge)
# --------------------------------------------------------------------------- #
def _pushforward(z, potential, grid, info):
    """``z [S, dim]`` -> ``(U [S, G] gauge-fixed, D [S], rates [S, 4], theta [S, npot])``.

    Reporting gauge is grid-mean-zero (``u - u.mean()``), matching
    ``sample.sample_posterior`` / ``infer.recovered_potential``.
    """
    npot = info["npot"]
    theta = z[:, :npot]
    if info["is_spline"]:
        B = potential._basis(grid)                      # [G, npot], fixed linear map
        U = theta @ B.T                                 # [S, G]
    else:
        specs = info["specs"]
        rows = []
        with torch.no_grad():
            for s in range(theta.shape[0]):
                pdict = _unflatten_bare(theta[s], specs)
                rows.append(functional_call(potential, pdict, args=(grid,)))
        U = torch.stack(rows)
    U = U - U.mean(dim=1, keepdim=True)                 # grid-mean-zero gauge
    D = z[:, npot].exp()
    rates = z[:, npot + 1:npot + 1 + N_RATES].exp()
    return U, D, rates, theta.clone()


def _unflatten_bare(flat, specs):
    """Flat potential slice -> {bare_param_name: view} for ``functional_call``."""
    out, ptr = {}, 0
    for name, shape, numel in specs:
        out[name] = flat[ptr:ptr + numel].view(shape)
        ptr += numel
    return out


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def laplace_importance_samples(
    result, batch, grid, C, R0, prior, *,
    n_samples=2000, oversample=4, proposal="student_t", df=5.0, cov_scale=1.2,
    eig_rtol=1e-8, gauge_sd=1.0, rate_sd=1.0, logD_sd=1.0, p0=None,
    seed=0, verbose=True,
) -> PosteriorSamples:
    """Posterior samples (error bars) from a MAP estimate via Laplace-proposal IS.

    Parameters
    ----------
    result : infer.FitResult
        The MAP estimate (uses ``.potential``, ``.D``, ``.rates``).  This is the only
        thing that changes run-to-run; it supplies the proposal *centre*.
    batch, grid, C, R0, prior
        The SAME context used for the fit.  ``prior.gp_sigma`` MUST be set (a proper
        landscape prior; the improper curvature penalty alone leaves the posterior
        improper and the Hessian non-invertible).  Pass the identical ``prior`` you fit
        with, otherwise the proposal is centred off-mode (IS still corrects for it, but
        ESS degrades).
    n_samples : int
        Number of equally-weighted draws returned.
    oversample : int
        Draw ``oversample * n_samples`` proposal points, then SIR-resample down.  Larger
        values give cleaner (less-duplicated) equal-weight draws when ESS is modest.
    proposal : {"student_t", "gaussian"}
        Student-t (default, heavy tails) gives finite-variance importance weights and is
        the robust choice; Gaussian is exact only when the posterior is truly Gaussian.
    df : float
        Student-t degrees of freedom (needs ``df > 2``; ~4-7 is the robust range).
    cov_scale : float
        Inflate the proposal covariance by this factor (1.1-1.5 raises ESS when the
        quadratic Laplace approximation under-covers the posterior bulk).
    eig_rtol : float
        Relative floor on the Hessian eigenvalues for a stable inverse.
    gauge_sd, rate_sd, logD_sd, p0
        Forwarded to :func:`sample.build_log_prob` (must match the sampling target).
    seed : int
        Reproducible proposal draws + resampling.
    verbose : bool
        Print ESS / Pareto-k / logZ diagnostics.

    Returns
    -------
    sample.PosteriorSamples
        Equally-weighted draws (fields ``U, D, rates, theta, z, grid``).  Error bars:
        e.g. ``samples.U_band((0.05, 0.95))``.  Import diagnostics are attached as
        attributes ``.ess``, ``.ess_frac``, ``.pareto_k``, ``.log_evidence``,
        ``.n_nonfinite`` and the Laplace covariance ``.cov`` / centre ``.z_map``.
    """
    if prior is None or getattr(prior, "gp_sigma", None) is None:
        raise ValueError(
            "laplace_importance_samples needs the SAME proper prior used for the fit, "
            "with prior.gp_sigma set (kT). The GP landscape prior makes the posterior "
            "proper so the Laplace Hessian is invertible; without it the mean-level and "
            "long-wavelength landscape directions are unconstrained. Re-run the fit with "
            "PriorConfig(..., gp_sigma=<kT>, gp_lengthscale=<nm>) and pass that prior."
        )

    device = grid.device
    batch = batch.to(device)
    C = C.to(device)

    # exact same target sample.sample_posterior / HMC use; z0 IS the MAP vector.
    log_prob_func, z_map, info = build_log_prob(
        batch, grid, result.potential, C, R0, prior, result.rates,
        D_init=float(result.D), gauge_sd=gauge_sd, rate_sd=rate_sd, logD_sd=logD_sd,
        p0=p0,   # eager float64: NOT forwarding compile_mode/propagate_dtype
    )
    z_map = z_map.to(device)

    out = _importance_sample(
        log_prob_func, z_map, n_samples=n_samples, oversample=oversample,
        dist=proposal, df=df, cov_scale=cov_scale, eig_rtol=eig_rtol,
        seed=seed, verbose=verbose,
    )

    z_res = out["z"][out["idx"]]
    U, D, rates, theta = _pushforward(z_res, result.potential, grid, info)
    ps = PosteriorSamples(U=U, D=D, rates=rates, theta=theta, z=z_res, grid=grid)

    # attach import diagnostics (harmless extra attributes on the dataclass instance)
    ps.ess = out["ess"]
    ps.ess_frac = out["ess_frac"]
    ps.pareto_k = out["pareto_k"]
    ps.log_evidence = out["log_evidence"]
    ps.n_nonfinite = out["n_nonfinite"]
    ps.cov = out["cov"]
    ps.z_map = out["z_map"]
    return ps
