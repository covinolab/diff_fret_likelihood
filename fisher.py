"""Fisher information and the Cramér–Rao bound for the smFRET likelihood.

Give it the traces and the ground-truth parameters; get back the Cramér–Rao
bound — the lower bound on the covariance of any unbiased estimator of those
parameters (`Cov(θ̂) ⪰ I(θ)⁻¹`, the Cramér–Rao inequality).

The bound is the inverse of the Fisher information matrix evaluated at the
truth.  At the true parameters the score has mean zero and
``I(θ) = E[score·scoreᵀ]``, so one pass of the per-trace scores gives the Fisher
information for the whole dataset (it is *additive* over independent traces:
``F_N = Σ_i s_i s_iᵀ = N·F₁``).  This reuses the package's own differentiable
likelihood (`forward.build_propagator_from_u`, `generator.stationary`) via
`torch.func.vmap`/`jacrev`, exactly the machinery validated in
``tests/test_bartlett_fisher.py``.

Parameterisation (flat vector ``φ``, ``P = K + 5``):

    φ = [ θ_1..θ_K  |  lnD  |  ln a_g, ln a_r, ln bg_g, ln bg_r ]

the spline knot heights (kT), the diffusion coefficient and the four
photophysics rates scored together.  Positives are scored in natural-log space
(matching ``infer.FreeRates`` / ``sample.py``); physical-unit σ's are recovered
by the delta method (``σ_D = D·σ_lnD`` etc.).

Gauge: the likelihood is exactly invariant to ``U → U + const``, so the
all-knots-equal direction is an exact null-space of the Fisher matrix.  The
covariance is therefore the pseudo-inverse restricted to the informative
subspace — i.e. the landscape bound is reported in the sum-to-zero gauge (each
knot's σ is relative to the mean level).  This is the SAME gauge ``infer.fit``
now *enforces* (its ``mean(theta)=0`` anchor) and the sampler anchors, so the CRB
σ and the fit/posterior spread are directly comparable.  (``recovered_potential``
reports the grid-mean-zero gauge, which differs by a constant only — shape and
per-knot σ are unaffected.)

Identifiability: the *pure* (prior-free) CRB is finite only on the identifiable
subspace.  Landscape directions the data don't constrain — the gauge, plus any
knots outside the FRET-observable window — are dropped by the pseudo-inverse;
``null_dim`` counts them (``1`` = gauge only).  When ``null_dim > 1`` the σ of a
knot lying in a dropped direction is a *lower bound* (the pseudo-inverse gives it
no variance from that direction), so inspect the returned Fisher diagonal to see
which knots are informed.  Add a landscape prior (regularised Fisher) if you need
a finite bound on every knot.

Caveats:
  * The bound is evaluated at the supplied ground-truth ``φ`` in the estimator's
    own finite (spline-knot) parameterisation — it is the CRB an MLE in that
    parameterisation would attain.  Keep ``grid`` to the data-visited region: a
    grid extending into a high-potential tail (U ≫ 10 kT) destabilises the
    eigendecomposition gradient and the scores go non-finite (raised as an error).
  * The batched likelihood conditions on the first photon and drops the leading
    and trailing survival gaps; this puts a small ``O(N/√M)`` windowing-boundary
    bias on the *absolute* photon-rate scores for short traces (see the
    ``tests/test_bartlett_fisher`` docstring).  Use longer traces if the
    absolute-rate CRB must be unbiased; the landscape and D bounds are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import DTYPE, PriorConfig
from .forward import build_propagator_from_u, marginal_loglik_batch
from .generator import stationary
from .objective import prior_penalty, gauge_penalty_from_offset
from .photophysics import EffectiveRates

_RATE_LOG_NAMES = ["log_a_g", "log_a_r", "log_bg_g", "log_bg_r"]


@dataclass
class CRBResult:
    """Cramér–Rao bound for one dataset at the ground-truth parameters.

    ``fisher``           : [P,P] total Fisher information ``F_N`` for the dataset.
    ``fisher_per_trace`` : [P,P] ``F_N / N`` (Fisher is additive over traces).
    ``cov``              : [P,P] the CRB = gauge-fixed ``inv(F_N)`` (pseudo-inverse
                           on the informative subspace; landscape in sum-to-zero gauge).
    ``sigma``            : [P] ``sqrt(diag(cov))`` in the inference parameterisation
                           (kT knots, lnD, ln-rates).
    ``param_names``      : [P] names, ``["knot_0",…,"logD","log_a_g",…]``.
    ``sigma_physical``   : dict of σ in physical units — ``knots`` (kT, per knot),
                           ``D`` (nm²/ms), ``a_g/a_r/bg_g/bg_r`` (kHz), via the delta method.
    ``n_traces``, ``n_photons`` : dataset size.
    ``null_dim``         : # of dropped/floored null directions.  Without a prior:
                           1 = the landscape gauge only.  With a prior: 0 for a
                           healthy proper posterior (any count > 0 flags a direction
                           the prior failed to pin numerically).
    ``info_matrix_rel_frob`` : if ``validate=True``, ‖Cov(score) − (−E[H])‖ / ‖E[H]‖
                           (information-matrix identity check); else ``None``.
    ``posterior_precision`` : if a ``prior`` was passed, the posterior information
                           ``F_N + H_prior`` (the analytic Laplace precision); ``cov``
                           is then its inverse (the posterior covariance) rather than
                           the pure-likelihood CRB.  ``None`` for the prior-free CRB.
    ``prior_included``   : whether ``cov``/``sigma`` include the prior (posterior)
                           or are the pure-likelihood Cramér–Rao bound.
    """

    fisher: torch.Tensor
    fisher_per_trace: torch.Tensor
    cov: torch.Tensor
    sigma: torch.Tensor
    param_names: list[str]
    sigma_physical: dict
    n_traces: int
    n_photons: int
    null_dim: int
    info_matrix_rel_frob: float | None = None
    posterior_precision: torch.Tensor | None = None
    prior_included: bool = False


# --------------------------------------------------------------------------- #
# Parameter (un)packing and the linear knot basis
# --------------------------------------------------------------------------- #
def _knot_basis(potential, grid):
    """(B [G,K], b0 [G]) so that ``on_grid = B @ theta + b0`` (linear knots)."""
    if not hasattr(potential, "theta"):
        raise NotImplementedError(
            "cramer_rao_bound requires a SplinePotential (a landscape linear in "
            f"its knot heights); got {type(potential).__name__}. An MLP potential "
            "has thousands of parameters and no dense CRB — refit with "
            "PotentialConfig(kind='spline')."
        )
    if hasattr(potential, "_basis"):
        B = potential._basis(grid)                       # on_grid = B @ theta
        b0 = torch.zeros(grid.shape[0], dtype=B.dtype, device=B.device)
        return B, b0
    # generic linear-in-theta fallback: probe on_grid at 0 and the unit knots.
    theta0 = potential.theta.detach().clone()
    G, K = grid.shape[0], theta0.numel()
    with torch.no_grad():
        potential.theta.zero_()
        b0 = potential.on_grid(grid).clone()
        B = torch.empty(G, K, dtype=b0.dtype, device=b0.device)
        for k in range(K):
            e = torch.zeros_like(theta0)
            e[k] = 1.0
            potential.theta.copy_(e)
            B[:, k] = potential.on_grid(grid) - b0
        potential.theta.copy_(theta0)
    return B, b0


def _pack_truth_phi(potential, D, rates, device):
    """Build ``φ* = [theta | lnD | ln rates]`` from the ground-truth objects."""
    theta = potential.theta.detach().to(device)
    K = theta.numel()
    phi = torch.empty(K + 5, dtype=DTYPE, device=device)
    phi[:K] = theta
    phi[K] = torch.log(torch.as_tensor(D, dtype=DTYPE, device=device))
    phi[K + 1] = torch.log(torch.as_tensor(rates.a_g, dtype=DTYPE, device=device))
    phi[K + 2] = torch.log(torch.as_tensor(rates.a_r, dtype=DTYPE, device=device))
    phi[K + 3] = torch.log(torch.as_tensor(rates.bg_g, dtype=DTYPE, device=device))
    phi[K + 4] = torch.log(torch.as_tensor(rates.bg_r, dtype=DTYPE, device=device))
    return phi


def _unpack_phi(phi, K):
    D = torch.exp(phi[K])
    r = phi[K + 1:K + 5]
    rates = EffectiveRates(torch.exp(r[0]), torch.exp(r[1]),
                           torch.exp(r[2]), torch.exp(r[3]))
    return phi[:K], D, rates


# --------------------------------------------------------------------------- #
# Functional single-trace log-likelihood + per-trace scores (vmap/jacrev)
# --------------------------------------------------------------------------- #
def _mp_recursion(prop, ipt, colors, mask, p0v, pdt):
    """Scaled forward recursion for one trace (lifted from the validated
    ``tests/test_bartlett_fisher._mp_recursion``): optional fp32 propagation with
    an fp64 running log-normaliser (mirrors ``forward._recur_step``)."""
    s = prop.s
    v = p0v / s
    c0 = v.abs().sum()
    v = v / c0
    log_norm = torch.log(c0)                              # stays float64
    lam, Q, muG, muR = prop.lam, prop.Q, prop.mu_G, prop.mu_R
    if pdt is not None:
        lam, Q, muG, muR = lam.to(pdt), Q.to(pdt), muG.to(pdt), muR.to(pdt)
        v = v.to(pdt)
    ones = torch.ones_like(muG)
    zero = torch.zeros((), dtype=v.dtype, device=v.device)
    for k in range(ipt.shape[0]):
        tau = torch.where(mask[k], ipt[k].to(v.dtype), zero)
        v = Q @ (torch.exp(lam * tau) * (Q.T @ v))        # e^{A tau} v
        emit = torch.where(colors[k] == 0, muG, muR)
        emit = torch.where(mask[k], emit, ones)
        v = v * emit
        c = v.abs().sum()
        c = torch.where(c > 0, c, torch.ones_like(c))
        v = v / c
        log_norm = log_norm + torch.log(c).to(log_norm.dtype)
    total = torch.dot(s, v.to(s.dtype))
    return torch.log(total.clamp_min(1e-300)) + log_norm


def _single_logL(phi, ipt, colors, mask, B, b0, grid, C, R0, dx, jitter, pdt, p0):
    """Marginal log-lik of ONE (padded) trace — differentiable in ``phi``."""
    K = B.shape[1]
    theta, D, rates = _unpack_phi(phi, K)
    u = B @ theta + b0
    u = u - u.min()                                       # gauge-fix (as forward does)
    prop = build_propagator_from_u(u, D, rates, grid, C, R0, dx, jitter)
    p0v = stationary(u) if p0 is None else p0
    return _mp_recursion(prop, ipt, colors, mask, p0v, pdt)


def _per_trace_scores(phi, ipt, colors, mask, B, b0, grid, C, R0, dx, jitter,
                      chunk, pdt, p0):
    """[N, P] per-trace scores ``d logL_i / d phi`` at ``phi``."""
    N = ipt.shape[0]

    def f(phi_, i, c, m):
        return _single_logL(phi_, i, c, m, B, b0, grid, C, R0, dx, jitter, pdt, p0)

    try:
        from torch.func import vmap, jacrev
        jac = vmap(jacrev(f, argnums=0), in_dims=(None, 0, 0, 0))
        outs = []
        for s0 in range(0, N, chunk):
            sl = slice(s0, min(s0 + chunk, N))
            outs.append(jac(phi, ipt[sl], colors[sl], mask[sl]).detach())
        return torch.cat(outs, 0)
    except Exception as exc:  # pragma: no cover - robustness fallback
        rows = []
        for i in range(N):
            p = phi.clone().requires_grad_(True)
            ll = f(p, ipt[i], colors[i], mask[i])
            (gi,) = torch.autograd.grad(ll, p)
            rows.append(gi.detach())
        return torch.stack(rows, 0)


def _mean_hessian(phi, ipt, colors, mask, B, b0, grid, C, R0, jitter, pdt, p0, hb=24):
    """E[H] = (1/N) d²(Σ_i logL_i)/dφ² via autograd double-backward (validation)."""

    class _LinearKnotPotential:
        def __init__(self, theta, B, b0):
            self.theta, self.B, self.b0 = theta, B, b0

        def on_grid(self, grid):
            return self.B @ self.theta + self.b0

    K, P = B.shape[1], phi.shape[0]
    N = ipt.shape[0]
    Hsum = torch.zeros(P, P, dtype=DTYPE, device=phi.device)
    for s0 in range(0, N, hb):
        sl = slice(s0, min(s0 + hb, N))
        p = phi.clone().requires_grad_(True)
        theta, D, rates = _unpack_phi(p, K)
        pot = _LinearKnotPotential(theta, B, b0)
        ll = marginal_loglik_batch(ipt[sl], colors[sl], mask[sl], pot, D, rates,
                                   grid, C, R0, p0=p0, jitter=jitter, reduce="sum",
                                   propagate_dtype=pdt)
        (g1,) = torch.autograd.grad(ll, p, create_graph=True)
        for j in range(P):
            (row,) = torch.autograd.grad(g1[j], p, retain_graph=(j < P - 1))
            Hsum[j] += row.detach()
        del ll, g1
    return Hsum / N


# --------------------------------------------------------------------------- #
# Gauge-aware pseudo-inverse
# --------------------------------------------------------------------------- #
def _gauge_pinv(F, rtol):
    """Symmetric-PSD pseudo-inverse: drop singular values ≤ ``rtol·σ_max`` — the
    exact landscape-gauge null direction, plus any genuinely data-unconstrained
    directions (knots outside the FRET-observable window).

    Uses the SVD (``gesdd``) rather than the symmetric eigensolver: the Fisher is
    deliberately rank-deficient here (the gauge is an exact null direction and
    edge-knot directions can be effectively unconstrained), and ``eigh``/``syevd``
    fails to converge on such matrices while the SVD stays robust."""
    F = 0.5 * (F + F.T)
    U, S, Vh = torch.linalg.svd(F)                        # S descending
    smax = S[0].clamp_min(torch.finfo(F.dtype).tiny)
    keep = S > rtol * smax
    inv = torch.where(keep, 1.0 / S, torch.zeros_like(S))
    cov = (Vh.T * inv) @ U.T                              # V Σ⁺ Uᵀ = pinv(F)
    return 0.5 * (cov + cov.T), int((~keep).sum())


def _floored_inverse(P, rtol):
    """Robust inverse of a (proper, near-PD) precision matrix.

    Unlike ``_gauge_pinv`` (which DROPS sub-``rtol`` directions -> zero variance, the
    right thing for the exactly-unidentified likelihood gauge), this FLOORS the
    singular values at ``rtol·σ_max``: it bounds the condition number for numerical
    safety but keeps every direction, so a weakly-but-properly-pinned posterior
    direction retains its (large but finite) variance instead of being zeroed."""
    P = 0.5 * (P + P.T)
    U, S, Vh = torch.linalg.svd(P)
    smax = S[0].clamp_min(torch.finfo(P.dtype).tiny)
    n_floored = int((S < rtol * smax).sum())
    S = torch.clamp(S, min=rtol * smax)
    cov = (Vh.T * (1.0 / S)) @ U.T
    return 0.5 * (cov + cov.T), n_floored


# --------------------------------------------------------------------------- #
# Prior Hessian  (turns the Fisher into the posterior information matrix)
# --------------------------------------------------------------------------- #
class _PriorModule(torch.nn.Module):
    """Wrap the potential so ``functional_call`` can swap ``theta`` for the whole
    ``prior_penalty`` evaluation (mirrors ``sample._NLPModule``)."""

    def __init__(self, potential):
        super().__init__()
        self.potential = potential

    def forward(self, D, grid, prior):
        return prior_penalty(self.potential, D, grid, prior)


def _prior_hessian(phi_star, potential, grid, prior, K, gauge_sd, rate_sd):
    """``d²(-log prior)/dφ²`` at ``phi_star`` — the prior's contribution to the
    posterior information matrix.

    Matches the target ``sample.build_log_prob`` (hence HMC / Laplace-IS) uses: the
    ``PriorConfig`` prior (curvature / GP / l2, via ``objective.prior_penalty``) plus
    a Gaussian gauge anchor (``gauge_sd``, pins the otherwise-flat ``mean(theta)``
    direction) and a Gaussian rate prior (``rate_sd``).  Every term is quadratic in
    ``φ`` so the Hessian is constant; it is taken by autograd double-backward for
    generality (the same machinery ``_mean_hessian`` uses).  ``gauge_sd``/``rate_sd``
    may be ``None`` to omit that term.
    """
    from torch.func import functional_call
    module = _PriorModule(potential)
    log_rates0 = phi_star[K + 1:K + 5].detach()

    def neg_log_prior(phi):
        theta = phi[:K]
        D = torch.exp(phi[K])
        log_rates = phi[K + 1:K + 5]
        out = functional_call(module, {"potential.theta": theta},
                              args=(D, grid, prior))
        if gauge_sd is not None:
            out = out + gauge_penalty_from_offset(theta.mean(), gauge_sd)
        if rate_sd is not None:
            out = out + 0.5 * (((log_rates - log_rates0) / rate_sd) ** 2).sum()
        return out

    H = torch.autograd.functional.hessian(neg_log_prior, phi_star.detach())
    return 0.5 * (H + H.T)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def cramer_rao_bound(
    batch,
    grid: torch.Tensor,
    potential,
    D,
    rates: EffectiveRates,
    C: torch.Tensor,
    R0: float,
    *,
    prior: "PriorConfig | None" = None,
    gauge_sd: "float | None" = 1.0,
    rate_sd: "float | None" = 1.0,
    jitter: float = 1e-12,
    score_chunk: int = 32,
    propagate_dtype: "torch.dtype | None" = None,
    p0: torch.Tensor | None = None,
    gauge_rtol: float = 1e-8,
    validate: bool = False,
) -> CRBResult:
    """Cramér–Rao bound on ``[landscape knots | D | photophysics rates]``.

    With ``prior=None`` (default) this is the pure-likelihood CRB: ``cov = pinv(F_N)``
    with the exactly-unidentified landscape gauge (and any data-unconstrained knots)
    dropped by the pseudo-inverse.

    With a ``prior`` (a ``PriorConfig`` with ``gp_sigma`` set, as for the sampler) the
    prior's Hessian is added to the Fisher, giving the **posterior information matrix**
    ``F_N + H_prior`` — i.e. the analytic Laplace precision the ``sample`` / ``laplace_is``
    modules expand around.  ``cov``/``sigma`` are then the **posterior** covariance/σ
    (the prior regularises the soft edge-knot and gauge directions, so the result is
    proper and finite on every parameter — no dropped directions).  The added prior
    terms match ``sample.build_log_prob`` exactly: the ``PriorConfig`` prior plus a
    Gaussian gauge anchor (``gauge_sd``) and Gaussian rate prior (``rate_sd``); set
    either to ``None`` to omit it.  Evaluate at a MAP (``dfl.fit`` output) for the
    Laplace interpretation; at the truth it is the posterior information at truth.

    Parameters
    ----------
    batch : dfl.simulate.Batch
        The traces (padded ``ipt``/``colors``/``mask``), on ``grid``'s device.
    grid : [G] tensor
        Spatial grid (``GridConfig(...).build(device)``).
    potential : SplinePotential
        The landscape whose knot heights (``.theta``) are the expansion point
        (ground truth for a CRB, or the MAP for a posterior/Laplace covariance).
    D : scalar tensor / float
        Diffusion coefficient (nm²/ms) at the expansion point.
    rates : EffectiveRates
        Photophysics (``a_g, a_r, bg_g, bg_r`` in kHz) at the expansion point.
    C : [2,2] tensor, R0 : float
        Fixed crosstalk / Förster radius (as in ``fit`` / ``marginal_loglik_batch``).
    prior : PriorConfig | None
        If given, include the prior's Hessian so ``cov`` is the posterior covariance
        (Laplace precision ``F_N + H_prior``) rather than the pure CRB.  Needs a
        proper landscape prior (``gp_sigma`` set) for a finite result on the gauge.
    gauge_sd, rate_sd : float | None
        Std of the Gaussian gauge anchor / rate prior added alongside ``prior``
        (match the values used in the fit / sampler).  ``None`` omits the term.
        Ignored when ``prior is None``.
    validate : bool
        Also compute the observed information ``E[-H]`` and report the
        information-matrix-identity relative-Frobenius agreement (diagnostic).

    Returns
    -------
    CRBResult
        Fisher / per-trace Fisher / covariance matrices, per-parameter σ (inference
        params and physical units), parameter names, and dataset sizes.  When ``prior``
        is given, ``posterior_precision`` holds ``F_N + H_prior`` and
        ``prior_included`` is ``True``.
    """
    device = grid.device
    B, b0 = _knot_basis(potential, grid)
    B = B.to(device=device, dtype=DTYPE)
    b0 = b0.to(device=device, dtype=DTYPE)
    K = B.shape[1]
    dx = float(grid[1] - grid[0]) if grid.shape[0] > 1 else 1.0

    phi = _pack_truth_phi(potential, D, rates, device)

    ipt = batch.ipt.to(device)
    colors = batch.colors.to(device)
    mask = batch.mask.to(device)
    N = ipt.shape[0]

    scores = _per_trace_scores(phi, ipt, colors, mask, B, b0, grid, C, R0, dx,
                               jitter, score_chunk, propagate_dtype, p0)  # [N,P]

    if not torch.isfinite(scores).all():
        n_bad = int((~torch.isfinite(scores).all(dim=1)).sum())
        u_max = float((B @ phi[:K] + b0).max() - (B @ phi[:K] + b0).min())
        raise ValueError(
            f"per-trace scores are non-finite for {n_bad}/{N} traces "
            f"(U spans {u_max:.0f} kT on the grid). This is an instability in the "
            f"likelihood's eigendecomposition gradient when the grid extends into a "
            f"high-potential tail. Narrow `grid` to the data-visited region so U "
            f"stays moderate (≲ 10 kT) on the grid, then retry."
        )

    fisher = scores.T @ scores                            # F_N = Σ_i s_i s_iᵀ
    fisher = 0.5 * (fisher + fisher.T)
    fisher_per_trace = fisher / N

    if prior is not None:
        # posterior information = Fisher + prior Hessian (the Laplace precision).
        Hprior = _prior_hessian(phi, potential, grid, prior, K, gauge_sd, rate_sd)
        posterior_precision = 0.5 * ((fisher + Hprior) + (fisher + Hprior).T)
        cov, null_dim = _floored_inverse(posterior_precision, gauge_rtol)
    else:
        posterior_precision = None
        cov, null_dim = _gauge_pinv(fisher, gauge_rtol)
    sigma = torch.sqrt(torch.diag(cov).clamp_min(0.0))

    param_names = [f"knot_{i}" for i in range(K)] + ["logD"] + _RATE_LOG_NAMES

    sig = sigma.detach().cpu()
    D_true = float(torch.as_tensor(D))
    sigma_physical = {
        "knots": sig[:K].numpy(),                         # kT (per knot)
        "D": D_true * float(sig[K]),                      # nm²/ms
        "a_g": float(rates.a_g) * float(sig[K + 1]),      # kHz
        "a_r": float(rates.a_r) * float(sig[K + 2]),
        "bg_g": float(rates.bg_g) * float(sig[K + 3]),
        "bg_r": float(rates.bg_r) * float(sig[K + 4]),
    }

    rel_frob = None
    if validate:
        EH = _mean_hessian(phi, ipt, colors, mask, B, b0, grid, C, R0, jitter,
                           propagate_dtype, p0)
        neg_EH = (-EH).detach()
        F1 = fisher_per_trace.detach()
        rel_frob = float(torch.linalg.norm(F1 - neg_EH) /
                         torch.linalg.norm(neg_EH).clamp_min(1e-30))

    n_photons = int(batch.lengths.sum()) if hasattr(batch, "lengths") else int(mask.sum())
    return CRBResult(
        fisher=fisher, fisher_per_trace=fisher_per_trace, cov=cov, sigma=sigma,
        param_names=param_names, sigma_physical=sigma_physical,
        n_traces=int(N), n_photons=n_photons, null_dim=null_dim,
        info_matrix_rel_frob=rel_frob,
        posterior_precision=posterior_precision, prior_included=(prior is not None),
    )
