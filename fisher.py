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
all-knots-equal direction is an exact null-space of the Fisher matrix.  The Gaussian
gauge anchor (``gauge_sd``) that ``infer.fit`` and the sampler apply is therefore
**always** added here too, independently of whether a ``PriorConfig`` prior is in
play: it is a choice of coordinates, not a belief about physics, and without it the
matrix to invert is genuinely singular and only an SVD threshold stands between the
caller and a meaningless number.  With it, the information matrix is normally
positive definite and is inverted EXACTLY by Cholesky — no threshold anywhere.

What the anchor costs, precisely.  ``H_gauge = 1·1ᵀ/(gauge_sd²·K²)`` lives entirely
in the gauge direction ``v = (1_K, 0…0)/√K``, while ``F_N`` lives entirely in
``v^⊥``.  They block-diagonalise against the same split, so

    cov = pinv(F_N) + gauge_sd²·K·v vᵀ

exactly.  ``v`` is zero on ``logD`` and on all four rates, so **σ_D and σ_rates are
untouched**, and so is every gauge-blind landscape functional (any ``d`` with
``dᵀ1 = 0``, e.g. a barrier height ``U(x_bar) − U(x_well)``).  What does move is the
per-knot σ: each knot's variance is larger by exactly ``gauge_sd²`` than it would be
in the strict sum-to-zero gauge.  ``cov`` is reported raw (no projection), so a
per-knot σ is a σ *in the weakly-anchored mean(theta) gauge* — which is the honest
Laplace posterior, but means a per-knot number is only interpretable together with
``gauge_sd``.  (``recovered_potential`` reports the grid-mean-zero gauge; it differs
by a constant only, so the landscape *shape* is unaffected.)

Identifiability: the anchor pins exactly ONE direction, the gauge.  Knots outside the
FRET-observable window are separately near-unconstrained by the data, and the anchor
does nothing for them — ``F_N + H_gauge`` can still be singular, in which case the
Cholesky refuses, the floored SVD takes over and ``null_dim`` reports how many
directions had to be floored (``0`` = an exact inverse, the normal case).  A σ in a
floored direction is a *lower bound*; inspect the Fisher diagonal to see which knots
are informed.  Note the curvature prior cannot fix this on its own: it is the improper
thin-plate limit, so it constrains roughness but leaves the flat directions free.  The
practical remedy is to narrow ``grid`` to the region the data actually inform (the
published fits reach ``null_dim = 0`` that way).

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

import warnings
from dataclasses import dataclass

import torch

from .config import DTYPE, PriorConfig
from .forward import build_propagator_from_u, marginal_loglik_batch
from .generator import stationary, min_gauge
from .objective import prior_penalty, gauge_penalty_from_offset, gauge_offset_from_theta
from .photophysics import EffectiveRates

_RATE_NAMES = ("a_g", "a_r", "bg_g", "bg_r")
_RATE_LOG_NAMES = [f"log_{n}" for n in _RATE_NAMES]


@dataclass
class CRBResult:
    """Cramér–Rao bound for one dataset at the ground-truth parameters.

    ``fisher``           : [P,P] total Fisher information ``F_N`` for the dataset.
                           The pure likelihood information — never includes the anchor
                           or the prior, so the gauge stays an exact null direction here.
    ``fisher_per_trace`` : [P,P] ``F_N / N`` (Fisher is additive over traces).
    ``cov``              : [P,P] ``inv(posterior_precision)`` — the bound WITH the gauge
                           anchor always included, reported raw (no gauge projection).
                           Landscape σ are therefore in the weakly-anchored
                           ``mean(theta)`` gauge: each knot's variance exceeds the strict
                           sum-to-zero value by exactly ``gauge_sd²``.  σ_D, σ_rates and
                           every gauge-blind functional are unaffected — see the module
                           docstring for the exact decomposition.
    ``sigma``            : [P] ``sqrt(diag(cov))`` in the inference parameterisation
                           (kT knots, lnD, ln-rates).
    ``param_names``      : [P] names, ``["knot_0",…,"logD","log_a_g",…]``.
    ``sigma_physical``   : dict of σ in physical units — ``knots`` (kT, per knot),
                           ``D`` (nm²/ms), ``a_g/a_r/bg_g/bg_r`` (kHz), via the delta method.
    ``n_traces``, ``n_photons`` : dataset size.
    ``null_dim``         : # of floored directions.  ``0`` (the normal case) means the
                           information matrix was positive definite and inverted EXACTLY
                           by Cholesky, with no threshold involved.  ``> 0`` means it was
                           genuinely singular and that many directions had to be floored
                           — a real unidentifiability (typically knots the data never
                           see), not a numerical cutoff.  The gauge itself no longer
                           shows up here: the anchor pins it.
    ``info_matrix_rel_frob`` : if ``validate=True``, ‖Cov(score) − (−E[H])‖ / ‖E[H]‖
                           (information-matrix identity check); else ``None``.
    ``posterior_precision`` : the matrix that was actually inverted to get ``cov`` —
                           ``F_N`` plus the gauge anchor's Hessian, plus the prior's and
                           the rate prior's if a ``prior`` was passed (the analytic
                           Laplace precision).  ``None`` only in the fully unpenalised
                           case (``prior=None`` AND ``gauge_sd=None``), where ``cov``
                           falls back to the bare pseudo-inverse of ``F_N``.
    ``prior_included``   : whether a ``PriorConfig`` contributed to ``cov``/``sigma``.
                           The gauge anchor is not a prior in this sense and is not
                           reflected here — it is on unless ``gauge_sd=None``.
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
    if not hasattr(potential, "_basis"):
        raise NotImplementedError(
            "cramer_rao_bound requires a SplinePotential (a landscape linear in "
            f"its knot heights); got {type(potential).__name__}. The whole scoring "
            "parameterisation below is the knot vector, so a potential without a "
            "knot basis has no dense CRB here."
        )
    B = potential._basis(grid)                           # on_grid = B @ theta
    b0 = torch.zeros(grid.shape[0], dtype=B.dtype, device=B.device)
    return B, b0


def _pack_truth_phi(potential, D, rates, device):
    """Build ``φ* = [theta | lnD | ln rates]`` from the ground-truth objects."""
    theta = potential.theta.detach().to(device)
    K = theta.numel()
    phi = torch.empty(K + 5, dtype=DTYPE, device=device)
    phi[:K] = theta
    phi[K] = torch.log(torch.as_tensor(D, dtype=DTYPE, device=device))
    for i, nm in enumerate(_RATE_NAMES):
        phi[K + 1 + i] = torch.log(
            torch.as_tensor(getattr(rates, nm), dtype=DTYPE, device=device)
        )
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
    u = min_gauge(u)                          # exp(-u) overflow guard (as forward does)
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
def _svd_inverse(M, rtol, *, floor):
    """Symmetric SVD (pseudo-)inverse of ``M``, returning ``(cov, n_small)``.

    Uses the SVD (``gesdd``) rather than the symmetric eigensolver: these matrices
    are deliberately rank-deficient / ill-conditioned (the gauge is an exact null
    direction and edge-knot directions can be effectively unconstrained), and
    ``eigh``/``syevd`` fails to converge on them while the SVD stays robust.

    Singular values ``≤ rtol·σ_max`` are treated as null. ``floor=False`` (the
    pseudo-inverse) DROPS them -> zero variance, the right thing for the exactly-
    unidentified likelihood gauge; ``floor=True`` instead FLOORS them at ``rtol·σ_max``,
    bounding the condition number for numerical safety while keeping every direction so
    a weakly-but-properly-pinned posterior direction retains its (large but finite)
    variance instead of being zeroed.  ``n_small`` counts the affected directions."""
    M = 0.5 * (M + M.T)
    U, S, Vh = torch.linalg.svd(M)                        # S descending
    smax = S[0].clamp_min(torch.finfo(M.dtype).tiny)
    small = S <= rtol * smax
    if floor:
        inv = 1.0 / torch.clamp(S, min=rtol * smax)
    else:
        inv = torch.where(small, torch.zeros_like(S), 1.0 / S)
    cov = (Vh.T * inv) @ U.T                              # V Σ⁺ Uᵀ
    return 0.5 * (cov + cov.T), int(small.sum())


def _psd_inverse(M, rtol):
    """Inverse of a symmetric POSITIVE DEFINITE ``M``, returning ``(cov, n_floored)``.

    Tries a Cholesky factorisation first.  If it succeeds, ``M`` is positive definite,
    the inverse is exact (no threshold anywhere) and ``n_floored`` is ``0``.  If it
    fails, ``M`` really is singular and we fall back to ``_svd_inverse(..., floor=True)``,
    which reports how many directions it had to floor.

    Why not just use ``_svd_inverse`` everywhere: its cutoff is RELATIVE, ``rtol·σ_max``,
    so it silently assumes ``cond(M) < 1/rtol``.  A posterior precision breaks that
    assumption as soon as the data are good -- ``σ_max`` tracks the photon count while
    ``σ_min`` sits at the prior scale -- and when it does, flooring clamps a precision UP,
    hence a variance DOWN.  The error is therefore ONE-SIDED: it can only ever make a
    reported σ too small, i.e. over-confident, and nothing in the returned values says so
    beyond ``n_floored``.  Cholesky removes the assumption: it either inverts exactly or
    refuses, and the factorisation succeeding is itself the proof that every direction is
    pinned (by curvature prior, gauge anchor and data together -- no one of them does it
    alone).
    """
    M = 0.5 * (M + M.T)
    try:
        L = torch.linalg.cholesky(M)
    except Exception:                                     # not PD -> genuinely singular
        return _svd_inverse(M, rtol, floor=True)
    cov = torch.cholesky_inverse(L)
    return 0.5 * (cov + cov.T), 0


# --------------------------------------------------------------------------- #
# Prior Hessian  (turns the Fisher into the posterior information matrix)
# --------------------------------------------------------------------------- #
class _PriorModule(torch.nn.Module):
    """Wrap the potential so ``functional_call`` can swap ``theta`` for the whole
    ``prior_penalty`` evaluation (mirrors ``sample._NLPModule``)."""

    def __init__(self, potential):
        super().__init__()
        self.potential = potential

    def forward(self, D, grid, prior, rates=None):
        return prior_penalty(self.potential, D, grid, prior, rates=rates)


def _penalty_hessian(phi_star, potential, grid, prior, K, gauge_sd):
    """``d²(-log penalty)/dφ²`` at ``phi_star`` — everything added to the Fisher.

    Two independently gated terms, and the distinction is the point:

    * the **gauge anchor** (``gauge_sd``) is a choice of coordinates, so it is included
      whenever ``gauge_sd is not None`` — *regardless of* ``prior``.  It is what makes
      the information matrix invertible at all.
    * the **``PriorConfig`` prior** (curvature and bg, via ``objective.prior_penalty``)
      is a belief about physics, so it opts in with ``prior``.

    The four log-rates are reconstructed from ``φ[K+1:K+5]`` and handed to
    ``prior_penalty`` so that a ``PriorConfig`` background prior (``bg_g_mean`` /
    ``bg_r_mean``) is differentiated here too.  Without that the posterior CRB would
    silently ignore the prior and report the far wider background-free bound.

    With a ``prior`` this reproduces the MAP objective ``infer.fit`` optimises: the
    prior penalty plus the gauge anchor, and NO prior on the photophysics rates.  Since
    0.4.0 that is also exactly the sampler's target, so all three agree on one objective.
    (An optional Gaussian prior on the four log-rates lived here until 0.4.0, to mirror
    one the sampler used to add.  Both are gone: ``a_g``/``a_r`` are pinned by the photon
    stream and the backgrounds by ``PriorConfig``'s Gamma, and every analysis had
    switched the term off explicitly anyway -- precisely so the bound describes the
    estimator that produced the MAP.)
    The Hessian is taken by autograd double-backward (the same machinery
    ``_mean_hessian`` uses).  Most terms are quadratic in ``φ`` and so contribute a
    constant Hessian, but the **bg** term is a Gamma and is NOT — so this must be
    evaluated AT the MAP, not at any convenient nearby point.

    Returns ``None`` if nothing at all is penalised (``prior is None`` and
    ``gauge_sd is None``), which the caller reads as "invert the bare Fisher".
    """
    if prior is None and gauge_sd is None:
        return None

    from torch.func import functional_call
    module = _PriorModule(potential)

    def neg_log_penalty(phi):
        theta = phi[:K]
        out = torch.zeros((), dtype=phi.dtype, device=phi.device)
        if prior is not None:
            D = torch.exp(phi[K])
            # rebuild the rates from phi so a bg prior is differentiated, not frozen
            r = torch.exp(phi[K + 1:K + 5])
            rates_phi = EffectiveRates(r[0], r[1], r[2], r[3])
            out = out + functional_call(module, {"potential.theta": theta},
                                        args=(D, grid, prior, rates_phi))
        if gauge_sd is not None:
            out = out + gauge_penalty_from_offset(
                gauge_offset_from_theta(theta), gauge_sd)
        return out

    H = torch.autograd.functional.hessian(neg_log_penalty, phi_star.detach())
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
    jitter: float = 1e-12,
    score_chunk: int = 32,
    propagate_dtype: "torch.dtype | None" = None,
    p0: torch.Tensor | None = None,
    gauge_rtol: float = 1e-8,
    validate: bool = False,
) -> CRBResult:
    """Cramér–Rao bound on ``[landscape knots | D | photophysics rates]``.

    The Gaussian gauge anchor (``gauge_sd``, the same one ``infer.fit`` and the sampler
    apply) is **always** included, whether or not a ``prior`` is given: the landscape
    offset is an exact flat direction, so without the anchor there is nothing to invert
    but a singular matrix.  With it, ``F_N + H_gauge`` is normally positive definite and
    is inverted exactly by Cholesky.  ``cov`` is reported raw, so per-knot σ carry the
    anchor's ``gauge_sd²`` of extra variance while σ_D, σ_rates and all gauge-blind
    landscape functionals are untouched — the module docstring gives the exact identity.

    With ``prior=None`` (default) this is therefore the likelihood CRB in the anchored
    gauge.  With a ``prior`` (the ``PriorConfig`` the fit used) the prior's Hessian is
    added as well, giving the **posterior information matrix** — the analytic Laplace
    precision of that MAP objective.
    ``cov``/``sigma`` are then the posterior covariance/σ, with the prior additionally
    regularising the soft edge-knot directions the anchor cannot reach.  Evaluate at a
    MAP (``dfl.fit`` output) for the Laplace interpretation; at the truth it is the
    posterior information at truth.

    Parameters
    ----------
    batch : dfl.simulate.Batch
        The traces (padded ``ipt``/``colors``/``mask``), on ``grid``'s device.
    grid : [G] tensor
        Spatial grid (``GridConfig(...).build(device)``).
    potential : SplinePotential
        The landscape whose knot heights (``.theta``) are the expansion point
        (ground truth for a CRB, or the MAP for a posterior/Laplace covariance).
        The scoring parameterisation *is* the knot vector, so this is required.
    D : scalar tensor / float
        Diffusion coefficient (nm²/ms) at the expansion point.
    rates : EffectiveRates
        Photophysics (``a_g, a_r, bg_g, bg_r`` in kHz) at the expansion point.
    C : [2,2] tensor, R0 : float
        Fixed crosstalk / Förster radius (as in ``fit`` / ``marginal_loglik_batch``).
    prior : PriorConfig | None
        If given, include the prior's Hessian so ``cov`` is the posterior covariance
        (Laplace precision) rather than the likelihood CRB.  Pass the SAME
        ``PriorConfig`` the fit used, or the bound describes a different objective than
        the one that produced the MAP.  The gauge is handled by ``gauge_sd``
        independently of this.
    gauge_sd : float | None
        Std of the Gaussian gauge anchor on ``mean(theta)``, added ALWAYS — not gated on
        ``prior``.  Match the value used in the fit / sampler (both default to ``1.0``).
        ``None`` omits it and falls back to the pseudo-inverse of the bare Fisher, which
        warns: per-knot σ then become lower bounds in a thresholded null space, and
        ``null_dim`` changes meaning.  Only pass ``None`` deliberately.
    validate : bool
        Also compute the observed information ``E[-H]`` and report the
        information-matrix-identity relative-Frobenius agreement (diagnostic).

    Returns
    -------
    CRBResult
        Fisher / per-trace Fisher / covariance matrices, per-parameter σ (inference
        params and physical units), parameter names, and dataset sizes.
        ``posterior_precision`` holds the matrix that was inverted (``F_N`` plus the
        anchor, plus the prior if given); ``prior_included`` reports whether a
        ``PriorConfig`` contributed.
    """
    if gauge_sd is None:
        warnings.warn(
            "cramer_rao_bound(gauge_sd=None): the landscape offset is an exact flat "
            "direction, so without the gauge anchor the matrix being inverted is "
            "singular and `cov` becomes a THRESHOLDED pseudo-inverse. Consequences for "
            "anything downstream: per-knot sigma are lower bounds rather than bounds, "
            "`null_dim` counts the gauge instead of real unidentifiability, and the "
            "result depends on `gauge_rtol`. Pass a gauge_sd (the fit and sampler both "
            "use 1.0) unless you specifically want the textbook unanchored CRB.",
            RuntimeWarning, stacklevel=2,
        )

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

    H = _penalty_hessian(phi, potential, grid, prior, K, gauge_sd)
    if H is not None:
        # information = Fisher + gauge anchor (always) + prior Hessian (if any).
        # Normally positive definite, so invert it EXACTLY; `_psd_inverse` falls back to
        # the floored SVD only if it genuinely is not (unseen knots, not the gauge).
        posterior_precision = 0.5 * ((fisher + H) + (fisher + H).T)
        cov, null_dim = _psd_inverse(posterior_precision, gauge_rtol)
    else:
        # gauge_sd=None and no prior: nothing pins the gauge, so it is an EXACT null
        # direction of what we invert -> drop it (pseudo-inverse).  Warned about above.
        posterior_precision = None
        cov, null_dim = _svd_inverse(fisher, gauge_rtol, floor=False)
    sigma = torch.sqrt(torch.diag(cov).clamp_min(0.0))

    param_names = [f"knot_{i}" for i in range(K)] + ["logD"] + _RATE_LOG_NAMES

    sig = sigma.detach().cpu()
    D_true = float(torch.as_tensor(D))
    sigma_physical = {
        "knots": sig[:K].numpy(),                         # kT (per knot)
        "D": D_true * float(sig[K]),                      # nm²/ms
    }
    for i, nm in enumerate(_RATE_NAMES):                  # kHz, delta method
        sigma_physical[nm] = float(getattr(rates, nm)) * float(sig[K + 1 + i])

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
