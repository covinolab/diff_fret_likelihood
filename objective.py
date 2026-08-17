"""Objectives, priors and regularisers.

* ``curvature_penalty`` -- discrete-curvature smoothness prior on ``u`` (SPEC 9).
* ``prior_penalty`` -- the single aggregated ``-log prior`` term (``None`` -> MLE).
* ``neg_log_posterior`` -- marginal-likelihood-based training objective.
"""

from __future__ import annotations

import math

import torch

from . import potential as dfl_potential

from .config import PriorConfig
from .forward import marginal_loglik_batch, _BasePotential_on_grid
from .generator import stationary


def _scalar_zero(ref: torch.Tensor) -> torch.Tensor:
    """A 0-d zero matching ``ref``'s dtype/device (empty penalty terms)."""
    return torch.zeros((), dtype=ref.dtype, device=ref.device)


def curvature_penalty(u_grid: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
    """Grid-invariant roughness ``~= integral (u''(x))^2 dx``.

    The discrete 2nd difference obeys ``d2_i = u_{i+1}-2u_i+u_{i-1} ~= u''(x_i) dx^2``,
    so ``sum_i d2_i^2 ~= dx^4 sum_i (u''_i)^2`` and the *resolution-independent*
    functional ``integral (u'')^2 dx = dx sum_i (u''_i)^2`` equals ``sum_i d2_i^2 / dx^3``.
    Dividing by ``dx**3`` therefore makes ``curvature_weight`` mean the same physical
    smoothing across grid resolutions (fixes the D-drifts-with-grid coupling).

    ``dx`` defaults to 1.0 so legacy positional calls keep the old (grid-dependent)
    behaviour; the objective always passes the real grid spacing.
    """
    d2 = u_grid[2:] - 2.0 * u_grid[1:-1] + u_grid[:-2]
    return (d2 ** 2).sum() / (dx ** 3)


def curvature_penalty_spline(theta: torch.Tensor, knots_x: torch.Tensor,
                             norm: str = "l2") -> torch.Tensor:
    """Roughness of the *knot heights* — the actual free params.
    Second difference on (possibly non-uniform) knots ~ theta'' at each interior knot.
    This is D2 @ theta; the penalty is ||D2 @ theta||^2 = theta^T (D2^T D2) theta,
    i.e. a GMRF prior with precision rho * D2^T D2. Grid-independent by construction:
    it never touches the fine grid, only the K knots that carry the DOF.

    ``norm="l1"`` switches ``sum d2^2`` -> ``sum |d2|``, i.e. a Laplace (total-variation)
    prior on the curvature instead of a Gaussian one.  The L2 form shrinks ALL curvature
    quadratically, so it cannot tell one large real feature (a barrier, an intermediate
    well) from many small noise wiggles and flattens both -- the same reason Gaussian
    smoothing blurs edges in image denoising.  The heavy-tailed L1 form suppresses many
    small values while leaving a few large ones nearly untouched, which matches the
    structure of a real landscape.  NOTE: the two norms are on different scales, so
    ``curvature_weight`` does NOT carry over between them -- calibrate it (e.g. match the
    penalty value at a reference landscape) before comparing.  Non-smooth at d2=0; Adam
    and the homotopy blur both handle the subgradient fine.
    """
    x = knots_x
    h_left  = x[1:-1] - x[:-2]      # (K-2,)
    h_right = x[2:]   - x[1:-1]
    # non-uniform 2nd-difference: theta''_i ~ 2/(hl+hr) * (theta_{i+1}/hr - theta_i(1/hl+1/hr) + theta_{i-1}/hl)
    d2 = 2.0 * ( theta[2:]   / (h_right * (h_left + h_right))
               - theta[1:-1] / (h_left * h_right)
               + theta[:-2]  / (h_left  * (h_left + h_right)) )
    if norm == "l1":
        return d2.abs().sum()
    if norm != "l2":
        raise ValueError(f"curvature norm must be 'l2' or 'l1', got {norm!r}")
    return (d2 ** 2).sum()


def logD_penalty(D: torch.Tensor, prior: PriorConfig) -> torch.Tensor:
    if prior.logD_mean is None:
        return _scalar_zero(D)
    logD = torch.log(D)
    return 0.5 * ((logD - prior.logD_mean) / prior.logD_std) ** 2


def bg_penalty(rates, prior: PriorConfig) -> torch.Tensor:
    """Gamma prior on the background rates, from an independent calibration.

    A background measured by counting is Gamma-distributed *exactly*: ``N`` photons in a
    blank window of length ``T`` give ``p(beta) ~ beta**N exp(-T beta)``.  So this is not
    a distributional assumption layered on the calibration -- it IS the calibration's own
    posterior.  (``logD_penalty`` above is a Gaussian instead, deliberately: a diffusion
    coefficient is not a counted rate.)

    The fit optimises ``ln bg``, so the Gamma is written as a proper density in THAT
    coordinate (the Jacobian is kept).  With ``r = bg / mean`` and ``k = (mean / sd)**2``
    the negative log density is, up to a constant,

        k * (r - ln r - 1)

    which is the Itakura-Saito / KL form and has three properties worth relying on:

    * it is exactly ``0`` at ``bg = mean`` and strictly positive either side, so the
      step-0 loss offset is exactly the penalty at the init;
    * its mode sits exactly at ``mean`` -- keeping the Jacobian is what puts it there;
      a Gamma density in ``bg`` would peak at ``mean * (1 - (sd/mean)**2)``;
    * its curvature in ``ln bg`` at the mode is exactly ``k``, i.e. the width in ``bg`` is
      exactly ``sd`` -- so the config's kHz error bar means what it says.

    ``k = (mean/sd)**2`` is the **equivalent photon count** behind the calibration: a
    blank window yielding ``N`` counts has ``sd/mean = 1/sqrt(N)``, so a +/-10% error bar
    is 100 counts.  Unlike a Gaussian on ``ln bg`` this carries the correct Gamma skew --
    a background much HIGHER than measured is strongly excluded (you would have counted
    more photons), a lower one much less so.
    """
    out = None
    for mean, sd, bg in ((prior.bg_g_mean, prior.bg_g_sd, rates.bg_g),
                         (prior.bg_r_mean, prior.bg_r_sd, rates.bg_r)):
        if mean is None:
            continue
        k = (mean / sd) ** 2
        r = bg / mean
        term = k * (r - torch.log(r) - 1.0)
        out = term if out is None else out + term
    return _scalar_zero(rates.bg_g) if out is None else out


def _gp_corr(x_ctrl: torch.Tensor, lengthscale: float, kernel: str) -> torch.Tensor:
    """Stationary correlation matrix ``[n,n]`` on control points (unit variance)."""
    r = (x_ctrl[:, None] - x_ctrl[None, :]).abs() / lengthscale
    if kernel == "rbf":
        return torch.exp(-0.5 * r ** 2)
    if kernel == "matern32":
        c = math.sqrt(3.0)
        return (1.0 + c * r) * torch.exp(-c * r)
    if kernel == "matern52":
        c = math.sqrt(5.0)
        return (1.0 + c * r + (5.0 / 3.0) * r ** 2) * torch.exp(-c * r)
    raise ValueError(f"unknown gp_kernel {kernel!r}")


def _interp(x_src: torch.Tensor, y_src: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of ``y_src`` (on ascending ``x_src``) at ``x_query``.

    Used only for the fixed GP prior mean, so ``y_src`` is detached (no grad).
    """
    x_src = x_src.detach()
    y_src = y_src.detach().to(dtype=x_query.dtype, device=x_query.device)
    n = x_src.shape[0]
    idx = torch.searchsorted(x_src, x_query).clamp(1, n - 1)
    x0, x1 = x_src[idx - 1], x_src[idx]
    y0, y1 = y_src[idx - 1], y_src[idx]
    denom = (x1 - x0).clamp_min(torch.finfo(x_query.dtype).tiny)
    return y0 + (x_query - x0) / denom * (y1 - y0)


def gp_penalty(potential, grid: torch.Tensor, prior: PriorConfig) -> torch.Tensor:
    """``-log`` of a proper GP prior over ``U(x)`` (up to a fixed constant).

    Evaluated on ``gp_n_ctrl`` control points (well-conditioned, unlike the full
    fine grid); the residual is MEAN-CENTERED so the prior is gauge-invariant
    (matches the constant-shift invariance of the likelihood / curvature term)
    and shrinks well-depths/barrier-heights symmetrically with SD ``gp_sigma``.
    Works for both spline (proper prior on ``theta``) and MLP (functional prior).

    Note: the constant ``0.5 logdet(2 pi K)`` is dropped -- valid because the GP
    hyperparameters are FIXED during a fit/chain, so it shifts the loss by a
    constant (raw loss values are therefore not comparable across GP settings).
    """
    if prior.gp_sigma is None:
        return _scalar_zero(grid)
    n = int(min(max(prior.gp_n_ctrl, 4), grid.shape[0]))
    x_ctrl = torch.linspace(
        float(grid.min()), float(grid.max()), n, dtype=grid.dtype, device=grid.device
    )
    u_ctrl = potential(x_ctrl)  # raw; grads flow to theta / net params
    if prior.gp_mean is not None:
        u_ctrl = u_ctrl - _interp(grid, prior.gp_mean, x_ctrl)
    r = u_ctrl - u_ctrl.mean()  # gauge-invariant (mean-centered, smooth)

    Kc = _gp_corr(x_ctrl, prior.gp_lengthscale, prior.gp_kernel)
    eye = torch.eye(n, dtype=grid.dtype, device=grid.device)
    # Fixed jitter; deterministic escalation only if Cholesky fails (inputs are
    # constant within a fit, so the landed jitter is stationary across calls).
    jit = float(prior.gp_jitter)
    for _ in range(6):
        try:
            L = torch.linalg.cholesky(Kc + jit * eye)
            break
        except RuntimeError:
            jit *= 10.0
    else:  # pragma: no cover - pathological kernel
        raise RuntimeError("GP kernel Cholesky failed even after jitter escalation")

    z = torch.linalg.solve_triangular(L, r.unsqueeze(-1), upper=False)
    return 0.5 * (z ** 2).sum() / (prior.gp_sigma ** 2)


def gauge_offset_from_theta(theta: torch.Tensor) -> torch.Tensor:
    """The pure-gauge offset coordinate of a spline landscape, from its knots alone.

    The marginal likelihood is exactly invariant to ``U -> U + const``.  For a
    ``SplinePotential`` (natural cubic, ``u = M_val @ theta`` with ``M_val`` a
    partition of unity) that flat direction is ``(1,...,1)`` in ``theta``-space, so
    ``mean(theta)`` is the offset whose gradient points *exactly* along it -- anchoring
    it therefore pins the gauge with ZERO bias on the identified shape (well depths,
    barriers, ``D``).

    Takes the knot tensor rather than the potential object because the two places that
    need it most -- ``sample.build_log_prob`` and ``fisher._penalty_hessian`` -- hold a
    swapped-in ``theta`` slice, not a live module.  This is the ONE definition of the
    offset; ``gauge_offset`` below is the potential-level convenience wrapper.
    """
    return theta.mean()


def gauge_offset(potential, grid: torch.Tensor) -> torch.Tensor:
    """The pure-gauge offset coordinate of ``U``, from a potential object.

    Delegates to ``gauge_offset_from_theta`` for a spline (the exact flat direction).

    For the MLP (no knots) it falls back to ``mean(U over grid)``, which is NOT an exact
    flat direction in parameter space -- an MLP cannot in general realise ``U + const``
    by moving along one parameter direction -- so anchoring it can bias the recovered
    shape slightly.  ``infer.fit`` accepts that trade for the sake of a converging
    optimiser; ``sample.build_log_prob`` instead drops the anchor entirely for
    non-splines.  That asymmetry is deliberate, and moot in practice: no live script
    fits an MLP, and ``fisher.cramer_rao_bound`` rejects one outright.
    """
    if hasattr(potential, "theta"):          # spline: exact flat direction
        return gauge_offset_from_theta(potential.theta)
    return potential.on_grid(grid).mean()     # MLP fallback (approximate; see above)


def gauge_penalty_from_offset(offset: torch.Tensor, gauge_sd: float = 1.0) -> torch.Tensor:
    """Gaussian anchor ``0.5 (offset / gauge_sd)^2`` on the pure-gauge offset."""
    return 0.5 * (offset / gauge_sd) ** 2


def gauge_penalty(potential, grid: torch.Tensor, gauge_sd: float = 1.0) -> torch.Tensor:
    """Anchor the pure-gauge offset of ``U`` toward zero (``mean(theta)=0`` gauge).

    Added to the *fit* objective only (see ``infer.fit``) so the otherwise-flat offset
    direction has a defined gradient/curvature and converges reproducibly, without
    biasing any identified quantity.  Kept OUT of ``prior_penalty``/``neg_log_posterior``
    so the sampler (which adds its own gauge anchor) does not double-count it.
    """
    return gauge_penalty_from_offset(gauge_offset(potential, grid), gauge_sd)


def prior_penalty(potential, D, grid: torch.Tensor, prior: PriorConfig | None,
                  *, rates=None) -> torch.Tensor:
    """Total prior / regulariser penalty ``= -log prior`` (up to a constant).

    This is the SINGLE place the prior enters the objective; ``neg_log_posterior``
    is just ``-loglik + prior_penalty(...)``.  ``prior=None`` means a **pure MLE**
    fit -- no regularisation at all -- and returns exactly ``0``.

    A ``None`` prior is numerically identical to a ``PriorConfig`` with every term
    off (``curvature_weight=0``, ``logD_mean=None``, ``gp_sigma=None``,
    ``l2_weight=0``), but skips the (zero-weighted) curvature evaluation, so it is
    both cleaner and slightly cheaper.
    """
    if prior is None:
        return _scalar_zero(grid)

    reg = _scalar_zero(grid)
    if prior.curvature_weight:
        if isinstance(potential, dfl_potential.SplinePotential):
            reg = reg + prior.curvature_weight * curvature_penalty_spline(
                potential.theta, potential.knots_x,
                norm=getattr(prior, "curvature_norm", "l2"),
            )
        else:
            u_grid = _BasePotential_on_grid(potential, grid)
            dx = float(grid[1] - grid[0]) if grid.shape[0] > 1 else 1.0
            reg = reg + prior.curvature_weight * curvature_penalty(u_grid, dx)
    if prior.logD_mean is not None:
        reg = reg + logD_penalty(D, prior)
    if prior.bg_g_mean is not None or prior.bg_r_mean is not None:
        # Fail loudly rather than silently skipping: the logD prior was dead for weeks
        # because a term could go missing without anything complaining (see
        # tests/test_logD_prior.py).  A caller that configures a bg prior but cannot
        # supply the rates is asking for something this function cannot deliver.
        if rates is None:
            raise ValueError(
                "prior_penalty: a background prior is configured (bg_g_mean/bg_r_mean) "
                "but `rates` was not passed, so the term cannot be evaluated. Call "
                "prior_penalty(..., rates=rates), or clear the bg means."
            )
        reg = reg + bg_penalty(rates, prior)
    if prior.gp_sigma is not None:
        reg = reg + gp_penalty(potential, grid, prior)
    if prior.l2_weight:
        pnorm = sum((p ** 2).sum() for p in potential.parameters())
        reg = reg + prior.l2_weight * pnorm
    return reg


def neg_log_posterior(
    ipt, colors, mask, potential, D, rates, grid, C, R0, prior: PriorConfig | None,
    p0=None, compile_mode=None, propagate_dtype=None,
) -> torch.Tensor:
    """``-loglik + prior_penalty`` (marginal-based).  Scalar tensor.

    With ``prior=None`` this is the pure negative log-likelihood, i.e. a true MLE
    objective (see ``prior_penalty``).  Otherwise the curvature and ``logD`` priors
    act on ``u_grid``/``D`` only.  ``compile_mode`` / ``propagate_dtype`` are
    forwarded to ``marginal_loglik_batch`` (defaults -> eager float64).
    """
    ll = marginal_loglik_batch(
        ipt, colors, mask, potential, D, rates, grid, C, R0, p0=p0,
        compile_mode=compile_mode, propagate_dtype=propagate_dtype,
    )
    return -ll + prior_penalty(potential, D, grid, prior, rates=rates)
