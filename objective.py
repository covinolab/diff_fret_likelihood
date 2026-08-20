from __future__ import annotations

import torch

from .config import PriorConfig
from .forward import marginal_loglik_batch


def _scalar_zero(ref: torch.Tensor) -> torch.Tensor:
    """A 0-d zero matching ``ref``'s dtype/device (empty penalty terms)."""
    return torch.zeros((), dtype=ref.dtype, device=ref.device)


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
    penalty value at a reference landscape) before comparing.  Non-smooth at d2=0
    (autograd yields a subgradient there).
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


def bg_penalty(rates, prior: PriorConfig) -> torch.Tensor:
    """Gamma prior on the background rates, from an independent calibration.

    A background measured by counting is Gamma-distributed *exactly*: ``N`` photons in a
    blank window of length ``T`` give ``p(beta) ~ beta**N exp(-T beta)``.

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

    A thin wrapper over ``gauge_offset_from_theta``: with the spline the only
    parameterisation, ``mean(theta)`` *is* the exact flat direction, so anchoring it
    costs nothing on the identified shape.  ``grid`` is unused and kept only because
    it is part of the signature every caller already passes.
    """
    return gauge_offset_from_theta(potential.theta)


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

    ``D`` is currently unused: its only consumer was the ``logD`` prior, removed in
    0.3.0.  The parameter stays because ``fisher._PriorModule`` and ``infer.fit_multi``
    pass it positionally, and because a prior on ``D`` is the obvious thing to add back
    here if one is ever wanted.

    A ``None`` prior is numerically identical to a ``PriorConfig`` with every term
    off (``curvature_weight=0`` and no ``bg_*_mean``), but skips the (zero-weighted)
    curvature evaluation, so it is both cleaner and slightly cheaper.
    """
    if prior is None:
        return _scalar_zero(grid)

    reg = _scalar_zero(grid)
    if prior.curvature_weight:
        reg = reg + prior.curvature_weight * curvature_penalty_spline(
            potential.theta, potential.knots_x,
            norm=getattr(prior, "curvature_norm", "l2"),
        )
    if prior.bg_g_mean is not None or prior.bg_r_mean is not None:
        # Fail loudly rather than silently skipping.  The precedent: the logD prior
        # (removed in 0.3.0) sat dead for weeks because a term could go missing without
        # anything complaining.  A caller that configures a bg prior but cannot supply
        # the rates is asking for something this function cannot deliver.
        if rates is None:
            raise ValueError(
                "prior_penalty: a background prior is configured (bg_g_mean/bg_r_mean) "
                "but `rates` was not passed, so the term cannot be evaluated. Call "
                "prior_penalty(..., rates=rates), or clear the bg means."
            )
        reg = reg + bg_penalty(rates, prior)
    return reg


def neg_log_posterior(
    ipt, colors, mask, potential, D, rates, grid, C, R0, prior: PriorConfig | None,
    p0=None, compile_mode=None, propagate_dtype=None,
) -> torch.Tensor:
    """``-loglik + prior_penalty`` (marginal-based).  Scalar tensor.

    With ``prior=None`` this is the pure negative log-likelihood, i.e. a true MLE
    objective (see ``prior_penalty``).  Otherwise the curvature prior acts on the knot
    heights and the background prior on ``rates``.  ``compile_mode`` /
    ``propagate_dtype`` are forwarded to ``marginal_loglik_batch`` (defaults -> eager
    float64).
    """
    ll = marginal_loglik_batch(
        ipt, colors, mask, potential, D, rates, grid, C, R0, p0=p0,
        compile_mode=compile_mode, propagate_dtype=propagate_dtype,
    )
    return -ll + prior_penalty(potential, D, grid, prior, rates=rates)
