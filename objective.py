"""Objectives, priors and regularisers.

* ``curvature_penalty`` -- discrete-curvature smoothness prior on ``u`` (SPEC 9).
* ``neg_log_posterior`` -- marginal-likelihood-based training objective.
* ``complete_data_loglik`` -- SECONDARY joint objective with the path as a
  latent variable (SPEC 4.5), used for the joint-vs-marginal D-bias diagnostic.
"""

from __future__ import annotations

import math

import torch

from .config import PriorConfig
from .dynamics import em_transition_logp
from .forward import marginal_loglik_batch, _BasePotential_on_grid
from .photophysics import emission_rates, EffectiveRates


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


def logD_penalty(D: torch.Tensor, prior: PriorConfig) -> torch.Tensor:
    if prior.logD_mean is None:
        return torch.zeros((), dtype=D.dtype, device=D.device)
    logD = torch.log(D)
    return 0.5 * ((logD - prior.logD_mean) / prior.logD_std) ** 2


# ---------------------------------------------------------------------------
# Proper GP prior over U(x)  (SPEC section 9; makes the landscape posterior
# proper -- essential for HMC sampling, see sample.py)
# ---------------------------------------------------------------------------
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
        return torch.zeros((), dtype=grid.dtype, device=grid.device)
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
    L = None
    for _ in range(6):
        try:
            L = torch.linalg.cholesky(Kc + jit * eye)
            break
        except RuntimeError:
            jit *= 10.0
    if L is None:  # pragma: no cover - pathological kernel
        raise RuntimeError("GP kernel Cholesky failed even after jitter escalation")

    z = torch.linalg.solve_triangular(L, r.unsqueeze(-1), upper=False)
    return 0.5 * (z.squeeze(-1) @ z.squeeze(-1)) / (prior.gp_sigma ** 2)


def neg_log_posterior(
    ipt, colors, mask, potential, D, rates, grid, C, R0, prior: PriorConfig,
    p0=None, compile_mode=None, propagate_dtype=None,
) -> torch.Tensor:
    """``-loglik + regularisers`` (marginal-based).  Scalar tensor.

    The curvature and ``logD`` priors act on ``u_grid``/``D`` only.
    ``compile_mode`` / ``propagate_dtype`` are forwarded to
    ``marginal_loglik_batch`` (defaults -> eager float64).
    """
    ll = marginal_loglik_batch(
        ipt, colors, mask, potential, D, rates, grid, C, R0, p0=p0,
        compile_mode=compile_mode, propagate_dtype=propagate_dtype,
    )
    u_grid = _BasePotential_on_grid(potential, grid)
    dx = float(grid[1] - grid[0]) if grid.shape[0] > 1 else 1.0
    reg = prior.curvature_weight * curvature_penalty(u_grid, dx)
    reg = reg + logD_penalty(D, prior)
    if prior.gp_sigma is not None:
        reg = reg + gp_penalty(potential, grid, prior)
    if prior.l2_weight:
        pnorm = sum((p ** 2).sum() for p in potential.parameters())
        reg = reg + prior.l2_weight * pnorm
    return -ll + reg


# ---------------------------------------------------------------------------
# Secondary complete-data (joint) objective
# ---------------------------------------------------------------------------
def complete_data_loglik(
    x_path: torch.Tensor,
    times: torch.Tensor,
    colors: torch.Tensor,
    T: float,
    time_grid: torch.Tensor,
    potential,
    D: torch.Tensor,
    rates: EffectiveRates,
    C: torch.Tensor,
    R0: float,
    log_p0=None,
) -> torch.Tensor:
    """Joint log-lik of a path + photons (SPEC 4.5).

    ``x_path``   : [M+1] latent positions on ``time_grid`` (step h uniform).
    ``time_grid``: [M+1] times (ms) of the path samples, ``time_grid[0]=0``.
    ``times``/``colors`` : photon arrival times (ms) / colours.
    Photon emission terms evaluate ``mu_c`` at the path point nearest ``t_k``.
    """
    h = float(time_grid[1] - time_grid[0])

    # --- dynamics term: sum_m log N(x_{m+1}; x_m - D u' h, 2 D h) ---
    trans = em_transition_logp(x_path, D, potential, h).sum()

    # --- emission at photons: sum_k log mu_{c_k}(x(t_k)) ---
    idx = torch.clamp(torch.round(times / h).long(), 0, x_path.shape[0] - 1)
    x_at_photon = x_path[idx]
    mu_G, mu_R = emission_rates(x_at_photon, rates, C, R0)
    mu_c = torch.where(colors == 0, mu_G, mu_R)
    emit = torch.log(mu_c).sum()

    # --- depletion: - integral_0^T mu(x(t)) dt  ~  - h sum_m mu(x_m) ---
    mu_G_path, mu_R_path = emission_rates(x_path, rates, C, R0)
    mu_path = mu_G_path + mu_R_path
    depl = -h * mu_path.sum()

    ll = trans + emit + depl
    if log_p0 is not None:
        ll = ll + log_p0(x_path[0])
    return ll
