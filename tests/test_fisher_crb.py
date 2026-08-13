"""Cramér–Rao bound (`cramer_rao_bound`) structural contract.

The CRB inverts the Fisher information (plus the always-on gauge anchor) evaluated at
the given parameters from the traces' scores.  These tests pin the *structural*
guarantees the function must hold for ANY data (they do not require the data to be at
the truth):

  * shape / param layout / physical-unit summary,
  * Fisher is symmetric positive-semidefinite,
  * the landscape has an exact gauge null-space (constant-knot direction) *in the
    Fisher*, which the anchor then pins in the covariance,
  * the covariance is finite with a positive diagonal,
  * physical-unit sigmas follow the delta method,
  * the Fisher is additive over independent traces.

The anchor's exact price is pinned too: `cov = pinv(F_N) + gauge_sd^2 * K * v v^T`
with `v` the unit constant-knot direction, so only per-knot sigma move and sigma_D /
sigma_rates / gauge-blind functionals are untouched.

Why the anchor is always on, and never gated on the prior -- the part these tests
assume rather than re-derive:

  * `v` is an EXACT null direction of `F_N` (tested below), so without the anchor the
    posterior precision is singular and has no inverse to report;
  * a landscape prior cannot stand in for it.  The mean-centred GP prior is itself
    gauge-invariant, and the curvature prior `D2^T D2` has a 2-dim null space
    {constant, linear} -- both are structurally blind to the gauge;
  * conversely the anchor pins exactly ONE direction.  Data that leave other knot
    directions weakly constrained still need a landscape prior for a finite per-knot
    sigma; the anchor does not rescue them.

**Everything here runs anchored.**  `gauge_sd=None` is deliberately not exercised: it
hands the inverter a matrix that is singular to working precision, so which way the
inverter resolves it is decided by the sign of a ~1e-17 rounding error -- which differs
between machines (it differed between a workstation and CI).  There is no stable answer
to assert, and `cramer_rao_bound` warns against the configuration anyway.

Statistical correctness at the truth (information-matrix identity `Cov(s)=-E[H]`
and the `1/sqrt(N)` law) is exercised separately with real simulated traces in
the end-to-end verification script.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import diff_fret_likelihood as dfl


def _make_batch(n_traces=30, K=12, seed=0):
    """A small padded batch of short random photon streams (leading gap 0)."""
    rng = np.random.default_rng(seed)
    raw = []
    for _ in range(n_traces):
        gaps = rng.uniform(0.001, 0.02, size=K)
        gaps[0] = 0.0
        colors = rng.integers(0, 2, size=K)
        raw.append((gaps.astype(np.float64), colors.astype(np.int64)))
    return dfl.simulate._stack(raw, max_photons=None, device="cpu")


@pytest.fixture
def crb_setup():
    K = 5
    grid = dfl.GridConfig(4.0, 8.0, 24).build()
    pcfg = dfl.PotentialConfig(kind="spline", n_knots=K)
    pot = dfl.build_potential(pcfg, grid)
    with torch.no_grad():
        pot.theta.copy_(torch.tensor([2.0, 0.0, 1.5, 0.0, 2.0]))  # bumpy landscape
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(kD=300.0, eta_g=0.85, eta_r=0.85,
                                            beta_g=21.25, beta_r=42.5)
    D = torch.tensor(10.0)
    batch = _make_batch()
    return dict(batch=batch, grid=grid, pot=pot, D=D, rates=rates,
                C=C, R0=consts.R0, K=K)


def _crb(s, **kw):
    return dfl.cramer_rao_bound(s["batch"], s["grid"], s["pot"], s["D"],
                                s["rates"], s["C"], s["R0"], **kw)


def _gauge_dir(K, P, dtype=torch.float64):
    """The unit constant-knot direction ``v`` -- the exact flat direction of U."""
    v = torch.zeros(P, dtype=dtype)
    v[:K] = 1.0 / np.sqrt(K)
    return v


def _anchor_hessian(K, P, gauge_sd, dtype=torch.float64):
    """The anchor's own Hessian: ``1 1^T / (gauge_sd^2 K^2)`` on the landscape block.

    Rank one, acting as ``1/(gauge_sd^2 K)`` along ``v`` and as exactly zero on every
    direction orthogonal to it -- the form asserted in
    ``test_prior_free_default_still_reports_no_prior_but_is_anchored``.  Subtracting it
    from ``posterior_precision`` recovers the un-anchored precision without running the
    CRB in a configuration that has no stable answer (see the module docstring).
    """
    H = torch.zeros(P, P, dtype=dtype)
    H[:K, :K] = 1.0 / (gauge_sd ** 2 * K ** 2)
    return H


def test_returns_expected_shapes_and_names(crb_setup):
    s = crb_setup
    res = _crb(s)
    P = s["K"] + 5
    assert res.fisher.shape == (P, P)
    assert res.fisher_per_trace.shape == (P, P)
    assert res.cov.shape == (P, P)
    assert res.sigma.shape == (P,)
    assert len(res.param_names) == P
    assert res.param_names[s["K"]:] == ["logD", "log_a_g", "log_a_r",
                                        "log_bg_g", "log_bg_r"]
    assert set(res.sigma_physical) >= {"D", "a_g", "a_r", "bg_g", "bg_r", "knots"}
    assert len(res.sigma_physical["knots"]) == s["K"]
    assert res.n_traces == s["batch"].n_traces


def test_fisher_symmetric_psd(crb_setup):
    res = _crb(crb_setup)
    F = res.fisher
    scale = float(F.abs().max())
    assert torch.allclose(F, F.T, atol=1e-8 * scale)
    evals = torch.linalg.eigvalsh(F)
    assert float(evals.min()) > -1e-8 * scale        # PSD


def test_landscape_constant_shift_is_a_gauge_null_direction(crb_setup):
    # The likelihood is exactly invariant to U -> U + const, so the constant-knot
    # direction is a machine-zero null direction of the FISHER.  This holds whatever
    # penalties are added afterwards: `fisher` is always the pure likelihood
    # information.  (That it is the *only* null direction needs model-distributed
    # data — asserted with real simulated traces in the e2e verification script.)
    s = crb_setup
    res = _crb(s)
    n = _gauge_dir(s["K"], s["K"] + 5, res.fisher.dtype)
    emax = float(torch.linalg.eigvalsh(res.fisher).max())
    assert abs(float(n @ res.fisher @ n)) < 1e-10 * emax        # Rayleigh quotient ~ 0
    assert float((res.fisher @ n).norm()) < 1e-7 * emax          # exact null direction


@pytest.mark.parametrize("gauge_sd", [1.0, 0.25])
def test_anchor_gives_the_gauge_exactly_its_prior_variance(crb_setup, gauge_sd):
    """``v @ cov @ v == gauge_sd^2 * K``, exactly and unconditionally.

    ``H_gauge = 1 1^T / (gauge_sd^2 K^2)`` acts as ``1/(gauge_sd^2 K)`` along ``v`` and as
    zero elsewhere, and ``v`` is an exact null direction of ``F_N``.  So the anchored
    inverse's gauge variance is the anchor's own variance, whatever the data.
    """
    s = crb_setup
    res = _crb(s, gauge_sd=gauge_sd)
    v = _gauge_dir(s["K"], s["K"] + 5, res.cov.dtype)
    assert res.null_dim == 0                         # exact Cholesky, no threshold
    assert res.posterior_precision is not None       # the anchor's Hessian went in
    assert float(v @ res.cov @ v) == pytest.approx(s["K"] * gauge_sd ** 2, rel=1e-8)


@pytest.mark.parametrize("gauge_sd", [1.0, 0.25])
def test_anchor_touches_nothing_but_the_gauge_direction(crb_setup, gauge_sd):
    """The anchor's whole effect is its rank-one term along ``v``:

        ``cov == pinv(F_N + H_prior) + gauge_sd**2 * K * v v^T``

    From a SINGLE anchored run.  The un-anchored precision ``A`` is recovered by
    subtracting the anchor's own Hessian, and the pseudo-inverse is taken *here* rather
    than asked of the CRB -- an actual ``gauge_sd=None`` call would hand the inverter a
    matrix that is singular to working precision, whose resolution is decided by a
    ~1e-17 rounding sign (see the module docstring).  Nothing below depends on which
    inverse branch ran.

    Uses the posterior (GP-prior) CRB so that the gauge is the only null direction of
    ``A``.  The GP prior is mean-centred, hence gauge-invariant, so it does not pin the
    gauge itself -- asserted below, and the reason the anchor is not gated on the prior.
    """
    s = crb_setup
    K, P = s["K"], s["K"] + 5
    r = _post(s, gauge_sd=gauge_sd)
    v = _gauge_dir(K, P, r.cov.dtype)

    A = r.posterior_precision - _anchor_hessian(K, P, gauge_sd, r.cov.dtype)
    A = 0.5 * (A + A.T)                                  # == F_N + H_prior
    # the prior left the gauge exactly as unpinned as the likelihood did
    assert abs(float(v @ A @ v)) < 1e-12 * float(torch.linalg.eigvalsh(A).max())
    pinvA = torch.linalg.pinv(A, rtol=1e-10)

    assert float((r.cov - (pinvA + gauge_sd ** 2 * K * torch.outer(v, v))).abs().max()) \
        < 1e-9 * float(r.cov.abs().max())

    # hence every quantity that lives off the gauge is untouched: logD, the four log
    # rates, and any gauge-blind landscape functional (a "barrier height" d, d . 1 = 0)
    assert torch.allclose(r.sigma[K:], torch.sqrt(torch.diag(pinvA))[K:], rtol=1e-9)
    d = torch.zeros(P, dtype=r.cov.dtype)
    d[0], d[K - 1] = 1.0, -1.0
    assert float(d @ r.cov @ d) == pytest.approx(float(d @ pinvA @ d), rel=1e-9)

    # and what DOES move: each knot variance carries exactly gauge_sd**2 (= K*gauge_sd**2
    # times v_i**2 = 1/K) over its strict sum-to-zero value.
    excess = (torch.diag(r.cov)[:K] - torch.diag(pinvA)[:K]).numpy()
    assert np.allclose(excess, gauge_sd ** 2, atol=1e-8)


def test_cov_finite_positive_diag(crb_setup):
    res = _crb(crb_setup)
    d = torch.diag(res.cov)
    assert torch.isfinite(d).all()
    assert bool((d > 0).all())
    assert torch.allclose(res.sigma, torch.sqrt(d))


def test_physical_units_follow_delta_method(crb_setup):
    s = crb_setup
    res = _crb(s)
    idx = {name: i for i, name in enumerate(res.param_names)}
    # D = exp(logD)  ->  sigma_D = D * sigma_logD
    assert res.sigma_physical["D"] == pytest.approx(
        float(s["D"]) * float(res.sigma[idx["logD"]]), rel=1e-6)
    # a_g = exp(log_a_g)  ->  sigma_ag = a_g * sigma_log_a_g
    assert res.sigma_physical["a_g"] == pytest.approx(
        float(s["rates"].a_g) * float(res.sigma[idx["log_a_g"]]), rel=1e-6)
    assert res.sigma_physical["bg_r"] == pytest.approx(
        float(s["rates"].bg_r) * float(res.sigma[idx["log_bg_r"]]), rel=1e-6)
    # knots are already in kT: the per-knot sigma is the landscape-block sigma
    assert np.allclose(res.sigma_physical["knots"],
                       res.sigma[:s["K"]].numpy())


def test_fisher_additive_over_traces(crb_setup):
    res = _crb(crb_setup)
    assert torch.allclose(res.fisher_per_trace, res.fisher / res.n_traces)


# --------------------------------------------------------------------------- #
# prior inclusion: cov becomes the posterior covariance (Fisher + prior Hessian)
# --------------------------------------------------------------------------- #
def _post(s, **kw):
    """The posterior CRB: same call as ``_crb`` plus a mean-centred GP landscape prior."""
    prior = dfl.PriorConfig(curvature_weight=0.0, gp_sigma=2.0, gp_lengthscale=1.0)
    return dfl.cramer_rao_bound(s["batch"], s["grid"], s["pot"], s["D"], s["rates"],
                                s["C"], s["R0"], prior=prior, **kw)


def test_prior_free_default_still_reports_no_prior_but_is_anchored(crb_setup):
    res = _crb(crb_setup)
    # `prior_included` tracks a PriorConfig only -- the gauge anchor is not a prior in
    # that sense, so it stays False ...
    assert res.prior_included is False
    # ... but the anchor DID go in, so there is a matrix that was inverted, and it is
    # the Fisher plus a PSD rank-one term in the gauge direction alone.
    assert res.posterior_precision is not None
    K, P = crb_setup["K"], crb_setup["K"] + 5
    H = res.posterior_precision - res.fisher
    H = 0.5 * (H + H.T)
    evals = torch.linalg.eigvalsh(H)
    assert float(evals.min()) > -1e-10 * float(evals.max())      # PSD
    assert int((evals > 1e-10 * evals.max()).sum()) == 1          # rank one
    v = _gauge_dir(K, P, H.dtype)
    assert float(v @ H @ v) == pytest.approx(1.0 / (K * 1.0 ** 2), rel=1e-8)
    # and it is exactly `1 1^T / (gauge_sd^2 K^2)` on the landscape block -- the form
    # `test_anchor_touches_nothing_but_the_gauge_direction` subtracts back off
    assert torch.allclose(H, _anchor_hessian(K, P, 1.0, H.dtype), atol=1e-12)


def test_prior_gives_posterior_precision(crb_setup):
    s = crb_setup
    res = _post(s)
    P = s["K"] + 5
    assert res.prior_included is True
    assert res.posterior_precision is not None
    assert res.posterior_precision.shape == (P, P)
    # symmetric
    assert torch.allclose(res.posterior_precision, res.posterior_precision.T, atol=1e-8)
    # posterior precision = Fisher + prior Hessian; the added prior Hessian is PSD
    prior_H = res.posterior_precision - res.fisher
    prior_H = 0.5 * (prior_H + prior_H.T)
    assert float(torch.linalg.eigvalsh(prior_H).min()) > -1e-8
    # proper prior => nothing dropped and every parameter has a finite, positive sigma
    assert res.null_dim == 0
    assert torch.isfinite(res.sigma).all() and bool((res.sigma > 0).all())
    assert torch.allclose(res.sigma, torch.sqrt(torch.diag(res.cov)))


def test_prior_shrinks_vs_the_anchored_crb(crb_setup):
    s = crb_setup
    K = s["K"]
    crb = _crb(s)                    # anchored, no PriorConfig
    post = _post(s)                  # anchored + GP prior
    v = _gauge_dir(K, K + 5, crb.cov.dtype)
    # both are anchored, so both give the gauge a finite variance; the mean-centered GP
    # prior is gauge-invariant, so it does not tighten the gauge itself.
    assert float(v @ crb.cov @ v) == pytest.approx(K * 1.0 ** 2, rel=1e-6)
    assert float(v @ post.cov @ v) == pytest.approx(K * 1.0 ** 2, rel=1e-6)
    # adding a prior can only tighten identified directions: posterior sigma <= CRB
    # sigma on the well-identified params (logD + the 4 log-rates).
    for i in range(K, K + 5):
        assert float(post.sigma[i]) <= float(crb.sigma[i]) + 1e-9
    # and on the landscape block too, where the GP prior actually bites
    for i in range(K):
        assert float(post.sigma[i]) <= float(crb.sigma[i]) + 1e-9
