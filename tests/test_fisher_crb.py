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
                                            k_gb=25.0, k_rb=50.0)
    D = torch.tensor(10.0)
    batch = _make_batch()
    return dict(batch=batch, grid=grid, pot=pot, D=D, rates=rates,
                C=C, R0=consts.R0, K=K)


def _crb(s, **kw):
    return dfl.cramer_rao_bound(s["batch"], s["grid"], s["pot"], s["D"],
                                s["rates"], s["C"], s["R0"], **kw)


def _unanchored(s, **kw):
    """The pre-anchor pseudo-inverse CRB, which now warns (that is the point)."""
    with pytest.warns(RuntimeWarning, match="gauge_sd=None"):
        return _crb(s, gauge_sd=None, **kw)


def _gauge_dir(K, P, dtype=torch.float64):
    """The unit constant-knot direction ``v`` -- the exact flat direction of U."""
    v = torch.zeros(P, dtype=dtype)
    v[:K] = 1.0 / np.sqrt(K)
    return v


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


def test_without_the_anchor_the_gauge_is_dropped_from_cov(crb_setup):
    # The old default: nothing pins the gauge, so the pseudo-inverse drops it and
    # reports it in null_dim.  Reachable only by asking for it, and it warns.
    s = crb_setup
    res = _unanchored(s)
    n = _gauge_dir(s["K"], s["K"] + 5, res.cov.dtype)
    assert res.null_dim >= 1                                     # dropped by the pinv
    assert res.posterior_precision is None                       # nothing was added
    assert float((res.cov @ n).norm()) < 1e-6 * float(res.cov.abs().max())


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


def test_anchor_touches_nothing_but_the_gauge_direction(crb_setup):
    """The anchor's whole effect is the rank-one term along ``v``.

    Uses the posterior (GP-prior) CRB, because there the gauge is the ONLY null
    direction -- the precondition for comparing against an unanchored run at all.  On
    the prior-free path the data leave other knot directions unconstrained too, and the
    two inverses then legitimately disagree about those (see the next test).

    Since the mean-centred GP prior is itself gauge-invariant, it does not pin the gauge:
    only the anchor does.  That is exactly why the anchor is not gated on the prior.
    """
    s = crb_setup
    K, P = s["K"], s["K"] + 5
    anchored, plain = _post(s), _post(s, gauge_sd=None, _warn=True)
    v = _gauge_dir(K, P, anchored.cov.dtype)
    proj = torch.eye(P, dtype=anchored.cov.dtype) - torch.outer(v, v)

    # off the gauge direction the two covariances agree to machine precision
    pa, pu = proj @ anchored.cov @ proj, proj @ plain.cov @ proj
    assert float((pa - pu).abs().max()) < 1e-10 * float(pa.abs().max())

    # hence every quantity that lives off it is untouched: D, the rates, and any
    # gauge-blind landscape functional (a "barrier height" d, with d . 1 = 0)
    assert anchored.sigma_physical["D"] == pytest.approx(
        plain.sigma_physical["D"], rel=1e-10)
    for nm in ("a_g", "a_r", "bg_g", "bg_r"):
        assert anchored.sigma_physical[nm] == pytest.approx(
            plain.sigma_physical[nm], rel=1e-10)
    d = torch.zeros(P, dtype=anchored.cov.dtype)
    d[0], d[K - 1] = 1.0, -1.0
    assert float(d @ anchored.cov @ d) == pytest.approx(
        float(d @ plain.cov @ d), rel=1e-10)

    # and what DOES move: each knot variance carries gauge_sd**2 (= K*gauge_sd**2 times
    # v_i**2 = 1/K) over its strict sum-to-zero value.  Self-contained, no second run.
    excess = (torch.diag(anchored.cov)[:K] - torch.diag(pa)[:K]).numpy()
    assert np.allclose(excess, 1.0, atol=1e-8)


def test_anchor_does_not_rescue_non_gauge_unconstrained_directions(crb_setup):
    """The documented limit: the anchor pins ONE direction, and pinv hides the rest.

    This fixture's random gaps are not model-distributed, so the Fisher is near-null in
    several knot directions beyond the gauge.  The unanchored pseudo-inverse *drops* them
    and so reports a sigma that is a lower bound masquerading as a bound; the anchored
    inverse keeps them and reports the honestly larger value.  A finite bound on every
    knot needs a landscape prior, not the anchor.
    """
    s = crb_setup
    anchored, plain = _crb(s), _unanchored(s)
    assert plain.null_dim > 1                        # more than just the gauge is null
    # the dropped directions carried real variance, so the pinv understated sigma_D
    assert anchored.sigma_physical["D"] > 2.0 * plain.sigma_physical["D"]
    # the prior is what fixes this, and then the two agree again (previous test)
    assert _post(s).sigma_physical["D"] < plain.sigma_physical["D"]


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
def _post(s, _warn=False, **kw):
    """The posterior CRB (GP prior).  ``_warn=True`` expects the gauge_sd=None warning."""
    prior = dfl.PriorConfig(curvature_weight=0.0, gp_sigma=2.0, gp_lengthscale=1.0)
    call = lambda: dfl.cramer_rao_bound(                            # noqa: E731
        s["batch"], s["grid"], s["pot"], s["D"], s["rates"],
        s["C"], s["R0"], prior=prior, **kw)
    if _warn:
        with pytest.warns(RuntimeWarning, match="gauge_sd=None"):
            return call()
    return call()


def test_prior_free_default_still_reports_no_prior_but_is_anchored(crb_setup):
    res = _crb(crb_setup)
    # `prior_included` tracks a PriorConfig only -- the gauge anchor is not a prior in
    # that sense, so it stays False ...
    assert res.prior_included is False
    # ... but the anchor DID go in, so there is a matrix that was inverted, and it is
    # the Fisher plus a PSD rank-one term in the gauge direction alone.
    assert res.posterior_precision is not None
    H = res.posterior_precision - res.fisher
    H = 0.5 * (H + H.T)
    evals = torch.linalg.eigvalsh(H)
    assert float(evals.min()) > -1e-10 * float(evals.max())      # PSD
    assert int((evals > 1e-10 * evals.max()).sum()) == 1          # rank one
    v = _gauge_dir(crb_setup["K"], crb_setup["K"] + 5, H.dtype)
    assert float(v @ H @ v) == pytest.approx(1.0 / (crb_setup["K"] * 1.0 ** 2), rel=1e-8)


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


def test_gauge_sd_none_omits_anchor(crb_setup):
    # Without the gauge anchor (and with a mean-centered, hence gauge-invariant, GP
    # prior) the gauge stays a null direction -> floored/dropped, reported in null_dim.
    # A prior alone does NOT pin it: that is why the anchor is not gated on the prior.
    res = _post(crb_setup, gauge_sd=None, _warn=True)
    assert res.null_dim >= 1
