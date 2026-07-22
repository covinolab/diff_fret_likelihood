"""Cramér–Rao bound (`cramer_rao_bound`) structural contract.

The CRB is `inv(Fisher)` evaluated at the ground-truth parameters from the
traces' scores.  These tests pin the *structural* guarantees the function must
hold for ANY data (they do not require the data to be at the truth):

  * shape / param layout / physical-unit summary,
  * Fisher is symmetric positive-semidefinite,
  * the landscape has an exact gauge null-space (constant-knot direction),
  * the gauge-fixed covariance is finite with a positive diagonal,
  * physical-unit sigmas follow the delta method,
  * the Fisher is additive over independent traces.

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


def _crb(s):
    return dfl.cramer_rao_bound(s["batch"], s["grid"], s["pot"], s["D"],
                                s["rates"], s["C"], s["R0"])


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
    # direction is a machine-zero null direction of the Fisher and is dropped from
    # the covariance.  (That it is the *only* null direction needs model-distributed
    # data — asserted with real simulated traces in the e2e verification script.)
    s = crb_setup
    res = _crb(s)
    K, P = s["K"], s["K"] + 5
    n = torch.zeros(P, dtype=res.fisher.dtype)
    n[:K] = 1.0 / np.sqrt(K)
    emax = float(torch.linalg.eigvalsh(res.fisher).max())
    assert abs(float(n @ res.fisher @ n)) < 1e-10 * emax        # Rayleigh quotient ~ 0
    assert float((res.fisher @ n).norm()) < 1e-7 * emax          # exact null direction
    assert res.null_dim >= 1                                      # dropped by the pinv
    assert float((res.cov @ n).norm()) < 1e-6 * float(res.cov.abs().max())


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
    prior = dfl.PriorConfig(curvature_weight=0.0, gp_sigma=2.0, gp_lengthscale=1.0)
    return dfl.cramer_rao_bound(s["batch"], s["grid"], s["pot"], s["D"], s["rates"],
                                s["C"], s["R0"], prior=prior, **kw)


def test_prior_free_default_unchanged(crb_setup):
    res = _crb(crb_setup)
    assert res.prior_included is False
    assert res.posterior_precision is None


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


def test_prior_pins_gauge_and_shrinks_vs_crb(crb_setup):
    s = crb_setup
    K = s["K"]
    crb = _crb(s)
    post = _post(s)
    # the pure CRB drops the landscape gauge (var ~ 0 along the constant-knot dir);
    # the gauge anchor gives it a finite variance = K * gauge_sd**2 (gauge_sd=1 default).
    n = torch.zeros(K + 5, dtype=crb.cov.dtype)
    n[:K] = 1.0 / np.sqrt(K)
    assert float(n @ crb.cov @ n) < 1e-6
    assert float(post.cov @ n @ n) == pytest.approx(K * 1.0 ** 2, rel=1e-6)
    # adding a prior can only tighten identified directions: posterior sigma <= CRB
    # sigma on the well-identified params (logD + the 4 log-rates).
    for i in range(K, K + 5):
        assert float(post.sigma[i]) <= float(crb.sigma[i]) + 1e-9


def test_gauge_sd_none_omits_anchor(crb_setup):
    s = crb_setup
    K = s["K"]
    # without the gauge anchor (and a mean-centered GP prior), the gauge stays a
    # null direction -> it is floored/dropped and reported in null_dim.
    res = _post(s, gauge_sd=None)
    assert res.null_dim >= 1
