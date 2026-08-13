"""The simulator and the likelihood share ONE emission model -- pinned here.

Both sides compute

    mu_G(x) = eta_g * kD * phi_g(x) + beta_g
    mu_R(x) = eta_r * kD * phi_r(x) + beta_r

with ``beta_g``/``beta_r`` DETECTED background rates, added *outside* the detector
efficiency.  The simulator used to take a pre-detector background and form
``eta_g * (kD*phi_g + k_gb)`` internally, so ``EffectiveRates.from_physics`` had to
multiply by ``eta`` to bridge the two.  That conversion is gone; if either side ever
drifts back, the tests below fail.

``test_bartlett_fisher.py`` also closes this loop, but it is ``@pytest.mark.slow`` and
excluded from CI.  These run in seconds.
"""

import numpy as np
import pytest
import torch

import diff_fret_likelihood as dfl
from diff_fret_likelihood.photophysics import EffectiveRates, emission_rates

# photophysics with eta well away from 1.0, so an eta on the background is unmissable
KD, ETA_G, ETA_R = 6.0, 0.5, 0.5
BETA_G, BETA_R = 0.425, 0.85
C_GR, C_RG = 0.10, 0.05
R0 = 6.0


def _consts_and_C(device="cpu"):
    consts = dfl.PhysicsConstants(
        R0=R0, C_gg=1.0 - C_GR, C_gr=C_GR, C_rg=C_RG, C_rr=1.0 - C_RG
    )
    return consts, consts.crosstalk_tensor(device)


def test_emission_rates_match_the_simulator_formula():
    """``emission_rates`` reproduces simulator.pyx:380-381 exactly, over all of x."""
    consts, C = _consts_and_C()
    rates = EffectiveRates.from_physics(KD, ETA_G, ETA_R, BETA_G, BETA_R)

    x = torch.linspace(2.0, 12.0, 257, dtype=dfl.config.DTYPE)
    mu_G, mu_R = emission_rates(x, rates, C, R0)

    # the simulator's per-step mean, re-derived independently in NumPy
    xn = x.numpy()
    E = R0 ** 6 / (R0 ** 6 + xn ** 6)
    phi_g = (1.0 - C_GR) * (1.0 - E) + C_RG * E
    phi_r = C_GR * (1.0 - E) + (1.0 - C_RG) * E
    lam_g = ETA_G * KD * phi_g + BETA_G          # background OUTSIDE eta
    lam_r = ETA_R * KD * phi_r + BETA_R

    np.testing.assert_allclose(mu_G.numpy(), lam_g, rtol=0, atol=1e-12)
    np.testing.assert_allclose(mu_R.numpy(), lam_r, rtol=0, atol=1e-12)


def test_from_physics_passes_background_through_untouched():
    """``bg_g`` IS ``beta_g`` -- no eta, no conversion.  The brightness still carries eta."""
    rates = EffectiveRates.from_physics(KD, ETA_G, ETA_R, BETA_G, BETA_R)
    assert float(rates.bg_g) == pytest.approx(BETA_G, rel=0, abs=1e-15)
    assert float(rates.bg_r) == pytest.approx(BETA_R, rel=0, abs=1e-15)
    assert float(rates.a_g) == pytest.approx(ETA_G * KD)
    assert float(rates.a_r) == pytest.approx(ETA_R * KD)


def test_background_only_limit_has_no_eta():
    """With kD = 0 the rate is pure background, so mu == beta with no eta anywhere."""
    _, C = _consts_and_C()
    rates = EffectiveRates.from_physics(0.0, ETA_G, ETA_R, BETA_G, BETA_R)
    x = torch.linspace(2.0, 12.0, 33, dtype=dfl.config.DTYPE)
    mu_G, mu_R = emission_rates(x, rates, C, R0)
    assert torch.allclose(mu_G, torch.full_like(mu_G, BETA_G))
    assert torch.allclose(mu_R, torch.full_like(mu_R, BETA_R))


# --------------------------------------------------------------------------- #
# against the COMPILED simulator, not just the two Python formulas
# --------------------------------------------------------------------------- #
X_KNOTS = np.linspace(2.0, 10.0, 15)
Y_KNOTS = 4.0 * (((X_KNOTS - 6.0) / 1.2) ** 2 - 1.0) ** 2


@pytest.mark.parametrize("beta_g,beta_r", [(20.0, 40.0)])
def test_simulated_background_rate_is_the_detected_rate(beta_g, beta_r):
    """kD = 0: every photon is background, so the realised count rate must be beta.

    This is the test that actually distinguishes the two conventions.  Under the old
    ``eta*(kD*phi + k_gb)`` form the same call would yield ``eta*beta = 0.5*beta``,
    which is ~250 Poisson sigma away from the tolerance below.
    """
    pytest.importorskip(
        "diff_fret_likelihood.simulator",
        reason="the Cython simulator extension is not built (needs GSL)",
    )
    T, n_traces = 50.0, 32
    batch = dfl.simulate.simulate_equilibrium(
        X_KNOTS, Y_KNOTS, 10.0, R0, 0.0, beta_g, beta_r,
        ETA_G, ETA_R, C_GR, C_RG, T, 5.0e-6,
        n_traces=n_traces, n_workers=4, seed=0, device="cpu", verbose=False,
    )

    valid = batch.mask
    n_green = int(((batch.colors == 0) & valid).sum())
    n_red = int(((batch.colors == 1) & valid).sum())
    total_T = float(batch.T.sum())

    # Poisson: relative sd = 1/sqrt(N).  N ~ 32k green, so 5 sigma is well under 3%.
    assert n_green / total_T == pytest.approx(beta_g, rel=5.0 / np.sqrt(n_green))
    assert n_red / total_T == pytest.approx(beta_r, rel=5.0 / np.sqrt(n_red))
