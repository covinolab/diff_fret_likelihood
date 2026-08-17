"""The Gamma prior on the background rates must reach the objective, and be a GAMMA.

``PriorConfig.bg_{g,r}_mean`` / ``bg_{g,r}_sd`` -- all in kHz, the same units as the
rates -- encode an independent background calibration.  Two things are tested here and neither is cosmetic:

* the term reaches ``prior_penalty`` -- the single place the prior enters the objective.
  ``logD`` was silently dropped once (see ``test_logD_prior.py``), so ``prior_penalty``
  RAISES when a bg prior is configured but ``rates`` is not supplied, and that is tested.
* it really is a Gamma, not a Gaussian on ``ln bg``.  The two agree to second order, so a
  copy-paste of ``logD_penalty`` would pass every test except the skew one below.

Why it matters: background and landscape width are one parameter, not two (the likelihood
pins their product ~15x tighter than either factor), so a calibrated background is what
makes D identifiable.  Simulator-independent -- everything here is analytic.
"""

import math

import pytest
import torch

import diff_fret_likelihood as dfl
from diff_fret_likelihood.objective import bg_penalty, prior_penalty

# a +/-10% calibration on each channel, expressed in kHz like everything else
MEAN_G, SD_G = 1.6, 0.16
MEAN_R, SD_R = 4.0, 0.40


def _grid(n=16, lo=4.0, hi=8.0):
    return dfl.GridConfig(lo, hi, n).build()


def _spline(grid, theta=(0.0, 1.0, 0.5, 2.0, 0.0, 1.0)):
    pot = dfl.build_potential(
        dfl.PotentialConfig(kind="spline", n_knots=len(theta)), grid
    )
    with torch.no_grad():
        pot.theta.copy_(torch.as_tensor(theta, dtype=torch.float64))
    return pot


def _rates(bg_g=MEAN_G, bg_r=MEAN_R):
    t = lambda v: torch.tensor(float(v), dtype=torch.float64)
    return dfl.EffectiveRates(t(24.0), t(24.0), t(bg_g), t(bg_r))


def _prior(**kw):
    kw.setdefault("curvature_weight", 0.01)
    return dfl.PriorConfig(**kw)


def test_bg_term_enters_prior_penalty():
    """prior_penalty(bg on) - prior_penalty(bg off) == bg_penalty exactly."""
    grid, pot = _grid(), _spline(_grid())
    D = torch.tensor(1.5, dtype=torch.float64)
    rates = _rates(bg_g=2.13, bg_r=4.45)          # the measured kde_H offset

    off = _prior()
    on = _prior(bg_g_mean=MEAN_G, bg_g_sd=SD_G, bg_r_mean=MEAN_R, bg_r_sd=SD_R)

    delta = float(prior_penalty(pot, D, grid, on, rates=rates)
                  - prior_penalty(pot, D, grid, off, rates=rates))
    assert delta == pytest.approx(float(bg_penalty(rates, on)), rel=0, abs=1e-12)


def test_zero_at_the_mean_and_positive_either_side():
    on = _prior(bg_g_mean=MEAN_G, bg_g_sd=SD_G)
    assert float(bg_penalty(_rates(bg_g=MEAN_G), on)) == pytest.approx(0.0, abs=1e-12)
    for bg in (0.5 * MEAN_G, 0.9 * MEAN_G, 1.1 * MEAN_G, 2.0 * MEAN_G):
        assert float(bg_penalty(_rates(bg_g=bg), on)) > 0.0


def test_it_is_a_gamma_not_a_gaussian_in_log_bg():
    """Asymmetric in log: too-LARGE a background costs more than too-small.

    A Gaussian on ln bg is exactly symmetric under bg -> mean**2/bg, so this asymmetry is
    the one property that separates the two -- a copy-paste of ``logD_penalty`` passes
    every other test in this file.

    The sign is the Gamma's, and it is the physically right one: with ``k`` counts behind
    the calibration, ``k(r - ln r - 1)`` has an exponential right tail (``r``) and only a
    logarithmic left one (``-ln r``).  A background much HIGHER than measured is strongly
    excluded -- you would have counted more photons -- while a lower one is not.
    """
    on = _prior(bg_g_mean=MEAN_G, bg_g_sd=0.8)       # wide (50%), so the skew is visible
    for f in (1.5, 2.0, 3.0):
        low = float(bg_penalty(_rates(bg_g=MEAN_G / f), on))
        high = float(bg_penalty(_rates(bg_g=MEAN_G * f), on))
        assert high > low, (f, low, high)
        # and it is a real asymmetry, not numerical noise
        assert high / low > 1.05


def test_curvature_in_log_bg_is_mean_over_sd_squared():
    """k = (mean/sd)**2 is what makes the kHz error bar mean what it says."""
    on = _prior(bg_g_mean=MEAN_G, bg_g_sd=SD_G)
    h = 1e-4
    f = lambda u: float(bg_penalty(_rates(bg_g=math.exp(u)), on))
    u0 = math.log(MEAN_G)
    d2 = (f(u0 + h) - 2 * f(u0) + f(u0 - h)) / h ** 2
    assert d2 == pytest.approx((MEAN_G / SD_G) ** 2, rel=1e-5)


def test_gradient_reaches_the_rate_parameter():
    log_bg = torch.tensor(math.log(2.13), dtype=torch.float64, requires_grad=True)
    t = lambda v: torch.tensor(float(v), dtype=torch.float64)
    rates = dfl.EffectiveRates(t(24.0), t(24.0), log_bg.exp(), t(MEAN_R))
    bg_penalty(rates, _prior(bg_g_mean=MEAN_G, bg_g_sd=SD_G)).backward()
    # d/d(ln bg) of k(r - ln r - 1) is k(r - 1) > 0 for bg above the mean
    assert float(log_bg.grad) == pytest.approx(
        (MEAN_G / SD_G) ** 2 * (2.13 / MEAN_G - 1.0), rel=1e-9)


def test_prior_penalty_raises_when_rates_are_missing():
    """A configured-but-unevaluable prior must fail loudly, never silently vanish."""
    grid, pot = _grid(), _spline(_grid())
    on = _prior(bg_g_mean=MEAN_G, bg_g_sd=SD_G)
    with pytest.raises(ValueError, match="rates"):
        prior_penalty(pot, torch.tensor(1.5, dtype=torch.float64), grid, on)


def test_channels_gate_independently_and_are_reported():
    g_only = _prior(bg_g_mean=MEAN_G, bg_g_sd=SD_G)
    r_only = _prior(bg_r_mean=MEAN_R, bg_r_sd=SD_R)
    both = _prior(bg_g_mean=MEAN_G, bg_g_sd=SD_G, bg_r_mean=MEAN_R, bg_r_sd=SD_R)
    rates = _rates(bg_g=2.0, bg_r=4.6)
    s = lambda p: float(bg_penalty(rates, p))
    assert s(both) == pytest.approx(s(g_only) + s(r_only), rel=1e-12)
    assert "bg" in both.active_terms() and "bg" in g_only.active_terms()
    assert "bg" not in _prior().active_terms()


def test_no_bg_prior_leaves_the_objective_untouched():
    """A PriorConfig without bg means must be byte-identical to before this feature."""
    grid, pot = _grid(), _spline(_grid())
    D = torch.tensor(1.5, dtype=torch.float64)
    p = _prior(logD_mean=math.log(1.5), logD_std=0.5)
    assert (float(prior_penalty(pot, D, grid, p))
            == float(prior_penalty(pot, D, grid, p, rates=_rates())))


def test_sd_is_in_kHz_not_a_fraction():
    """The width in bg is the configured sd, in the same units as the mean.

    Two channels with the same RELATIVE error (10%) but different means must get
    different absolute widths -- which is the whole point of moving off a shared
    relative sd.  Checked through the log-curvature, k = (mean/sd)**2.
    """
    h = 1e-4
    for mean, sd in ((MEAN_G, SD_G), (MEAN_R, SD_R), (MEAN_R, SD_G)):
        on = _prior(bg_g_mean=mean, bg_g_sd=sd)
        f = lambda u: float(bg_penalty(_rates(bg_g=math.exp(u)), on))
        u0 = math.log(mean)
        d2 = (f(u0 + h) - 2 * f(u0) + f(u0 - h)) / h ** 2
        assert 1.0 / math.sqrt(d2) * mean == pytest.approx(sd, rel=1e-5)


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError, match="bg_g_sd"):
        dfl.PriorConfig(bg_g_mean=MEAN_G, bg_g_sd=0.0)
    with pytest.raises(ValueError, match="bg_g_mean"):
        dfl.PriorConfig(bg_g_mean=-1.0, bg_g_sd=SD_G)


def test_a_mean_without_an_error_bar_is_rejected():
    """A calibrated background needs its uncertainty; silently defaulting one would be
    an invented measurement."""
    with pytest.raises(ValueError, match="without bg_g_sd"):
        dfl.PriorConfig(bg_g_mean=MEAN_G)
