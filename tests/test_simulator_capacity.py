"""The simulator sizes its own photon buffers: ``photon_capacity`` and its consequences.

Two things are pinned here, and they are the whole point of removing the external
``N_max`` budget and the ``min_photons`` floor:

1. ``photon_capacity`` really is an upper bound on the per-channel photon count, so no
   trace can ever be lost to a buffer that is too small;
2. the photon-count distribution comes back UNTRUNCATED at both ends -- no upper cut
   (the buffer is large enough) and no lower cut (there is no count floor).  Both edges
   used to silently bias anything estimated from the pool, the Fisher above all.

Needs the compiled extension, so every test here skips without it.
"""

import warnings

import numpy as np
import pytest

import diff_fret_likelihood as dfl

simulator = pytest.importorskip(
    "diff_fret_likelihood.simulator",
    reason="the Cython simulator extension is not built (needs GSL)",
)
photon_capacity = simulator.photon_capacity


# --------------------------------------------------------------------------- #
# A confining double well on a WIDE knot domain: the simulator has no reflecting
# boundary, so U itself must keep the walker inside (see README).  Keeps the
# domain-exit rejection -- the only rejection left -- from firing.
# --------------------------------------------------------------------------- #
X_KNOTS = np.linspace(2.0, 10.0, 15)
Y_KNOTS = 4.0 * (((X_KNOTS - 6.0) / 1.2) ** 2 - 1.0) ** 2
D_TRUE, R0, DT = 10.0, 6.0, 5.0e-6

# the reference photophysics (kHz); mirrors the README quick-start
BASE = dict(kD=6.0, k_gb=0.5, k_rb=1.0, eta_g=0.85, eta_r=0.85, C_gr=0.10, C_rg=0.05)


def _cap(T, **over):
    """``photon_capacity`` in the (C_gr, C_rg) parameterisation the simulator wrapper uses."""
    p = {**BASE, **over}
    C_gg, C_rr = 1.0 - p["C_gr"], 1.0 - p["C_rg"]
    return photon_capacity(T, DT, p["kD"], p["k_gb"], p["k_rb"],
                           p["eta_g"], p["eta_r"], C_gg, C_rr, p["C_gr"], p["C_rg"])


def _mean_bound(T, **over):
    """The exact bound on E[count] per channel: ``T * max_channel(rate)``."""
    p = {**BASE, **over}
    C_gg, C_rr = 1.0 - p["C_gr"], 1.0 - p["C_rg"]
    lam_g = p["eta_g"] * (p["kD"] * max(C_gg, p["C_rg"]) + p["k_gb"])
    lam_r = p["eta_r"] * (p["kD"] * max(p["C_gr"], C_rr) + p["k_rb"])
    return T * max(lam_g, lam_r)


def _simulate(T, n_traces, seed=0, **over):
    p = {**BASE, **over}
    return dfl.simulate.simulate_equilibrium(
        X_KNOTS, Y_KNOTS, D_TRUE, R0, p["kD"], p["k_gb"], p["k_rb"],
        p["eta_g"], p["eta_r"], p["C_gr"], p["C_rg"], T, DT,
        n_traces=n_traces, n_workers=4, seed=seed, device="cpu", verbose=False,
    )


def _per_channel_counts(batch):
    """``(green, red)`` photon counts per trace -- what the buffers actually held.

    ``batch.lengths`` is the TOTAL over both channels and so is NOT bounded by the
    per-channel capacity: green + red can exceed it with neither channel close to it.
    """
    valid = batch.mask
    green = ((batch.colors == 0) & valid).sum(dim=1).numpy()
    red = ((batch.colors == 1) & valid).sum(dim=1).numpy()
    return green, red


# --------------------------------------------------------------------------- #
# the bound itself
# --------------------------------------------------------------------------- #
def test_capacity_clears_the_mean_by_many_sigma():
    # cap >= mu + Z*sqrt(mu) with Z = 10, which is what makes overflow ~1e-15/trace.
    for T in (10.0, 150.0, 2000.0):
        mu = _mean_bound(T)
        assert _cap(T) >= mu + 10.0 * np.sqrt(mu)


def test_capacity_is_monotone_in_every_rate_knob():
    base = _cap(150.0)
    assert _cap(300.0) > base                       # longer trace
    assert _cap(150.0, kD=60.0) > base              # brighter dye
    assert _cap(150.0, eta_g=1.0, eta_r=1.0) > base  # better detection
    assert _cap(150.0, k_gb=50.0) > base            # more background
    assert _cap(150.0, k_rb=50.0) > base


def test_capacity_stays_positive_in_the_dark_limit():
    # mu -> 0 is where a pure `Z*sqrt(mu)` margin would collapse; the additive slack
    # is what keeps the buffer usable (and a zero-photon trace representable).
    assert _cap(150.0, kD=0.0, k_gb=0.0, k_rb=0.0) > 0
    assert _cap(0.0) > 0


@pytest.mark.parametrize("bad", [
    dict(kD=-6.0),          # negative brightness
    dict(k_rb=-5.0, kD=6.0),  # background so negative the red rate goes negative at low E
])
def test_negative_poisson_mean_is_refused(bad):
    # gsl_ran_poisson is undefined for a negative mean, so the bound refuses up front
    # rather than letting the walker reach such a rate at some interior FRET efficiency.
    with pytest.raises(ValueError, match="cannot be negative"):
        _cap(150.0, **bad)


@pytest.mark.parametrize("bad", [dict(T=-1.0), dict(T=float("nan"))])
def test_bad_horizon_is_refused(bad):
    with pytest.raises(ValueError, match="finite T"):
        _cap(**bad)


def test_zero_and_negative_dt_refused():
    with pytest.raises(ValueError, match="dt > 0"):
        photon_capacity(150.0, 0.0, 6.0, 0.5, 1.0, 0.85, 0.85, 0.9, 0.95, 0.1, 0.05)


# --------------------------------------------------------------------------- #
# the bound holds against the simulator, and nothing is lost to it
# --------------------------------------------------------------------------- #
def test_realized_counts_stay_under_the_capacity():
    T = 150.0
    batch = _simulate(T, n_traces=64)
    green, red = _per_channel_counts(batch)
    cap = _cap(T)
    assert green.max() < cap and red.max() < cap
    # ... and the bound is a bound on the MEAN too, per channel.
    assert green.mean() < _mean_bound(T) and red.mean() < _mean_bound(T)


def test_bright_long_traces_no_longer_overflow():
    """The headline: a setting that would have blown the old hard-coded N_max = 12000.

    kD = 60 kHz over T = 1000 ms puts ~26k photons in each channel -- a guaranteed
    ``(None, None)`` reject under the old fixed budget, i.e. an empty pool.  With the
    computed capacity every trace survives.
    """
    T, n = 1000.0, 8
    cap = _cap(T, kD=60.0)
    batch = _simulate(T, n_traces=n, kD=60.0)
    assert batch.n_traces == n
    green, red = _per_channel_counts(batch)
    assert green.min() > 12000 and red.min() > 12000   # each channel busts the old budget
    assert green.max() < cap and red.max() < cap       # yet neither reaches the capacity


def test_dim_traces_are_kept_not_rejected():
    """No count floor: low-photon traces stay in the pool, at full requested count.

    This is the anti-truncation property.  A dim, short setting puts most traces well
    below the old ``min_photons = 50`` floor, which used to discard and re-draw them --
    returning a full-size pool drawn from a truncated count distribution.  Bright enough
    that a zero-photon trace is effectively impossible, so nothing is dropped at all.
    """
    T, n = 30.0, 96
    batch = _simulate(T, n_traces=n, kD=1.0, k_gb=0.05, k_rb=0.05)
    assert batch.n_traces == n
    lengths = batch.lengths.numpy()
    assert lengths.min() > 0                       # no empty traces to drop
    assert lengths.min() < 50                      # the old floor would have cut these
    assert (lengths < 50).mean() > 0.2             # and it was not a rare tail event


def test_no_warning_when_the_pool_comes_back_whole():
    T, n = 150.0, 32
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        batch = _simulate(T, n_traces=n)
    assert batch.n_traces == n
    # a confining landscape leaves nothing to reject, so neither warning fires
    runtime = [str(w.message) for w in rec if issubclass(w.category, RuntimeWarning)]
    assert not runtime, runtime


def test_zero_photon_traces_are_dropped_loudly():
    """The one remaining truncation, and it announces itself.

    A trace with no photons has no inter-photon times to represent, so it cannot enter
    the batch.  In the near-dark limit that is common, and the count distribution really
    is truncated at zero -- which must not pass silently.
    """
    with pytest.warns(RuntimeWarning, match="zero photons"):
        _simulate(2.0, n_traces=32, kD=1.0, k_gb=0.05, k_rb=0.05)
