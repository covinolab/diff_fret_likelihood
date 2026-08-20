"""Joint multi-dataset fit (``fit_multi``): shared ``(U, D)``, per-dataset photophysics.

Simulator-independent -- tiny synthetic potentials / photon streams built here,
same ``_Batch``-stub style as ``test_prior_none.py``.

The statistical contract under test:

    loss = - sum_d loglik_d(U, D, rates_d ; R0_d, C_d)  +  prior(U, D)  +  gauge(U)

with the prior and gauge added ONCE on the shared (U, D).
"""

import pytest
import torch

import diff_fret_likelihood as dfl
from diff_fret_likelihood.forward import marginal_loglik_batch
from diff_fret_likelihood.objective import prior_penalty


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _grid(n=16, lo=4.0, hi=8.0):
    return dfl.GridConfig(lo, hi, n).build()


_THETA = [0.2, -0.8, 0.9, -0.5, 0.7]


def _fresh_pot(grid, theta=_THETA):
    """A fresh spline potential (fit mutates ``theta`` in place, so never share)."""
    pot = dfl.build_potential(
        dfl.PotentialConfig(n_knots=len(theta)), grid
    )
    with torch.no_grad():
        pot.theta.copy_(torch.as_tensor(theta, dtype=torch.float64))
    return pot


def _tiny_batch(seed=3, n=8):
    torch.manual_seed(seed)
    gaps = torch.rand(n) * 0.01
    gaps[0] = 0.0
    ipt = gaps[None, :]
    colors = torch.randint(0, 2, (n,))[None, :]
    mask = torch.ones(1, n, dtype=torch.bool)
    return _Batch(ipt, colors, mask)


class _Batch:
    """Minimal batch object exposing the attrs ``infer.fit`` reads."""

    def __init__(self, ipt, colors, mask):
        self.ipt, self.colors, self.mask = ipt, colors, mask

    def to(self, device):
        return self


def _rates(kD=300.0):
    return dfl.EffectiveRates.from_physics(kD, 0.85, 0.85, 25, 50)


# --------------------------------------------------------------------------- #
# 1. single-dataset equivalence -- the anchor against loop drift
# --------------------------------------------------------------------------- #
def test_fit_multi_single_dataset_matches_fit():
    """fit_multi with one-element lists is bit-identical to fit (same knobs; the
    guarded LBFGS loop is deterministic)."""
    grid = _grid()
    b = _tiny_batch()
    C = dfl.PhysicsConstants().crosstalk_tensor()
    R0 = dfl.PhysicsConstants().R0
    rates = _rates()
    prior = dfl.PriorConfig(curvature_weight=0.05)
    optim = dfl.OptimConfig(steps=8, log_every=1)

    common = dict(D_init=10.0, prior=prior, optim=optim, fit_D=True,
                  fit_rates=True, verbose=False)

    pot_a, pot_b = _fresh_pot(grid), _fresh_pot(grid)
    res_single = dfl.fit(b, grid, pot_a, C, R0, rates_init=rates, **common)
    res_multi = dfl.fit_multi([b], grid, pot_b, [C], [R0],
                              rates_init_list=[rates], **common)

    u_single = dfl.recovered_potential(pot_a, grid)
    u_multi = dfl.recovered_potential(pot_b, grid)
    assert torch.allclose(u_single, u_multi, atol=1e-10, rtol=0.0)
    assert res_single.D == pytest.approx(res_multi.D, abs=1e-10)
    assert res_single.best_loss == pytest.approx(res_multi.best_loss, abs=1e-9)


# --------------------------------------------------------------------------- #
# 2. the prior is counted ONCE, not once per dataset
# --------------------------------------------------------------------------- #
def test_fit_multi_prior_counted_once():
    """With N identical datasets, prior-on minus prior-off loss == prior_penalty (x1)."""
    grid = _grid()
    b = _tiny_batch()
    C = dfl.PhysicsConstants().crosstalk_tensor()
    R0 = dfl.PhysicsConstants().R0
    rates = _rates()
    prior = dfl.PriorConfig(curvature_weight=0.1)

    # best_loss with steps=1 is exactly the loss at the initial parameters (the
    # snapshot is taken BEFORE the quasi-Newton step moves them).
    optim = dfl.OptimConfig(steps=1, log_every=1)
    common = dict(D_init=10.0, optim=optim, fit_D=True, fit_rates=True,
                  verbose=False)

    n_ds = 3
    loss_on = dfl.fit_multi([b] * n_ds, grid, _fresh_pot(grid), [C] * n_ds,
                            [R0] * n_ds, rates_init_list=[rates] * n_ds,
                            prior=prior, **common).best_loss
    loss_off = dfl.fit_multi([b] * n_ds, grid, _fresh_pot(grid), [C] * n_ds,
                             [R0] * n_ds, rates_init_list=[rates] * n_ds,
                             prior=None, **common).best_loss

    reg = float(prior_penalty(_fresh_pot(grid), torch.tensor(10.0), grid, prior))
    assert reg > 0.0
    # prior enters exactly once -- NOT n_ds times.
    assert (loss_on - loss_off) == pytest.approx(reg, abs=1e-7)
    assert (loss_on - loss_off) != pytest.approx(n_ds * reg, abs=1e-3)


# --------------------------------------------------------------------------- #
# 3. per-dataset rates are optimised independently; result shape is per input
# --------------------------------------------------------------------------- #
def test_fit_multi_rates_independent():
    grid = _grid()
    b = _tiny_batch()
    C = dfl.PhysicsConstants().crosstalk_tensor()
    rates0, rates1 = _rates(kD=100.0), _rates(kD=500.0)
    optim = dfl.OptimConfig(steps=12, log_every=1)

    res = dfl.fit_multi(
        [b, b], grid, _fresh_pot(grid), [C, C], [5.0, 6.0], D_init=10.0,
        rates_init_list=[rates0, rates1], prior=None, optim=optim,
        fit_D=True, fit_rates=True, verbose=False,
    )

    assert len(res.rates) == 2
    # the two fitted rate sets differ from each other ...
    assert not torch.isclose(res.rates[0].a_g, res.rates[1].a_g)
    # ... and each has moved away from its own init (independent optimisation).
    assert not torch.isclose(res.rates[0].a_g, rates0.a_g)
    assert not torch.isclose(res.rates[1].a_g, rates1.a_g)


# --------------------------------------------------------------------------- #
# 4. fit_rates=False holds every dataset's rates fixed at its init
# --------------------------------------------------------------------------- #
def test_fit_multi_fit_rates_false_keeps_inits():
    grid = _grid()
    b = _tiny_batch()
    C = dfl.PhysicsConstants().crosstalk_tensor()
    rates0, rates1 = _rates(kD=100.0), _rates(kD=500.0)
    pot = _fresh_pot(grid)
    theta0 = pot.theta.detach().clone()
    optim = dfl.OptimConfig(steps=10, log_every=1)

    res = dfl.fit_multi(
        [b, b], grid, pot, [C, C], [5.0, 6.0], D_init=10.0,
        rates_init_list=[rates0, rates1], prior=None, optim=optim,
        fit_D=True, fit_rates=False, verbose=False,
    )

    for got, init in zip(res.rates, [rates0, rates1]):
        assert torch.allclose(got.a_g, init.a_g, atol=1e-12)
        assert torch.allclose(got.bg_r, init.bg_r, atol=1e-12)
    # the landscape still moved (only the rates were frozen).
    assert not torch.allclose(pot.theta.detach(), theta0, atol=1e-6)


# --------------------------------------------------------------------------- #
# 5. mismatched list lengths raise a clear error, not a deep IndexError
# --------------------------------------------------------------------------- #
def test_fit_multi_length_mismatch_raises():
    grid = _grid()
    b = _tiny_batch()
    C = dfl.PhysicsConstants().crosstalk_tensor()
    rates = _rates()
    with pytest.raises(ValueError):
        dfl.fit_multi(
            [b, b], grid, _fresh_pot(grid), [C, C], [5.0],  # 2 batches, 1 R0
            D_init=10.0, rates_init_list=[rates, rates], prior=None,
            verbose=False,
        )
