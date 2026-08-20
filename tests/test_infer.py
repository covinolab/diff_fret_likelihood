"""Regression: forward-backward occupancy is self-consistent (audit fix)."""

import math

import torch

import diff_fret_likelihood as dfl
from diff_fret_likelihood.forward import build_propagator_from_u, _BasePotential_on_grid
from diff_fret_likelihood.generator import stationary
from diff_fret_likelihood.objective import gauge_penalty, neg_log_posterior


class _OneTraceBatch:
    def __init__(self, ipt, colors, n):
        self.ipt = ipt[None, :]
        self.colors = colors[None, :]
        self.mask = torch.ones(1, n, dtype=torch.bool)
        self.lengths = torch.tensor([n])
        self.n_traces = 1

    def to(self, device):
        return self


def test_forward_backward_consistency():
    """Sum_i a~_k * b~_k must be constant over k and equal the marginal likelihood."""
    torch.manual_seed(1)
    grid = dfl.GridConfig(4, 8, 20).build()
    pot = dfl.build_potential(dfl.PotentialConfig(n_knots=6), grid)
    with torch.no_grad():
        pot.theta.copy_(torch.tensor([0.2, -0.8, 0.9, -0.5, 0.7, 0.1]))
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300, .85, .85, 25, 50)
    D = torch.tensor(10.0)
    dx = float(grid[1] - grid[0])
    u = _BasePotential_on_grid(pot, grid)
    prop = build_propagator_from_u(u, D, rates, grid, C, consts.R0, dx)
    s = prop.s
    n = 9
    gaps = torch.rand(n) * 0.01
    gaps[0] = 0.0
    cols = torch.randint(0, 2, (n,))

    a, v = [], stationary(u) / s
    for k in range(n):
        v = prop.propagate(v, gaps[k])
        v = v * (prop.mu_G if cols[k] == 0 else prop.mu_R)
        a.append(v.clone())
    b = [None] * n
    b[n - 1] = s.clone()
    for k in range(n - 1, 0, -1):
        em = prop.mu_G if cols[k] == 0 else prop.mu_R
        b[k - 1] = prop.propagate(b[k] * em, gaps[k])
    smoothed = torch.stack([(a[k] * b[k]).sum() for k in range(n)])
    assert (smoothed.max() / smoothed.min() - 1).abs() < 1e-9

    times = torch.cumsum(gaps, 0)
    ll = dfl.marginal_loglik(times, cols, float(times[-1]), pot, D, rates, grid, C, consts.R0)
    assert abs(math.log(float(smoothed[0])) - float(ll)) < 1e-8


def test_best_loss_matches_objective_at_returned_state():
    """``best_loss`` must be the objective evaluated AT the state the fit returns.

    Regression: with ``max_iter=1``, ``LBFGS.step`` returns the loss at the parameters
    it was ENTERED with and leaves them one quasi-Newton step further -- so pairing that
    loss with a snapshot taken after the step reports a value belonging to a parameter
    vector that was never saved (the same off-by-one this test originally caught for the
    noise-injected Adam loop).  The fit must snapshot BEFORE stepping.

    The settings matter: a non-monotone (overshooting) trajectory is what exposes it,
    because the best step is then not the last one and the step taken away from it moves
    the objective measurably.  ``lbfgs_lr=3.0`` forces that on this problem (measured:
    9 uphill steps, best at step 1, ~0.7-nat mispairing under the bug -- vs a monotone,
    fully converged run at the default lr where the mismatch collapses to zero and the
    bug hides completely).  The premise is asserted below so the test cannot silently
    lose its teeth if the trajectory changes.
    """
    torch.manual_seed(0)
    grid = dfl.GridConfig(4, 8, 24).build()
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300, .85, .85, 25, 50)
    n = 60
    gaps = torch.rand(n) * 0.01
    gaps[0] = 0.0
    cols = torch.randint(0, 2, (n,))
    batch = _OneTraceBatch(gaps, cols, n)

    prior = dfl.PriorConfig(curvature_weight=0.1)
    optim = dfl.OptimConfig(steps=20, lbfgs_lr=3.0, log_every=1)
    pot = dfl.build_potential(dfl.PotentialConfig(n_knots=6), grid)
    with torch.no_grad():
        pot.theta.copy_(torch.tensor([0.5, 0.0, -0.9, 0.8, -0.3, 0.4]))

    gauge_sd = 1.0
    res = dfl.fit(batch, grid, pot, C, consts.R0, D_init=10.0, rates_init=rates,
                  prior=prior, optim=optim, fit_D=True, verbose=False,
                  gauge_sd=gauge_sd)

    # PREMISE: the trajectory must be non-monotone with the best step not the last,
    # otherwise this test cannot detect a snapshot/loss off-by-one.
    losses = [h["loss"] for h in res.history[:-1]]
    assert any(l2 > l1 for l1, l2 in zip(losses, losses[1:])), \
        "trajectory became monotone -- the pairing test lost its forcing mechanism"
    assert losses.index(min(losses)) < len(losses) - 1

    # Rebuild the fit's own objective rather than reimplementing it, so the test cannot
    # drift from the fit path: neg_log_posterior + gauge_penalty, same as closure_value.
    with torch.no_grad():
        recomputed = float(
            neg_log_posterior(batch.ipt, batch.colors, batch.mask, res.potential,
                              torch.as_tensor(res.D, dtype=dfl.DTYPE), res.rates,
                              grid, C, consts.R0, prior)
            + gauge_penalty(res.potential, grid, gauge_sd)
        )
    assert abs(recomputed - res.best_loss) < 1e-6 * max(1.0, abs(res.best_loss)), (
        f"best_loss={res.best_loss!r} but the objective at the returned state is "
        f"{recomputed!r} (difference {recomputed - res.best_loss:+.6g})"
    )


def test_guard_recovers_and_plateau_stops():
    """The two safeguards fire and report themselves in ``stop_reason``.

    (1) A step size known to blow up (lr=30 diverges within a few steps here) must end
    with ``recovered@N``, finite returned params and a finite best_loss.
    (2) At the safe default lr with a generous step budget, the plateau stop must end
    the fit long before the ``steps`` cap.
    """
    torch.manual_seed(0)
    grid = dfl.GridConfig(4, 8, 24).build()
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300, .85, .85, 25, 50)
    n = 60
    gaps = torch.rand(n) * 0.01
    gaps[0] = 0.0
    cols = torch.randint(0, 2, (n,))
    batch = _OneTraceBatch(gaps, cols, n)
    prior = dfl.PriorConfig(curvature_weight=0.1)

    def fresh_pot():
        pot = dfl.build_potential(dfl.PotentialConfig(n_knots=6), grid)
        with torch.no_grad():
            pot.theta.copy_(torch.tensor([0.5, 0.0, -0.9, 0.8, -0.3, 0.4]))
        return pot

    # (1) divergence guard
    res = dfl.fit(batch, grid, fresh_pot(), C, consts.R0, D_init=10.0,
                  rates_init=rates, prior=prior, fit_D=True, verbose=False,
                  optim=dfl.OptimConfig(steps=20, lbfgs_lr=30.0, log_every=1))
    assert res.stop_reason.startswith("recovered@"), res.stop_reason
    assert torch.isfinite(res.potential.theta).all()
    assert math.isfinite(res.best_loss) and math.isfinite(res.D)

    # (2) plateau stop (or LBFGS's own convergence): well before the cap
    res = dfl.fit(batch, grid, fresh_pot(), C, consts.R0, D_init=10.0,
                  rates_init=rates, prior=prior, fit_D=True, verbose=False,
                  optim=dfl.OptimConfig(steps=600, log_every=1))
    assert res.stop_reason.startswith(("plateau@", "converged@")), res.stop_reason
    assert res.history[-1]["step"] < 599


def test_fit_enforces_mean_theta_zero_and_preserves_identified():
    """The fit's gauge anchor pins mean(theta)=0 with ZERO bias on identified quantities.

    Fit the same data twice from identical init: once with the anchor on (gauge_sd=1)
    and once with it effectively off (gauge_sd large). A proper prior determines the
    non-offset directions, so the *only* thing the anchor may change is the pure-gauge
    offset. Assert the anchor pins mean(theta)=0 while D, the grid-mean-zero shape, and
    the barrier height are unchanged.
    """
    torch.manual_seed(0)
    grid = dfl.GridConfig(4, 8, 24).build()
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300, .85, .85, 25, 50)
    n = 60
    gaps = torch.rand(n) * 0.01
    gaps[0] = 0.0
    cols = torch.randint(0, 2, (n,))
    batch = _OneTraceBatch(gaps, cols, n)

    # a shape prior on the knots; it does not constrain the pure-gauge offset, which is
    # exactly what makes it a valid control for the anchor test.
    prior = dfl.PriorConfig(curvature_weight=0.1)
    # tight stop_min_delta: the gauge term is tiny in nats, so the default 0.1-nat
    # plateau threshold would stop the fit before mean(theta) is pinned to < 5e-3.
    optim = dfl.OptimConfig(steps=600, stop_min_delta=1e-3, log_every=200)
    init_theta = torch.tensor([0.5, 0.0, -0.9, 0.8, -0.3, 0.4])

    def fresh_pot():
        pot = dfl.build_potential(dfl.PotentialConfig(n_knots=6), grid)
        with torch.no_grad():
            pot.theta.copy_(init_theta)
        return pot

    # the guarded LBFGS loop is deterministic, so the two fits differ only by the
    # gauge anchor, as the test intends.
    res_anchor = dfl.fit(batch, grid, fresh_pot(), C, consts.R0, D_init=10.0,
                         rates_init=rates, prior=prior, optim=optim, fit_D=True,
                         verbose=False, gauge_sd=1.0)
    res_base = dfl.fit(batch, grid, fresh_pot(), C, consts.R0, D_init=10.0,
                       rates_init=rates, prior=prior, optim=optim, fit_D=True,
                       verbose=False, gauge_sd=1e6)

    # (1) the anchor pins the enforcement gauge mean(theta)=0; the baseline does not
    #     (nothing else constrains the offset, so the baseline stays near its init mean).
    assert abs(float(res_anchor.potential.theta.mean())) < 5e-3
    assert abs(float(res_base.potential.theta.mean())) > 3e-2

    # (2) reporting gauge is grid-mean-zero by definition.
    U_a = dfl.recovered_potential(res_anchor.potential, grid)
    U_b = dfl.recovered_potential(res_base.potential, grid)
    assert abs(float(U_a.mean())) < 1e-8

    # (3) ZERO BIAS: D, the (grid-mean-zero) shape, and the barrier height are unchanged
    #     -- only the unobservable offset differed between the two fits.
    assert abs(res_anchor.D - res_base.D) < 1e-2 * res_base.D
    assert float(torch.sqrt(((U_a - U_b) ** 2).mean())) < 1e-2
    assert abs(float(U_a.max() - U_a.min()) - float(U_b.max() - U_b.min())) < 1e-2
