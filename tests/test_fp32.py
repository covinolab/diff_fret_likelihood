"""Stage B: mixed-precision recursion (``propagate_dtype=torch.float32``) keeps
the numerically delicate parts -- the ``eigh`` and the running log-normaliser --
in float64, and runs only the per-photon propagate/emit/normalize in float32.

These are the accuracy GATES for the fp32 flag (the real speedup is on the GPU;
the accuracy is device-independent so tested on CPU).  Value must match fp64 to
~1e-3 relative, and gradients must stay directionally identical (cosine > 0.999)
so gradient-based MAP fitting is unaffected.  The end-to-end science gate is
``tests/test_bartlett_fisher.py`` re-run under fp32 (separate, heavier).
"""

import torch

import diff_fret_likelihood as dfl


class _Batch:
    def __init__(self, ipt, colors, mask):
        self.ipt, self.colors, self.mask = ipt, colors, mask
        self.lengths = mask.sum(1)
        self.n_traces = ipt.shape[0]

    def to(self, device):
        return self


def _setup(G=24):
    grid = dfl.GridConfig(4.0, 8.0, G).build()
    pot = dfl.build_potential(dfl.PotentialConfig(kind="spline", n_knots=6), grid)
    with torch.no_grad():
        pot.theta.copy_(torch.tensor([0.3, -0.8, 0.9, -0.5, 0.7, 0.1]))
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300.0, 0.85, 0.85, 25.0, 50.0)
    return grid, pot, C, rates, consts.R0


def _batch():
    torch.manual_seed(1)
    lengths = [40, 55, 30, 48]
    Kmax = max(lengths)
    ipt = torch.zeros(len(lengths), Kmax)
    colors = torch.zeros(len(lengths), Kmax, dtype=torch.long)
    mask = torch.zeros(len(lengths), Kmax, dtype=torch.bool)
    for b, n in enumerate(lengths):
        g = torch.rand(n) * 0.01
        g[0] = 0.0
        ipt[b, :n] = g
        colors[b, :n] = torch.randint(0, 2, (n,))
        mask[b, :n] = True
    return ipt, colors, mask


def test_fp32_value_close_to_fp64():
    """Per-trace loglik in fp32 matches fp64 to ~1e-3 relative (log-normaliser
    stays fp64, so the value does not drift over the photon stream)."""
    grid, pot, C, rates, R0 = _setup()
    ipt, colors, mask = _batch()
    D = torch.tensor(10.0)
    ll64 = dfl.marginal_loglik_batch(
        ipt, colors, mask, pot, D, rates, grid, C, R0, reduce="none")
    ll32 = dfl.marginal_loglik_batch(
        ipt, colors, mask, pot, D, rates, grid, C, R0, reduce="none",
        propagate_dtype=torch.float32)
    rel = (ll32 - ll64).abs() / ll64.abs().clamp_min(1.0)
    assert rel.max() < 1e-3, f"fp32 rel err {rel.max():.2e}\n{ll64}\n{ll32}"


def test_fp32_holds_at_long_stream():
    """The fp64 log-normaliser must keep fp32 accurate at REALISTIC lengths
    (~2000 photons), where naive fp32 accumulation would drift.  This is the
    load-bearing accuracy claim for the flag."""
    grid, pot, C, rates, R0 = _setup()
    torch.manual_seed(3)
    n = 2000
    ipt = torch.zeros(3, n)
    colors = torch.zeros(3, n, dtype=torch.long)
    mask = torch.ones(3, n, dtype=torch.bool)
    for b in range(3):
        g = torch.rand(n) * 0.01
        g[0] = 0.0
        ipt[b] = g
        colors[b] = torch.randint(0, 2, (n,))
    D = torch.tensor(10.0)
    ll64 = dfl.marginal_loglik_batch(
        ipt, colors, mask, pot, D, rates, grid, C, R0, reduce="none")
    ll32 = dfl.marginal_loglik_batch(
        ipt, colors, mask, pot, D, rates, grid, C, R0, reduce="none",
        propagate_dtype=torch.float32)
    rel = (ll32 - ll64).abs() / ll64.abs().clamp_min(1.0)
    assert rel.max() < 1e-3, f"fp32 rel err at {n} photons: {rel.max():.2e}"


def test_fp32_gradient_cosine():
    """fp32 gradients stay directionally identical to fp64 (cosine > 0.999),
    so MAP optimisation steps point the same way."""
    grid, pot, C, rates, R0 = _setup()
    ipt, colors, mask = _batch()

    D = torch.tensor(10.0, requires_grad=True)
    ll64 = dfl.marginal_loglik_batch(
        ipt, colors, mask, pot, D, rates, grid, C, R0, reduce="sum")
    g64 = torch.autograd.grad(ll64, [D, pot.theta])

    D2 = torch.tensor(10.0, requires_grad=True)
    ll32 = dfl.marginal_loglik_batch(
        ipt, colors, mask, pot, D2, rates, grid, C, R0, reduce="sum",
        propagate_dtype=torch.float32)
    g32 = torch.autograd.grad(ll32, [D2, pot.theta])

    for a, b in zip(g64, g32):
        a = a.flatten().double()
        b = b.flatten().double()
        cos = (a @ b) / (a.norm() * b.norm()).clamp_min(1e-30)
        assert cos > 0.999, f"grad cosine {float(cos):.5f}"


def _fit_D(propagate_dtype):
    grid = dfl.GridConfig(4.0, 8.0, 20).build()
    pot = dfl.build_potential(dfl.PotentialConfig(kind="spline", n_knots=6), grid)
    with torch.no_grad():
        pot.theta.copy_(torch.tensor([0.2, -0.6, 0.7, -0.4, 0.5, 0.1]))
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300.0, 0.85, 0.85, 25.0, 50.0)
    torch.manual_seed(7)
    n = 300
    ipt = torch.zeros(6, n)
    colors = torch.zeros(6, n, dtype=torch.long)
    mask = torch.ones(6, n, dtype=torch.bool)
    for b in range(6):
        g = torch.rand(n) * 0.01
        g[0] = 0.0
        ipt[b] = g
        colors[b] = torch.randint(0, 2, (n,))
    optim = dfl.OptimConfig(steps=40, propagate_dtype=propagate_dtype)
    res = dfl.fit(_Batch(ipt, colors, mask), grid, pot, C, consts.R0,
                  D_init=5.0, rates_init=rates, prior=dfl.PriorConfig(curvature_weight=0.05),
                  optim=optim, fit_D=True, fit_rates=False, verbose=False)
    return res.D, res.best_loss


def test_fp32_fit_recovers_same_as_fp64():
    """End-to-end: an fp32 MAP fit reaches the same D / loss as the fp64 fit on
    identical data (the guarded LBFGS fit tolerates the ~1e-8 fp32 gradient noise)."""
    torch.manual_seed(0)
    D64, loss64 = _fit_D(None)
    torch.manual_seed(0)
    D32, loss32 = _fit_D(torch.float32)
    assert abs(D32 - D64) / D64 < 0.02, f"D fp64={D64:.4f} fp32={D32:.4f}"
    assert abs(loss32 - loss64) / abs(loss64) < 1e-3, f"loss {loss64:.3f} vs {loss32:.3f}"
