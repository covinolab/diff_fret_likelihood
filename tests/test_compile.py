"""Stage A: ``torch.compile`` of the photon recursion must be numerically
transparent -- the compiled path equals the eager path in both value and
gradient (the default path, ``compile_mode=None``, stays byte-identical).

The correctness property is device-agnostic, so these run on CPU (inductor
``default`` mode) as part of the suite; the GPU ``reduce-overhead`` (CUDA-graph)
path and the actual speedup are validated by ``benchmarks/`` (not a unit test,
since wall-clock is environment/noise dependent).
"""

import pytest
import torch

import diff_fret_likelihood as dfl


def _setup(G=12):
    grid = dfl.GridConfig(4.0, 8.0, G).build()
    pot = dfl.build_potential(dfl.PotentialConfig(n_knots=5), grid)
    with torch.no_grad():
        pot.theta.copy_(torch.tensor([0.5, -0.3, 0.8, -0.2, 0.4]))
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300.0, 0.85, 0.85, 25.0, 50.0)
    return grid, pot, C, rates, consts.R0


def _batch():
    torch.manual_seed(0)
    lengths = [6, 9, 4]
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


def test_compiled_batch_equals_eager():
    """compile_mode='default' per-trace loglik == eager (compile_mode=None)."""
    grid, pot, C, rates, R0 = _setup()
    ipt, colors, mask = _batch()
    D = torch.tensor(10.0)
    eager = dfl.marginal_loglik_batch(
        ipt, colors, mask, pot, D, rates, grid, C, R0, reduce="none")
    comp = dfl.marginal_loglik_batch(
        ipt, colors, mask, pot, D, rates, grid, C, R0, reduce="none",
        compile_mode="default")
    assert torch.allclose(eager, comp, atol=1e-8, rtol=1e-6), f"{eager} vs {comp}"


def test_compiled_grad_equals_eager():
    """Gradients (wrt D and spline theta) match between compiled and eager."""
    grid, pot, C, rates, R0 = _setup()
    ipt, colors, mask = _batch()

    D = torch.tensor(10.0, requires_grad=True)
    lle = dfl.marginal_loglik_batch(
        ipt, colors, mask, pot, D, rates, grid, C, R0, reduce="sum")
    ge = torch.autograd.grad(lle, [D, pot.theta])

    D2 = torch.tensor(10.0, requires_grad=True)
    llc = dfl.marginal_loglik_batch(
        ipt, colors, mask, pot, D2, rates, grid, C, R0, reduce="sum",
        compile_mode="default")
    gc = torch.autograd.grad(llc, [D2, pot.theta])

    for a, b in zip(ge, gc):
        assert torch.allclose(a, b, atol=1e-8, rtol=1e-6), f"{a} vs {b}"
