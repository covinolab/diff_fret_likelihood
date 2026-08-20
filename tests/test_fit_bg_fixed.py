"""``FreeRates(fit_bg=False)`` freezes the background rates but not the brightnesses.

Used to calibrate out a known background instead of inferring it jointly.  The frozen
pair must be invisible to ``parameters()`` -- the optimiser param list in ``infer.fit``
is built from it -- while still surviving ``.to()`` and still producing the right
values from ``build()``.
"""

import torch

from diff_fret_likelihood.infer import FreeRates
from diff_fret_likelihood.photophysics import EffectiveRates


def _init(a_g=2.0, a_r=2.1, bg_g=0.16, bg_r=0.40):
    f = lambda v: torch.tensor(float(v), dtype=torch.float64)
    return EffectiveRates(f(a_g), f(a_r), f(bg_g), f(bg_r))


def test_default_fits_all_four_rates():
    fr = FreeRates(_init())
    names = {n for n, _ in fr.named_parameters()}
    assert names == {"log_a_g", "log_a_r", "log_bg_g", "log_bg_r"}


def test_fit_bg_false_leaves_only_the_brightnesses_free():
    fr = FreeRates(_init(), fit_bg=False)
    names = {n for n, _ in fr.named_parameters()}
    assert names == {"log_a_g", "log_a_r"}
    # the frozen pair is still there, as buffers
    assert {n for n, _ in fr.named_buffers()} == {"log_bg_g", "log_bg_r"}


def test_frozen_bg_keeps_its_init_value_through_build():
    init = _init(bg_g=0.16, bg_r=0.40)
    fr = FreeRates(init, fit_bg=False)
    built = fr.build()
    assert abs(float(built.bg_g) - 0.16) < 1e-12
    assert abs(float(built.bg_r) - 0.40) < 1e-12


def test_optimiser_moves_brightness_but_not_background():
    """An Adam step over parameters() must leave the frozen bg bit-identical."""
    fr = FreeRates(_init(), fit_bg=False)
    opt = torch.optim.Adam(fr.parameters(), lr=0.1)

    before = fr.build()
    bg_g0, bg_r0, a_g0 = float(before.bg_g), float(before.bg_r), float(before.a_g)

    built = fr.build()
    # a loss that pulls on every rate, frozen or not
    loss = built.a_g + built.a_r + built.bg_g + built.bg_r
    loss.backward()
    opt.step()

    after = fr.build()
    assert float(after.bg_g) == bg_g0
    assert float(after.bg_r) == bg_r0
    assert float(after.a_g) != a_g0


def test_frozen_bg_carries_no_grad():
    fr = FreeRates(_init(), fit_bg=False)
    built = fr.build()
    (built.a_g + built.bg_g).backward()
    assert fr.log_a_g.grad is not None
    assert not fr.log_bg_g.requires_grad
    assert getattr(fr.log_bg_g, "grad", None) is None
