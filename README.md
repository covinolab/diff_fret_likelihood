# diff_fret_likelihood

Get a free-energy landscape out of a single-molecule FRET photon stream.

You hand it the arrival times and colours of the photons your detectors
recorded during a continuous-illumination smFRET experiment. It hands back the
free-energy landscape `U(x)` along the FRET coordinate and the diffusion
coefficient `D` of the molecule moving on it, by maximum likelihood. The photon
times are used as they were measured — nothing is binned into frames, and no
discrete set of states is assumed. The whole model is written in PyTorch and is
differentiable end to end, which is what makes the fit ordinary gradient descent
and the error bars cheap to compute.

## What you need

| | package | why |
|---|---|---|
| **required** | Python ≥ 3.10, [PyTorch](https://pytorch.org) ≥ 2.1, numpy ≥ 1.24, scipy ≥ 1.10 | the likelihood and the fit |
| **for simulated data** | GSL (`libgsl-dev`) and `pkg-config` | system libraries for the built-in trace simulator |
| optional | `pyro-ppl` ≥ 1.9 (extra `sampling`), `arviz` ≥ 0.17 (extra `diagnostics`) | full posterior sampling instead of a single fit |
| optional | `pytest`, `matplotlib` (extra `dev`) | running the test suite |

Nothing in this README needs the optional packages.

## Install

```bash
# 1. PyTorch first, in the CPU or CUDA flavour your machine wants.
#    It is deliberately NOT a pinned dependency -- see https://pytorch.org
pip install torch

# 2. GSL, only if you want to simulate traces. It is a system library, not a
#    pip package:
sudo apt-get install libgsl-dev pkg-config     # Debian / Ubuntu
# conda install -c conda-forge gsl pkg-config  # conda
# brew install gsl pkg-config                  # macOS

# 3. This package
pip install .
```

Two variants worth knowing about:

```bash
DFL_NATIVE=1 pip install .              # faster simulator, compiled for THIS cpu only
pip install -e . --no-build-isolation   # editable install, for working on the code
```

If GSL is missing, `import diff_fret_likelihood` still works and you can still
fit your own data — only the trace simulator is unavailable.

## Units

Everything is in these units, with no conversion happening anywhere. If you feed
in seconds, you will get a wrong answer rather than an error.

| quantity | unit |
|---|---|
| times, inter-photon gaps, window length | milliseconds (ms) |
| distance: `x`, the Förster radius `R0`, the grid | nanometres (nm) |
| diffusion coefficient `D` | nm² / ms |
| all rates: brightness, background, emission | kHz (= 1/ms) |
| free energy `U(x)` | kT (thermal energy units) |

Call `dfl.use_float64()` once before you build anything. The likelihood needs
double precision to be accurate; importing the package deliberately does not
change PyTorch's global settings behind your back.

## Your data: the photon trace format

Every function here takes a `Batch` — the photons of one experiment, one row per
molecule, all tensors on the same device:

| field | shape | meaning |
|---|---|---|
| `ipt` | `[B, Kmax]` | gap since the previous photon, in ms; `ipt[:, 0] = 0` |
| `colors` | `[B, Kmax]` | `0` = green photon, `1` = red photon |
| `mask` | `[B, Kmax]` | `True` where there is a real photon |
| `lengths` | `[B]` | number of photons in each trace |
| `T` | `[B]` | length of each observation window in ms |

`B` traces of different lengths are padded out to the longest one, `Kmax`, and
`mask` says which entries are real; the padding is ignored. Each trace must be
sorted in time. The first photon starts the clock (`ipt[:, 0] = 0`), so the
window is `T = ipt.sum() = t_last - t_first`.

This is all it takes to build one from your own photon lists:

```python
import numpy as np
import torch
import diff_fret_likelihood as dfl

def make_batch(traces, device="cpu"):
    """traces: list of (times, colors) per molecule. times in ms, sorted."""
    B, Kmax = len(traces), max(len(t) for t, _ in traces)
    ipt     = torch.zeros(B, Kmax, dtype=torch.float64)
    colors  = torch.zeros(B, Kmax, dtype=torch.int64)
    mask    = torch.zeros(B, Kmax, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.int64)
    T       = torch.zeros(B, dtype=torch.float64)
    for b, (t, c) in enumerate(traces):
        t, n = np.asarray(t, dtype=float), len(t)
        ipt[b, 1:n] = torch.as_tensor(np.diff(t))     # ipt[b, 0] stays 0
        colors[b, :n] = torch.as_tensor(np.asarray(c, dtype=np.int64))
        mask[b, :n] = True
        lengths[b] = n
        T[b] = t[-1] - t[0]
    return dfl.simulate.Batch(ipt, colors, mask, lengths, T).to(device)
```

That is the whole data contract. If your files give you photon times and colours
per molecule, you are ready to fit.

## Example 1 — fit a landscape

Two steps: get some data, then fit it. Here the data is simulated so the example
runs on its own and you can compare the answer to the truth; substitute
`make_batch(...)` above for your own traces.

```python
import numpy as np
import diff_fret_likelihood as dfl

dfl.use_float64()

# --- a molecule to simulate: a double well, plus the photophysics ---
R0      = 6.0                                                # Förster radius, nm
x_knots = np.linspace(2.0, 10.0, 15)                         # nm
y_knots = 4.0 * (((x_knots - 6.0) / 1.2) ** 2 - 1.0) ** 2    # U at the knots, kT
D_true  = 10.0                                               # nm^2/ms
kD, eta_g, eta_r = 6.0, 0.85, 0.85          # donor brightness (kHz), efficiencies
beta_g, beta_r   = 0.425, 0.85              # detected background per channel, kHz
C_gr, C_rg       = 0.10, 0.05               # channel crosstalk

batch = dfl.simulate.simulate_equilibrium(
    x_knots, y_knots, D_true, R0, kD, beta_g, beta_r, eta_g, eta_r, C_gr, C_rg,
    T=150.0, dt=5e-6, n_traces=32, seed=0,
)
```

The landscape is defined by its value at 15 knots between 2 and 10 nm — two
wells at 4.8 and 7.2 nm separated by a 4 kT barrier. That range is deliberately
much wider than the part FRET can see: the steep walls of `U` are the only thing
keeping the simulated molecule in, and a walker that wanders past 2 or 10 nm has
its trace thrown away.

Now the fit. The grid it is fitted on is the *narrow* window, 3.5 to 8.5 nm,
because that is the range where the FRET efficiency actually changes with
distance and the photons therefore carry information about position:

```python
grid   = dfl.GridConfig(x_min=3.5, x_max=8.5, n_grid=160).build("cpu")
pot    = dfl.build_potential(dfl.PotentialConfig(n_knots=9), grid)
consts = dfl.PhysicsConstants(R0=R0)
rates  = dfl.EffectiveRates.from_physics(kD, eta_g, eta_r, beta_g, beta_r)

res = dfl.fit(
    batch, grid, pot,
    C=consts.crosstalk_tensor(), R0=consts.R0,
    D_init=5.0,                    # starting guess for D
    rates_init=rates,              # photophysics, held fixed here
    prior=dfl.PriorConfig(curvature_weight=0.002),
    fit_D=True, fit_rates=False,
)

U = dfl.recovered_potential(res.potential, grid)   # [160] landscape in kT
print(res.D, res.stop_reason)
```

```
D = PLACEHOLDER_D nm^2/ms   (true 10.0)   stop: PLACEHOLDER_STOP
```

About PLACEHOLDER_FITTIME on a laptop CPU, plus PLACEHOLDER_SIMTIME to simulate
the traces. Reading the output:

* `U` is the landscape on the 160 grid points, shifted so its average is zero.
  Only *differences* in `U` mean anything — a landscape and the same landscape
  moved up by 3 kT describe the same physics and fit the data equally well.
* `res.D` is the fitted diffusion coefficient in nm²/ms, `res.rates` the
  photophysics (unchanged here, since `fit_rates=False`).
* `res.stop_reason` tells you how the fit ended. `plateau@N` is the normal one:
  it stopped improving at step N. `max_steps` means it ran out of steps and you
  should raise them; `recovered@N` means a step blew up and the best earlier
  state was restored, usually a sign of too little data or a grid that is too
  wide.
* `curvature_weight` is a mild smoothness preference on `U`, which keeps the
  landscape from growing wiggles the photons cannot justify. Set `prior=None`
  for a pure unpenalised maximum-likelihood fit — honest, but it needs a lot
  more photons to behave.

## Example 2 — error bars

How well is the landscape actually determined by this much data? Continuing from
Example 1:

```python
crb = dfl.cramer_rao_bound(batch, grid, res.potential, res.D, res.rates,
                           C=consts.crosstalk_tensor(), R0=consts.R0)

print(f"D = {res.D:.2f} +/- {crb.sigma_physical['D']:.2f} nm^2/ms")
for x, u, s in zip(res.potential.knots_x, res.potential.theta.detach(),
                   crb.sigma_physical["knots"]):
    print(f"U({float(x):.2f} nm) = {float(u):+.2f} +/- {s:.2f} kT")
print("unconstrained directions:", crb.null_dim)
```

```
PLACEHOLDER_CRB_OUTPUT
```

This is the Cramér–Rao bound: the smallest error bar *any* unbiased method could
achieve from this many photons. Treat it as the best case — a real fit does not
beat it and usually does slightly worse. It costs one pass over the traces
(PLACEHOLDER_CRBTIME here), which is why it is the cheap way to answer "do I
need more molecules?" before taking more data.

Two things to keep in mind when reading the numbers:

* `crb.null_dim` should be `0`. Anything larger counts directions in the
  landscape that the data does not constrain at all — normally knots sitting
  where the molecule never went, or a grid stretching past the FRET-visible
  range.
* Because the overall height of `U` is arbitrary, every per-knot `±` carries
  that arbitrary offset in it. Differences are determined much better than the
  individual numbers look: a barrier height `U(top) - U(well)` has a smaller
  error bar than adding the two `±` above would suggest.

Also available: `crb.sigma_physical` has entries for the photophysics rates
(`a_g`, `a_r`, `bg_g`, `bg_r`, in kHz), and `crb.n_traces` / `crb.n_photons`
record how much data went in.

## Tests

```bash
python -m pytest tests -q
```

## Good to know

* Only about 3 nm of distance is visible to FRET — the efficiency is steep for
  `x ≈ 4.4 … 7.2` nm when `R0 = 6` nm, and saturated outside. Fit on that band
  and do not expect to see features beyond it.
* Keep the grid inside the region the molecule actually visited. A grid reaching
  up a steep wall of `U` (beyond roughly 10 kT) makes the fit unstable.
* If your traces have no absolute start time or window length, the likelihood
  simply conditions on the first photon (`t_1 = 0`, `T = t_last`). Give it the
  real window when you know it.
* `D` is a physical number in nm²/ms wherever you see it. Internally the fit
  works with `log D`, which is why it can never wander negative.

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE).
