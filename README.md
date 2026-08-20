# diff_fret_likelihood

Get a free-energy landscape out of a single-molecule FRET photon stream.

You hand it the arrival times and colours of the photons your detectors recorded
during a continuous-illumination smFRET experiment. It hands back the free-energy
landscape `U(x)` along the FRET coordinate and the diffusion coefficient `D` of
the molecule moving on it, by maximum likelihood. The photon times are used as
they were measured — nothing is binned into frames, and no discrete set of states
is assumed. The whole model is written in PyTorch and is differentiable end to
end, which is what makes fitting it ordinary gradient descent and makes error
bars cheap to compute.

## What you need

| | package | what for |
|---|---|---|
| **required** | Python ≥ 3.10, [PyTorch](https://pytorch.org) ≥ 2.1, numpy ≥ 1.24, scipy ≥ 1.10 | the likelihood and the fit |
| **for simulated data** | GSL (`libgsl-dev`) and `pkg-config` | system libraries the built-in trace simulator compiles against |
| optional | `pyro-ppl` ≥ 1.9 (extra `sampling`), `arviz` ≥ 0.17 (extra `diagnostics`) | sampling the full posterior instead of a single fit |
| optional | `pytest`, `matplotlib` (extra `dev`) | running the test suite |

Nothing in this README needs the optional packages.

## Install

```bash
# 1. PyTorch first, in the CPU or CUDA flavour your machine wants. It is
#    deliberately NOT a pinned dependency -- see https://pytorch.org
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

Everything is in these units, and no conversion happens anywhere. Feed in
seconds and you get a wrong answer rather than an error.

| quantity | unit |
|---|---|
| times, inter-photon gaps, window length | milliseconds (ms) |
| distance: `x`, the Förster radius `R0`, the grid | nanometres (nm) |
| diffusion coefficient `D` | nm² / ms |
| all rates: brightness, background, emission | kHz (= 1/ms) |
| free energy `U(x)` | kT (thermal energy units) |

Call `dfl.use_float64()` once before building anything. The likelihood needs
double precision to be accurate, and importing the package deliberately does not
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
| `T` | `[B]` | length of each observation window, in ms |

`B` traces of different lengths are padded out to the longest one, `Kmax`, and
`mask` says which entries are real photons; the padding is ignored. Each trace
must be sorted in time. The first photon starts the clock (`ipt[:, 0] = 0`), so
the window is `T = ipt.sum() = t_last - t_first`.

This is all it takes to build a `Batch` from your own photon lists:

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

Two steps: get some data, then fit it. The data here is simulated so the example
runs on its own and you can compare the answer to the truth — swap in
`make_batch(...)` from above for your own traces.

```python
import numpy as np
import diff_fret_likelihood as dfl

dfl.use_float64()

# --- the molecule to simulate: a double well, plus the photophysics ---
R0      = 6.0                                                # Förster radius, nm
x_knots = np.linspace(2.0, 10.0, 15)                         # nm
y_knots = 4.0 * (((x_knots - 6.0) / 1.2) ** 2 - 1.0) ** 2    # U at the knots, kT
D_true  = 10.0                                               # nm^2/ms
kD, eta_g, eta_r = 6.0, 0.85, 0.85          # donor brightness (kHz), efficiencies
beta_g, beta_r   = 0.425, 0.85              # detected background per channel, kHz
C_gr, C_rg       = 0.10, 0.05               # channel crosstalk

batch = dfl.simulate.simulate_equilibrium(
    x_knots, y_knots, D_true, R0, kD, beta_g, beta_r, eta_g, eta_r, C_gr, C_rg,
    T=150.0, dt=5e-6, n_traces=128, seed=0,
)
# 128/128 traces in 15.0s | photons/trace min/med/max = [867 956 1029]
```

That landscape is two wells at 4.8 and 7.2 nm with a 4 kT barrier between them,
defined by its height at 15 knots from 2 to 10 nm. The knot range is deliberately
much wider than the part FRET can see: the steep walls of `U` are the only thing
keeping the simulated molecule from wandering off, and a walker that leaves the
range has its trace discarded.

Now the fit. It runs on a *narrow* grid, 3.5 to 8.5 nm, because that is where the
FRET efficiency actually changes with distance and the photons therefore say
something about position:

```python
grid   = dfl.GridConfig(x_min=3.5, x_max=8.5, n_grid=160).build("cpu")
pot    = dfl.build_potential(dfl.PotentialConfig(n_knots=9), grid)
consts = dfl.PhysicsConstants(R0=R0)
rates  = dfl.EffectiveRates.from_physics(kD, eta_g, eta_r, beta_g, beta_r)
prior  = dfl.PriorConfig(curvature_weight=0.002)   # mild smoothness on U

res = dfl.fit(
    batch, grid, pot,
    C=consts.crosstalk_tensor(), R0=consts.R0,
    D_init=5.0,                    # starting guess for D
    rates_init=rates,              # photophysics, held fixed here
    prior=prior,
    fit_D=True, fit_rates=False,
)

U = dfl.recovered_potential(res.potential, grid)   # [160] landscape in kT
print(res.D, res.stop_reason)
```

```
D = 7.47 nm^2/ms  (true 10.0)   stop: plateau@67
```

About 2 to 3 minutes on a 24-core CPU, plus ~16 s to simulate the traces.
Reading the output:

* `U` is the landscape on the 160 grid points, shifted so its average is zero.
  Only *differences* in `U` mean anything: the same landscape moved up by 3 kT is
  the same physics and fits the data equally well.
* `res.D` is the fitted diffusion coefficient in nm²/ms, `res.rates` the
  photophysics (unchanged here, because `fit_rates=False`).
* `res.stop_reason` says how the fit ended. `plateau@N` is the normal one: it
  stopped improving at step N. `max_steps` means it ran out of steps, so raise
  them with `optim=dfl.OptimConfig(steps=...)`. `recovered@N` means a step blew
  up and the best earlier state was restored — usually too little data, or a grid
  reaching too far.
* `prior` asks for a mildly smooth `U`, which stops the landscape growing wiggles
  the photons cannot justify. `prior=None` drops it for a pure unpenalised
  maximum-likelihood fit, and on this dataset that is a real trade rather than a
  disaster: it finds a marginally better likelihood and a barrier of 4.2 kT, but
  returns `D = 15` instead of 7.5, with error bars 7× wider on the barrier and
  15× wider on `D` (next section).
  The smoothness assumption buys precision by accepting a little bias. Pick its
  weight by comparing the likelihood on data you did *not* fit — not against a
  truth you would not have in a real experiment.

With 128 traces the recovered barrier is 3.2 kT of the true 4, and `D` is 7.5 of
the true 10. Fewer traces is faster and much worse: with 32 the same fit returns
a 0.3 kT barrier and `D = 2.6`, i.e. the landscape has washed out entirely.

Run-to-run the last digits move a little — the likelihood is very flat in `D`, so
two runs on the same data can stop at 7.47 and 7.51.

## Example 2 — error bars

How well does this much data really pin the landscape down? Continuing from
Example 1:

```python
crb = dfl.cramer_rao_bound(batch, grid, res.potential, res.D, res.rates,
                           C=consts.crosstalk_tensor(), R0=consts.R0,
                           prior=prior)

print(f"D = {res.D:.2f} +/- {crb.sigma_physical['D']:.2f} nm^2/ms")
for x, u, s in zip(res.potential.knots_x, res.potential.theta.detach(),
                   crb.sigma_physical["knots"]):
    print(f"U({float(x):.2f} nm) = {float(u):+.2f} +/- {s:.2f} kT")
print("unconstrained directions:", crb.null_dim)
```

```
D = 7.47 +/- 12.61 nm^2/ms
U(3.50 nm) = +1.56 +/- 6.14 kT
U(4.12 nm) = -0.40 +/- 2.94 kT
U(4.75 nm) = -1.82 +/- 2.56 kT
U(5.38 nm) = -1.39 +/- 2.22 kT
U(6.00 nm) = +1.38 +/- 3.12 kT
U(6.62 nm) = -0.87 +/- 3.54 kT
U(7.25 nm) = -2.28 +/- 2.37 kT
U(7.88 nm) = +0.71 +/- 4.21 kT
U(8.50 nm) = +3.10 +/- 11.13 kT
unconstrained directions: 0
```

That takes under 10 seconds — one pass over the traces — which is what makes it
the cheap way to answer "do I need more molecules?" *before* spending a week
taking more data. The numbers are the Cramér–Rao bound: the smallest error bar
*any* unbiased method could achieve from this many photons. Treat it as the best
case, not as a promise about your fit.

The `± 12.6` on `D` is the honest headline here: 122,000 photons from dyes this
dim do not pin the diffusion coefficient down at all, even though the fit
returned a plausible-looking 7.5. Three things to know when reading the rest:

* Pass the same `prior` you fitted with. Without it the bound has to allow for
  all the wiggly landscapes the smoothness assumption ruled out, and at this data
  size that inflates every number by a factor of 20 to 80.
* The error bars cover the photophysics too — the brightness and background are
  scored alongside the landscape even though the fit held them fixed. A separately
  calibrated background makes them tighter.
* The overall height of `U` is arbitrary, so every per-knot `±` carries that
  arbitrariness inside it. **Differences are much better determined than the
  individual numbers suggest** — which is why you should quote a barrier height
  rather than a knot value:

```python
top, well = 4, 2       # knots at 6.00 nm (barrier top) and 4.75 nm (well)
theta = res.potential.theta.detach()
var = crb.cov[top, top] + crb.cov[well, well] - 2 * crb.cov[top, well]
print(f"barrier = {float(theta[top] - theta[well]):+.2f} "
      f"+/- {float(var.sqrt()):.2f} kT")
```

```
barrier = +3.20 +/- 1.91 kT        (true 4.00)
```

± 1.91 kT, against 5.68 kT if you had naively added the two knot error bars.

`crb.sigma_physical` also has entries for the photophysics rates (`a_g`, `a_r`,
`bg_g`, `bg_r`, in kHz), and `crb.n_traces` / `crb.n_photons` record how much
data went in. `crb.null_dim` should come out `0`; anything larger counts
directions in the landscape the data does not constrain at all, normally knots
sitting where the molecule never went.

## Example 3 — the full posterior

The fit gives one landscape and the Cramér–Rao bound gives the best error bar any
method could manage. Between them sits the question they cannot answer: what does
the *posterior* actually look like — is it a tidy ellipse around the fit, or
something long and banana-shaped that a single number cannot summarise?
`sample_posterior` answers that by running Hamiltonian Monte Carlo (pyro's NUTS)
over the same objective the fit minimises. Continuing from Example 1:

```python
prior = dfl.PriorConfig(curvature_weight=0.002,
                        bg_g_mean=beta_g, bg_g_sd=0.1 * beta_g,   # calibrated background
                        bg_r_mean=beta_r, bg_r_sd=0.1 * beta_r)

post = dfl.sample.sample_posterior(
    batch, grid, pot, C=consts.crosstalk_tensor(), R0=consts.R0, prior=prior,
    num_samples=400, warmup=400,
    compile_mode="default", propagate_dtype=torch.float32,   # much faster per gradient
)

U_mean = post.U_mean()                # [G] posterior-mean landscape, grid-mean 0
lo, hi = post.U_band((0.05, 0.95))    # [2, G] 90% band of U(x)
print(f"D = {float(post.D.median()):.2f} "
      f"[{float(post.D.quantile(0.05)):.2f}, {float(post.D.quantile(0.95)):.2f}]")
```

You do not have to set up anything before that call. `rates_init` defaults to
`init.stream_rates`, the landscape is initialised from the FRET histogram
(`init.kde_potential_init`), and the chain then starts from a quick MAP fit —
which is what keeps burn-in short. `kde_bin_ms=...` pins the histogram bin width
and skips the held-out scan that chooses it, which is the expensive part of the
warm start.

**The sampler targets exactly the objective `fit` minimises**, negated. It adds no
prior of its own: the curvature and background terms come from the `PriorConfig`
you pass, exactly as they do for `dfl.fit`, and `prior=None` is a pure
maximum-likelihood target here just as it is there. So a chain and a fit on the
same `prior` and `gauge_sd` describe the same posterior — the fit reports its
mode, the chain its shape.

Multiple chains, which is how you find out whether to believe any of it:

```python
multi = dfl.sample.sample_posterior_multi(
    batch, grid, pot, C=consts.crosstalk_tensor(), R0=consts.R0, prior=prior,
    num_chains=4, num_samples=400, warmup=400,
    compile_mode="default", propagate_dtype=torch.float32,
)
print(multi.summary())        # arviz: R-hat and ESS per parameter
```

Read `r_hat` first. Anything above about 1.01 means the chains have not agreed and
the band is not yet a posterior — give it more warmup, or narrow the grid.

Things worth knowing before you trust a band:

* **It is much slower than the fit.** Every leapfrog step costs one gradient of the
  full likelihood, and NUTS takes tens of them per draw, so a chain is hundreds of
  times the work of a fit. `compile_mode="default"` and
  `propagate_dtype=torch.float32` are both worth setting.
* **The curvature prior does not make the landscape posterior proper.** It
  penalises roughness and leaves the constant and linear directions of `U` free.
  The constant is pinned internally by the gauge anchor; the tilt is left to the
  photons. Keep the grid inside the region they actually inform — the same advice
  as for the fit, and it matters more here.
* **Only differences in `U` are meaningful**, so read the band as a band on shape.
  A barrier height is the honest thing to quote, exactly as in Example 2.
* `curvature_norm="l1"` is a poor choice for sampling and warns: it is not
  differentiable where the curvature crosses zero, and the leapfrog integrator
  cannot cross that kink cleanly. Use the default `"l2"`.
* **`fit_bg=False`** holds the backgrounds at their calibrated values instead of
  sampling them. `fit_rates` is a warm-start knob only — the chain always samples
  the brightnesses, since marginalising over the photophysics is the point.

## Tests

```bash
python -m pytest tests -q                 # everything
python -m pytest tests -q -m "not slow"   # skips the multi-minute statistical tests
```

## Good to know

* Only about 3 nm of distance is visible to FRET — the efficiency is steep for
  `x ≈ 4.4 … 7.2` nm when `R0 = 6` nm, and saturated outside it. Fit on that band
  and do not expect to see features beyond it.
* Keep the grid inside the region the molecule actually visited. A grid running up
  a steep wall of `U` (past roughly 10 kT) makes the fit unstable.
* If your traces have no absolute start time or window length, the likelihood
  conditions on the first photon (`t_1 = 0`, `T = t_last`). Give it the real
  window when you know it.
* `D` is a physical number in nm²/ms wherever you see it. Internally the fit
  works with `log D`, which is why it can never wander negative.

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE).
