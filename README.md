# `diff_fret_likelihood`

A differentiable likelihood for **continuous-illumination smFRET photon streams**.
It scores the *actual inter-photon times* with a marked (coloured) inhomogeneous
Poisson observation model — no binning — and parameterises the free-energy
landscape `u_θ(x)` with a neural network or a cubic spline. The potential shape,
the diffusion coefficient `D`, and the photophysics are all differentiable, so
they can be estimated by gradient descent (MAP), bounded by the Cramér–Rao
inequality, or explored by HMC (full posterior).

## Model

Units are **ms / nm / kHz** throughout. Overdamped Langevin dynamics on a reduced
potential `u = U / k_BT`:

    dx = -D u'(x) dt + sqrt(2D) dW,     π(x) ∝ e^{-u(x)}

FRET efficiency and per-channel emission intensities (green `G`, red `R`):

    E(x)   = R0^6 / (R0^6 + x^6)
    μ_G(x) = a_g [C_gg(1-E) + C_rg E] + bg_g
    μ_R(x) = a_r [C_gr(1-E) + C_rr E] + bg_r

The four positive rates `a_g, a_r, bg_g, bg_r` (brightnesses and backgrounds) are
the identifiable emission parameters exposed to the optimiser; `R0` and the
crosstalk matrix `C` (with `C_gg = 1-C_gr`, `C_rr = 1-C_rg`) are fixed calibration.

The marginal likelihood of a trace with photons `c_1..c_K` at gaps `τ_1..τ_K` on a
spatial grid, using the detailed-balance Smoluchowski generator `L`, `Λ = diag(μ)`
and per-colour emission `V_c = diag(μ_c)`:

    p = 1ᵀ e^{(L-Λ)(T-t_K)} V_{c_K} e^{(L-Λ)τ_K} … V_{c_1} e^{(L-Λ)τ_1} p_0

It is evaluated in the symmetric basis (`s_i = e^{-u_i/2}`) via a single `eigh` of
the tilted generator `A = L_sym - Λ`; each inter-photon gap is then two mat-vecs
sharing that spectrum, with a running log-normaliser in float64. See `forward.py`.

## Installation

The likelihood itself is pure PyTorch. The optional trace **simulator** is a
Cython extension linking the GNU Scientific Library (GSL).

```bash
# 1. PyTorch (>= 2.1) — install the CPU/CUDA build for your platform first;
#    it is deliberately not pinned as a dependency.  See https://pytorch.org
# 2. GSL (only needed for the simulator):
#    apt-get install libgsl-dev pkg-config   # or: dnf install gsl-devel pkgconf-pkg-config
#    conda install -c conda-forge gsl pkg-config   # or: brew install gsl pkg-config

pip install .                          # portable build
DFL_NATIVE=1 pip install .             # CPU-specific fast-math build (faster, non-portable)
pip install -e . --no-build-isolation  # editable dev install (builds against your numpy)
```

Optional extras: `.[sampling]` (pyro HMC/NUTS), `.[diagnostics]` (arviz R-hat/ESS),
`.[dev]` (pytest + all optionals). If GSL is missing, `import diff_fret_likelihood`
still works — only `simulate.simulate_equilibrium` requires the compiled extension.

**float64.** Importing the package does not change global torch state. The
likelihood needs double precision (the `eigh` + log-normalisers), so call
`dfl.use_float64()` once before building anything, or pass `dtype=dfl.DTYPE`.

## Quick start — MAP fit

```python
import numpy as np
import diff_fret_likelihood as dfl

dfl.use_float64()

# --- ground-truth model (ms / nm / kHz) ---
# The simulator has no reflecting boundary, so U must confine the particle:
# use a WIDE knot domain with steep walls; score the fit on the FRET band only.
R0 = 6.0
x_knots = np.linspace(2.0, 10.0, 15)                            # nm (wide, self-confining)
y_knots = 4.0 * (((x_knots - 6.0) / 1.2) ** 2 - 1.0) ** 2       # U(x_knots) in kT: wells 4.8/7.2, 4 kT barrier
D_true  = 10.0                                                  # nm^2/ms
kD, eta_g, eta_r = 6.0, 0.85, 0.85                              # kHz, detection efficiencies
beta_g, beta_r   = 0.425, 0.85                                  # kHz, DETECTED background
C_gr, C_rg = 0.10, 0.05                                         # crosstalk

# --- simulate photon streams (CPU-only Cython simulator; run before any CUDA) ---
batch = dfl.simulate.simulate_equilibrium(
    x_knots, y_knots, D_true, R0, kD, beta_g, beta_r, eta_g, eta_r, C_gr, C_rg,
    T=150.0, dt=5e-6, n_traces=64, seed=0,
)                                                               # -> dfl.simulate.Batch

# --- fit (grid = FRET-observable band, not the wide simulation domain) ---
device = "cpu"                                                  # or "cuda"
grid   = dfl.GridConfig(x_min=3.5, x_max=8.5, n_grid=160).build(device)
pot    = dfl.build_potential(dfl.PotentialConfig(kind="spline", n_knots=9), grid).to(device)
consts = dfl.PhysicsConstants(R0=R0)                            # C_gr=0.10, C_rg=0.05 defaults
rates  = dfl.EffectiveRates.from_physics(kD, eta_g, eta_r, beta_g, beta_r, device=device)

res = dfl.fit(
    batch.to(device), grid, pot,
    C=consts.crosstalk_tensor(device), R0=consts.R0,
    D_init=5.0, rates_init=rates,
    prior=dfl.PriorConfig(curvature_weight=0.05),
    fit_D=True, fit_rates=False,
)

U = dfl.recovered_potential(res.potential, grid)                # [G] recovered U(x), grid-mean 0
print(res.D, res.best_loss)
```

`fit` returns a `FitResult` (`.potential`, `.D`, `.rates`, `.best_loss`,
`.history`). It runs Adam with a graduated-non-convexity (annealed gradient noise)
schedule to escape high-barrier basin traps; set `blur="none"` for plain Adam.
Tune the schedule via `optim=dfl.OptimConfig(adam_steps=..., adam_lr=...)`
(defaults: 300 steps, `lr=0.03`).

## Posterior sampling (HMC)

`sample.sample_posterior` draws from the full posterior with pyro NUTS/HMC,
reusing `neg_log_posterior` verbatim as the potential energy. Two requirements:

* **A proper landscape prior** — set `prior.gp_sigma`. The curvature penalty alone
  is improper (constant/linear directions unpenalised) and HMC will not mix;
  sampling raises if `gp_sigma is None`.
* Use the **spline** potential (low-dimensional; the GP prior acts on its knots).
  The additive gauge constant of `U` is pinned internally by a Gaussian anchor.

```python
prior = dfl.PriorConfig(gp_sigma=1.5, gp_lengthscale=1.0, gp_kernel="matern52")

post = dfl.sample.sample_posterior(
    batch.to(device), grid, pot,
    C=consts.crosstalk_tensor(device), R0=consts.R0,
    prior=prior, rates_init=rates, D_init=5.0,
    num_samples=1000, warmup=400, map_warmstart=True,          # start at MAP -> short burn-in
)

U_mean = post.U_mean()               # [G] posterior-mean landscape (grid-mean 0)
lo, hi = post.U_band((0.05, 0.95))   # [2, G] 90% band of U(x)
D_draws = post.D                     # [S] posterior draws of D

# multi-chain R-hat / ESS (needs arviz):
multi = dfl.sample.sample_posterior_multi(
    batch.to(device), grid, pot, C=consts.crosstalk_tensor(device), R0=consts.R0,
    prior=prior, rates_init=rates, D_init=5.0, num_chains=4, num_samples=1000,
)
print(multi.summary())
```

Mixing knobs on `sample_posterior`: `full_mass` (dense vs diagonal adapted mass),
`max_tree_depth`, `step_size`/`target_accept`, `sampler="nuts"|"hmc"`.

## Cramér–Rao bound

The lower bound on the covariance of any unbiased estimator of
`[landscape knots | D | rates]`, from the Fisher information at the given
parameters (requires a `SplinePotential`):

```python
crb = dfl.cramer_rao_bound(
    batch.to(device), grid, res.potential, res.D, res.rates,
    C=consts.crosstalk_tensor(device), R0=consts.R0,
)
print(crb.sigma_physical["D"])        # σ_D lower bound (nm^2/ms)
print(crb.sigma_physical["knots"])    # per-knot σ (kT)
```

The Gaussian **gauge anchor** (`gauge_sd`, default `1.0`, the same one `fit` uses) is
always included — the landscape offset is an exact flat direction, so without it the
information matrix is singular and only an SVD threshold stands between you and a
meaningless number. With it, `F_N + H_gauge` is inverted exactly by Cholesky. The cost
is confined to the per-knot σ, which carry `gauge_sd²` of extra variance:
`cov = pinv(F_N) + gauge_sd²·K·v vᵀ` exactly, with `v` the unit constant-knot
direction. Since `v` is zero on `logD` and the rates, **σ_D, σ_rates and every
gauge-blind functional** (barrier heights, CRB bands) are unaffected. `gauge_sd=None`
restores the unanchored pseudo-inverse and warns.

Pass a proper `prior` (with `gp_sigma`) to also add the prior's Hessian, giving the
**posterior** covariance (the analytic Laplace precision) rather than the likelihood
bound; that is what pins knots the data never see, which the anchor alone cannot do.
Evaluate at a `fit` MAP for the Laplace interpretation, or at the truth for the bound
at truth.

## Reconstructing the hidden trajectory

Running the filter once forward and once backward answers "what did the molecule
actually do?". With the backward filter `β(x,t)` (the adjoint of the tilted
generator, `β(·,T) = 1`, `β(·,t_k⁻) = μ_{c_k} β(·,t_k⁺)`), the likelihood can be
evaluated by stopping anywhere,

```
log L = log ⟨β(·,t), ρ(·,t)⟩     for any t ∈ [0, T],
```

and the posterior over the latent coordinate given *all* the data — the smoothing
distribution — is `γ(x,t) = β(x,t) ρ(x,t) / L`:

```python
res = dfl.reconstruct_trace(
    times, colors, T, res_fit.potential, res_fit.D, res_fit.rates,
    grid, consts.crosstalk_tensor(), consts.R0,
    t_out=torch.arange(0.0, T, 0.1),   # None -> report at the photon times
    n_paths=5,                         # exact posterior sample trajectories
)
res.x_mean, res.x_sd     # [M] reconstruction with error bands (nm)
res.paths                # [5, M] sampled trajectories, dynamically admissible
res.loglik, res.loglik_spread   # log L via ⟨β,ρ⟩ and its constancy over t
```

`loglik_spread` is a sharp self-test on the whole evaluator: `⟨β,ρ⟩` must be the
same at every `t` and equal `marginal_loglik` (~1e-12 in practice).

`gamma` is the full answer; the rest is one line from it — in particular

```python
E_mean = res.gamma @ dfl.fret_efficiency(res.grid, consts.R0)
```

is the model's posterior FRET efficiency, the calibrated counterpart of a binned
"apparent E(t)" trace. Note `E_mean ≠ E(x_mean)`. The pointwise mode is
`res.grid[res.gamma.argmax(-1)]`, which differs from `x_mean` exactly when `γ` is
bimodal — for a hopping molecule the *mean* sits on the barrier top, where the
molecule essentially never is; sampled `paths` are the honest single-trajectory view.

Cost is ~2× one likelihood evaluation (plus one pass per sampled path), it runs
under `no_grad` in float64, and the reconstruction near either end of the window is
prior-dominated over roughly one relaxation time. `reconstruct_batch` loops over the
traces of a `Batch`.

## Bring your own data

The likelihood consumes a `simulate.Batch` — padded per-trace tensors, all on one
device:

| field | shape | meaning |
|---|---|---|
| `ipt` | `[B, Kmax]` | inter-photon gaps (ms); `ipt[:, 0] = 0` |
| `colors` | `[B, Kmax]` | int64, `0` = green, `1` = red |
| `mask` | `[B, Kmax]` | bool; `True` for real photons |
| `lengths` | `[B]` | photons per trace |
| `T` | `[B]` | window length (ms) `= sum(ipt)` |

Build one from your own `(times, colors)` arrays by sorting each trace, taking
`ipt = [0, *diff(times)]`, and zero-padding to the longest trace.

## Priors

`PriorConfig` (evaluated in one place, `objective.prior_penalty`, so
`neg_log_posterior = -loglik + prior_penalty`). Every term is off by default —
`PriorConfig()` is pure MLE:

* `curvature_weight` — grid-invariant roughness `≈ ∫ (u'')² dx` (improper; good for MAP).
* `gp_sigma` / `gp_lengthscale` / `gp_kernel` — a **proper** stationary-kernel GP
  prior over `U(x)` (`rbf` | `matern32` | `matern52`), gauge-invariant. **Required
  for HMC.** Use it as the single shape prior (`curvature_weight=0`).
* `logD_mean` / `logD_std` — weak Gaussian prior on `log D`.
* `l2_weight` — weak L2 on the potential parameters.

Inspect active terms with `prior.active_terms()` / `prior.describe()`. Pass
`prior=None` to `fit` for a pure MLE (or `PriorConfig.none()` where an instance is
required); HMC requires a proper prior and raises on `None`.

The **gauge anchor** is deliberately *not* one of these terms. It pins the offset of
`U` (a coordinate choice, not a belief), so it is applied unconditionally by `fit`,
`fit_multi`, the sampler and `cramer_rao_bound` — independently of `prior` — and is
tuned by the separate `gauge_sd` argument.

## Potentials

`build_potential(PotentialConfig(kind=...), grid)`:

* `kind="mlp"` — smooth-activation MLP; flexible, the general-purpose choice.
* `kind="spline"` — natural-cubic potential-knot spline (`n_knots`), linear in its
  knot heights. Low-dimensional; required for `cramer_rao_bound` and the best HMC
  target. Force `-∇u` is obtained by autodiff for both.

## Performance & precision knobs

On `OptimConfig` (forwarded to the batched likelihood):

* `compile=True` (`compile_mode="default"` | `"reduce-overhead"`) — `torch.compile`
  the photon recursion; numerically transparent.
* `propagate_dtype=torch.float32` — mixed-precision recursion (fp32 mat-vecs, fp64
  `eigh` + log-normaliser). Large GPU win where fp64 is throttled.

The eigensolver falls back (CPU-LAPACK → jitter) if cuSOLVER fails to converge on
steep landscapes; gradients still flow.

## Module layout

| module | role |
|---|---|
| `config.py` | dataclasses: `GridConfig`, `PotentialConfig`, `PhysicsConstants`, `PriorConfig`, `OptimConfig` |
| `potential.py` | `u_θ`: `MLPPotential`, `SplinePotential`; force via autodiff |
| `photophysics.py` | `E(x)`, crosstalk mixing, emission rates `μ_G, μ_R`, `EffectiveRates` |
| `generator.py` | detailed-balance Smoluchowski `L`; symmetrisation; validity checks |
| `forward.py` | marginal log-likelihood (eigendecomp propagator, single + batched); robust `eigh`; compile / fp32 paths |
| `objective.py` | `neg_log_posterior` (marginal + priors); curvature / GP / `logD` / L2 priors; the gauge anchor |
| `infer.py` | `fit` (Adam + graduated non-convexity MAP); `recovered_potential` |
| `fisher.py` | `cramer_rao_bound`: Fisher information and CRB (pure or posterior) |
| `reconstruct.py` | backward filter `β`; smoothing posterior over `x(t)` with error bands; exact posterior sample paths |
| `sample.py` | HMC/NUTS posterior sampling of `U(x)`, `D`, rates (single / multi-chain) |
| `init.py` | `warmstart_potential` (fit a potential to a target profile); rough initial `EffectiveRates` |
| `simulate.py` | Cython simulator wrapper; parallel equilibrium trace generation → `Batch` |
| `utils.py` | seeding, positivity transforms, log-space helpers |

## Tests

```bash
python -m pytest tests -q
```

## Conventions & caveats

* `D` is physical (nm²/ms); the optimiser works in natural-log space (`D = e^{logD}`).
* Only ~3 nm of `x` is FRET-observable (`E` is steep for `x ≈ [4.4, 7.2]` nm at
  `R0 = 6`); score `U(x)` on that band. Far wells are FRET-saturated and unidentifiable.
* Keep `grid` to the data-visited region: a grid running into a high-potential tail
  (`U ≫ 10 kT`) destabilises the eigendecomposition gradient.
* If a trace has no absolute first-photon time or window length, the likelihood
  conditions on the first photon (`t_1 = 0`, `T = t_K`); supply explicit `t_1`/`T`
  when known.
