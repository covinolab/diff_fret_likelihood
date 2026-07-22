# `diff_fret_likelihood` — differentiable smFRET marked-point-process likelihood

A fully differentiable likelihood for **continuous-illumination smFRET photon
streams**. It scores the *actual inter-photon times* with a **marked (coloured)
inhomogeneous Poisson** observation model and parameterises the free-energy
landscape `u_θ(x)` with a neural network (or a cubic spline), so the potential
shape, the diffusion coefficient `D`, and the photophysics can all be estimated
by **gradient descent** (MAP) or explored by **HMC** (full posterior).

It reformulates the SKIPPER-FRET model (Bryan & Pressé, bioRxiv
2022.09.12.507719) in two places:

* binned-Poisson observation model → **marked point process** — the exact
  photon-by-photon likelihood, no binning, no single-state-per-window blur;
* SKI-GP prior on `U` → a differentiable **`u_θ`** (MLP or spline) with force by
  autodiff, plus an optional **proper Gaussian-process prior** over `U(x)`.

## Why a marked point process (vs. a binned HMM)

Binning photons into µs windows and running a Poisson-count HMM forces a single
emission state per window, which smears fast barrier crossings by the window
width. This package instead evaluates the exact photon-by-photon likelihood — a
continuous-potential generalisation of Gopich–Szabo — so the barrier is never
smeared by a bin. This matters here specifically because the transition paths of
interest are **microsecond**-scale.

## Model (units: ms / nm / kHz, matching `smFRET_sbi`)

Overdamped Langevin on a reduced potential `u = U/k_BT`:

    dx = -D u'(x) dt + sqrt(2D) dW,   π(x) ∝ e^{-u(x)}.

FRET / emission (matches `smFRET_sbi.channel_rates` exactly):

    E(x)   = R0^6 / (R0^6 + x^6)
    μ_G(x) = a_g [C_gg(1-E) + C_rg E] + bg_g
    μ_R(x) = a_r [C_gr(1-E) + C_rr E] + bg_r

with brightnesses `a_g=η_g kD`, `a_r=η_r kD`, backgrounds `bg_g=η_g k_gb`,
`bg_r=η_r k_rb`; `R0` and crosstalk `C` are fixed calibration.

> **Background convention.** The background is folded *inside* the η factor to
> match the simulator (`smFRET_simulator.pyx`:160-161). The SPEC writes it
> *outside* η — the **same λ family** (`bg` is a free positive rate), so the
> likelihood value is identical; only the parameter's interpretation differs.
>
> **Freed emission parameters.** The four positive brightnesses
> `a_g, a_r, bg_g, bg_r` are freed *independently* (the identifiable emission
> combos), rather than a single `kD` with η fixed. One DOF more permissive, still
> fully identifiable.

Marginal likelihood on a spatial grid (the primary evaluator), with the
detailed-balance Smoluchowski generator `L`, `Λ=diag(μ)`, `V_c=diag(μ_c)`:

    p = 1ᵀ e^{(L-Λ)(T-t_K)} V_{c_K} e^{(L-Λ)τ_K} … V_{c_1} e^{(L-Λ)τ_1} p_0.

Evaluated in the symmetric basis (`s_i = e^{-u_i/2}`) via one `eigh` of the
symmetric tilted generator `A = L_sym - diag(μ)`; each inter-photon gap is then
two mat-vecs sharing the spectrum, with a running log-normaliser (float64
throughout). See `forward.py` for the derivation in the module docstring.

## Layout

| module | role |
|---|---|
| `config.py` | dataclasses: `GridConfig`, `PotentialConfig`, `PhysicsConstants`, `PriorConfig`, `OptimConfig` |
| `potential.py` | `u_θ`: `MLPPotential` (primary) or `SplinePotential` (natural-cubic, linear in knots); force via autodiff |
| `photophysics.py` | `E(x)`, crosstalk mixing, emission rates `μ_G, μ_R`; `EffectiveRates` |
| `generator.py` | detailed-balance Smoluchowski `L`; symmetrisation; `assert_generator_valid` checks |
| `forward.py` | **primary** marginal log-lik (eigendecomp propagator, single + batched); robust `eigh`; `torch.compile` / fp32 paths |
| `objective.py` | `neg_log_posterior` (marginal + priors); curvature & **GP** priors; secondary complete-data joint objective |
| `dynamics.py` | Euler–Maruyama transition density (used only by the joint objective) |
| `infer.py` | `fit`: Adam→LBFGS MAP; `recovered_potential`, `posterior_occupancy` diagnostics |
| `sample.py` | **HMC/NUTS posterior sampling** (pyro) of `U(x)`, `D`, rates; single- and multi-chain (R-hat/ESS) |
| `init.py` | data-driven / external warm-starts: histogram landscape `u≈-log π̂`, `D` from autocorrelation, profile fitting |
| `simulate.py` | thin wrapper around the in-project Cython `simulator` (GSL); parallel equilibrium trace generation |
| `utils.py` | seeding, positivity transforms, log-space helpers |

## Installation

Installed from source; the Cython simulator is compiled locally.

**System prerequisite — GSL.** The simulator links the GNU Scientific Library,
which is *not* pip-installable. Install it first:

```
Debian/Ubuntu : sudo apt-get install libgsl-dev pkg-config
Fedora/RHEL   : sudo dnf install gsl-devel pkgconf-pkg-config
conda-forge   : conda install -c conda-forge gsl pkg-config
macOS (brew)  : brew install gsl pkg-config
```

If GSL lives in a non-standard location, set `GSL_DIR=/path/to/gsl`.

**Prerequisite — PyTorch.** Install the build matching your platform (CPU or a
specific CUDA version) from <https://pytorch.org> *before* installing this
package — torch is deliberately not pinned as a dependency. Requires `torch >= 2.1`.

**Install:**

```bash
pip install .                          # portable build
DFL_NATIVE=1 pip install .             # CPU-specific fast-math build (faster, non-portable)
pip install -e . --no-build-isolation  # editable dev install (builds against your numpy)
```

Optional extras: `pip install ".[sampling]"` (pyro HMC/NUTS),
`".[diagnostics]"` (arviz R-hat/ESS), `".[dev]"` (pytest + all optionals).

> If you hit a numpy ABI error at import, reinstall with `--no-build-isolation`
> so the extension builds against your installed numpy.

## Requirements

* Python ≥ 3.10, `torch >= 2.1` (prerequisite, see above), `numpy`, `scipy`
* `pyro-ppl` — optional, for `sample.py` (HMC/NUTS)
* `arviz` — optional, for R-hat / ESS diagnostics on posterior chains

**float64.** Importing the package does **not** change global torch state. The
likelihood is designed for double precision (the `eigh` + log-normalisers need
it), so call `dfl.use_float64()` once before building configs / running a fit, or
pass `dtype=dfl.DTYPE` explicitly. (The test suite sets this in `conftest.py`.)

## Quick start — MAP fit

```python
import diff_fret_likelihood as dfl

# 1. simulate real photon streams from a known landscape (CPU, before any CUDA)
batch, _ = dfl.simulate.simulate_traces(cfg, theta_gt, "equilibrium", n_traces=40)
consts, rates = dfl.simulate.constants_from_theta(theta_gt, num_knots, device="cuda")

# 2. fit on GPU
grid = dfl.GridConfig(4.0, 8.0, 140).build("cuda")
pot  = dfl.build_potential(dfl.PotentialConfig(kind="mlp"), grid).to("cuda")
res  = dfl.fit(
    batch.to("cuda"), grid, pot,
    C=consts.crosstalk_tensor("cuda"), R0=consts.R0,
    D_init=3.0, rates_init=rates,
    prior=dfl.PriorConfig(curvature_weight=0.05),
    optim=dfl.OptimConfig(adam_steps=300, lbfgs_steps=50),  # defaults are tiny
    fit_D=True, fit_rates=False,
)
U_rec = dfl.recovered_potential(res.potential, grid)   # min-zero recovered U(x)
```

`fit` returns a `FitResult` (`.potential`, `.D`, `.rates`, `.final_loss`,
`.history`). `OptimConfig` defaults to a very short schedule (10 Adam steps, no
LBFGS) — pass a real schedule as above for production fits.

## Quick start — posterior sampling (HMC)

Where `fit` gives a point estimate, `sample.sample_posterior` draws from the full
posterior with pyro NUTS/HMC. It reuses `neg_log_posterior` verbatim as the
potential energy (`potential_fn = -log_prob`), so it is a thin driver, not a
re-implementation. Two requirements make sampling well-posed:

* **A proper landscape prior** — set `prior.gp_sigma`. The curvature penalty
  alone is the improper thin-plate limit (constant+linear directions unpenalised)
  and HMC will not mix; `sample_posterior` raises if `gp_sigma is None`.
* **A gauge anchor** — the additive constant of `U` is unidentified; a tight
  Gaussian anchor (`gauge_sd`) pins that pure-gauge direction. Handled internally.

Target the **spline** potential (low-dimensional; the GP prior acts directly on
its knots). The MLP is supported but a poor HMC target.

```python
grid  = dfl.GridConfig(4.0, 8.0, 120).build("cuda")
pot   = dfl.build_potential(dfl.PotentialConfig(kind="spline", n_knots=9), grid).to("cuda")
prior = dfl.PriorConfig(curvature_weight=0.0,          # let the GP be the single shape prior
                        gp_sigma=1.5, gp_lengthscale=1.0, gp_kernel="matern52")

post = dfl.sample.sample_posterior(
    batch, grid, pot, C=consts.crosstalk_tensor("cuda"), R0=consts.R0,
    prior=prior, rates_init=rates, D_init=3.0,
    num_samples=1000, warmup=400, map_warmstart=True,  # start at the MAP -> short burn-in
)
U_mean = post.U_mean()          # [G] posterior-mean landscape (min-zero)
lo, hi = post.U_band((0.05, 0.95))   # 90% band of U(x)
D_draws = post.D                # [S] posterior draws of D

# multi-chain for R-hat / ESS (needs arviz):
multi = dfl.sample.sample_posterior_multi(batch, grid, pot, C=..., R0=...,
                                          prior=prior, rates_init=rates,
                                          num_chains=4, num_samples=1000)
print(multi.summary())          # arviz R-hat / ESS table
```

Mixing knobs are documented on `sample_posterior`: `full_mass` (dense vs.
diagonal adapted mass matrix), `max_tree_depth`, `step_size`/`target_accept`,
`sampler="nuts"|"hmc"`, and over-dispersion (`overdisperse`) for multi-chain
R-hat.

## Initialisation (optional but recommended)

The marginal objective is non-convex in the landscape parameters, so `init.py`
provides rough, pure warm-starts applied *before* `fit` (no change to the fit
path):

```python
from diff_fret_likelihood import init
init.warmstart_potential(pot, grid, init.occupancy_hist_init(batch, grid, R0))  # u ≈ -log π̂
D0 = init.estimate_D_init(batch, grid, R0)     # D ≈ Var(x)/τ_c from the FRET autocorrelation
# or fit an external profile (grid array, (x,u) pair, or callable u(x)):
init.warmstart_potential(pot, grid, init.resolve_u_target((x_ext, u_ext), grid, R0))
```

These estimates are deliberately coarse (apparent efficiency is biased by
crosstalk/background; binning has its own timescale) — they are starting points
the fit refines.

## Priors

`PriorConfig` (defined in `config.py`; evaluated by `objective.prior_penalty`):

* `curvature_weight` — grid-invariant roughness `≈ ∫ (u'')² dx` (improper
  thin-plate limit; good for MAP).
* `gp_sigma` / `gp_lengthscale` / `gp_kernel` — a **proper** stationary-kernel GP
  prior over `U(x)` (`rbf` | `matern32` | `matern52`), mean-centered so it stays
  gauge-invariant. Makes the landscape posterior proper — **required** for HMC.
  Recommendation: use the GP as the *single* shape prior (`curvature_weight=0`);
  don't stack two strong smoothness priors.
* `logD_mean` / `logD_std` — optional weak Gaussian prior on `log D`.
* `l2_weight` — optional weak L2 on the potential parameters.

The prior enters the objective in exactly one place, `objective.prior_penalty`, so
`neg_log_posterior = -loglik + prior_penalty(...)`. Each term is toggled by a
sentinel (weight `0` or `None`), and `PriorConfig()` is **prior-free by default**
(`curvature_weight=0.0`, all terms off) — switch on the terms you want. Inspect
which terms are on with `prior.active_terms()` / `prior.describe()`.

**True MLE.** Pass `prior=None` to `fit` / `neg_log_posterior` for a pure
maximum-likelihood fit (no regularisation; `prior_penalty` returns `0`), or use
`PriorConfig.none()` where a `PriorConfig` instance is required:

```python
res_mle = fit(batch, grid, pot, C, R0, D_init=D0, rates_init=rates, prior=None)
res_map = fit(batch, grid, pot, C, R0, D_init=D0, rates_init=rates,
              prior=PriorConfig(curvature_weight=0.05))
```

`prior=None` is only for point estimation; HMC sampling (`sample_posterior`)
requires a proper prior (`gp_sigma` set) and raises on `None`.

## Performance knobs

Set on `OptimConfig` (forwarded through the objective to `marginal_loglik_batch`):

* `compile=True` (`compile_mode="default"` / `"reduce-overhead"`) —
  `torch.compile` the photon recursion; numerically transparent
  (`tests/test_compile.py`).
* `propagate_dtype=torch.float32` — mixed-precision recursion (fp32 matmuls, fp64
  `eigh` and log-normaliser). Large GPU win where fp64 is throttled;
  accuracy-gated (`tests/test_fp32.py`, `tests/test_bartlett_fisher.py`).

`forward._robust_eigh` falls back CPU-LAPACK→jitter if cuSOLVER fails to converge
on the steep landscapes HMC transiently proposes — gradients still flow.

## Two objectives

* **Primary** — the marginal photon-stream likelihood (`forward.py`,
  `objective.neg_log_posterior`). This is what `fit` and `sample` optimise.
* **Secondary** — the complete-data (joint path + photons) log-likelihood
  (`objective.complete_data_loglik`, `dynamics.py`), kept for the
  joint-vs-marginal `D`-bias diagnostic; not used in the main fit.

## Tests

```bash
python -m pytest tests -q
```

Gates: force `= -∇u` (gradcheck) and gauge invariance (`test_potential`);
generator detailed balance / reflecting BCs / Boltzmann stationary / NSD spectrum
(`test_generator`); **G=1 Poisson identity**, fast-path `==` matrix-exp reference,
**2-state Gopich–Szabo** match, batched `==` single, marginal-likelihood gradcheck
(`test_forward`); grid convergence (`test_grid_convergence`); GP-prior math /
gauge / backward-compat (`test_gp_prior`); HMC log-prob grads / gauge anchor /
short-chain round-trip (`test_sample`); initializers (`test_init`); compile and
fp32 numerical transparency (`test_compile`, `test_fp32`); forward-backward
occupancy self-consistency (`test_infer`); and a **Bartlett + Fisher
score-at-truth** validation of the equilibrium likelihood (`test_bartlett_fisher`).

## Conventions & caveats

* The `smFRET_sbi` wrapper drops the absolute first-photon time (`ipt[0]=0`) and
  stores no window length, so the likelihood conditions on the first photon:
  `t_1=0`, `T=t_K` (leading/trailing survival gaps = 0). Explicit `t_1`/`T` can be
  supplied when known.
* `D` is a *physical* value in nm²/ms. The optimiser works in **natural**-log
  space (`D=e^{logD}`); the `smFRET_sbi` simulator separately takes **base-10**
  `log10(D)` as input — the two conventions must not be conflated.
* Only ~3 nm of `x` is FRET-observable (`E` steep for `x≈[4.4,7.2]` nm at R0=6);
  score `U(x)` on that band. Far wells are FRET-saturated and unidentifiable.
* Binding-mode transition paths are a *conditioned* (Doob) ensemble, not free
  Langevin; expect an upward barrier / `D` bias vs. the clean equilibrium fit.

## See also

* A self-contained example notebook (equilibrium recovery + inference
  walkthrough) accompanies this package.
