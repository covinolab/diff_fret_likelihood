# Changelog

## 0.3.0

Removes the **GP prior**, the **logD prior** and **`l2_weight`** — three prior terms that
no analysis ever switched on. **No numeric path changed**: every result produced with
0.1.0/0.2.0 remains valid and nothing needs re-running. The code behind the published
results is still tagged `v0.1.0-results`; the state immediately before this change is
tagged `v0.2.0-pre-gp-removal`.

Checked, not asserted. `regression_gate.py` was re-baselined at 0.2.0 and re-run: **66
arrays bit-identical**, 2 within `1e-12` (the two `torch.linalg.lstsq` calls, which are
not reproducible on identical inputs even single-threaded), 7 removed on purpose. The
published posterior CRB still recomputes against the stored `crb_posterior.npz`.

The evidence that removal is inert: all **51** archived run configs under
`diff_fret_analysis/results/**/config.yaml` record `gp_sigma: null`, `logD_mean: null`
and `l2_weight: 0.0` — not just the live configs, but a per-run snapshot of what each
experiment actually used. Both terms were Python-gated (`if prior.gp_sigma is not None:`),
so deleting a branch that never executed cannot change the arithmetic or its order.

### Removed

* `objective.gp_penalty`, `objective._gp_corr`, `objective._interp`,
  `objective.logD_penalty`, and the three now-dead branches of `objective.prior_penalty`.
* `PriorConfig`: `gp_sigma`, `gp_lengthscale`, `gp_kernel`, `gp_n_ctrl`, `gp_mean`,
  `gp_jitter`, `logD_mean`, `logD_std`, `l2_weight` — nine fields, plus their
  `__post_init__` validation and their `active_terms()` entries. **BREAKING** for any
  caller that passed them.

`PriorConfig` is now exactly the two terms the analysis uses: the curvature roughness
prior (`curvature_weight` + `curvature_norm`) and the background Gamma prior (`bg_*`).

### `curvature_norm` is NOT affected

Worth stating because the names invite confusion. `curvature_norm: "l2" | "l1"` selects
the norm applied to the *curvature* (second difference of the knot heights) inside
`curvature_penalty_spline`, and is untouched — `fit.build_prior` still reads it and
`tune_curvature.py` still records it. The removed `l2_weight` was a different thing
entirely: `sum(theta²)`, shrinkage of the landscape toward `U ≡ 0` rather than toward
smoothness. It dates from the initial commit, when `potential.parameters()` was the MLP's
weights and this was ordinary weight decay; on a spline it also penalised `mean(theta)`,
the pure-gauge offset, working against `gauge_penalty`.

### Consequence: there is no longer a proper landscape prior

The GP prior was the only one. The surviving curvature prior is the improper thin-plate
limit — it penalises roughness and leaves the constant and linear directions of `U` free.
Docstrings in `fisher.py` that promised "a finite bound on every knot" from a landscape
prior have been corrected; the practical remedy is to narrow the grid to the informed
region, which is what the published fits do (they reach `null_dim = 0`).

### `sample.py` is knowingly non-functional

It is being rewritten and will not use `PriorConfig`, so it was deliberately left
untouched and still reads `prior.gp_sigma` / `replace(prior, logD_mean=…)`. It raises on
first use. `tests/test_sample.py` is skipped at module level (9 tests) until the rewrite
lands.

To recover the removed implementations:
`git show v0.2.0-pre-gp-removal:objective.py`,
`git show v0.2.0-pre-gp-removal:tests/test_gp_prior.py`.

### Tests

133 → 102 passing, every case accounted for: −19 (`test_gp_prior.py`, 13 defs of which
three are parametrized over 3 kernels), −4 (`test_logD_prior.py`), −9 (`test_sample.py`,
skipped), +1 new. `test_logD_prior`'s `test_active_terms_matches_what_the_objective_charges`
was **kept and generalised** into `test_prior_none.py`: it now loops over every term
`active_terms()` can report and asserts each actually costs something — the invariant that
the dead-logD-prior incident violated. `test_fisher_crb._post` and three `test_infer.py`
tests were rebased from the GP onto the curvature prior; none had to be dropped, and the
posterior-CRB test still gets `null_dim == 0`.

### Analysis repo

Only `scripts/fit.py`'s `build_prior` needed an edit (7 dead kwargs). `crb_dye_budget.py`
needed none: both of its reflective uses are safe — `build_prior` builds from
`dataclasses.fields(PriorConfig)` so removed names simply stop being picked up, and its
`dataclasses.replace` touches only `bg_*`. The 9 yaml configs keep their now-dead keys (so
they still match the archived copies under `results/`) with a comment marking them inert.

## 0.2.0

Removal-only cleanup. **No numeric path changed**: every result produced with
0.1.0 remains valid and nothing needs re-running. The exact code behind the
results in `../diff_fret_analysis/results/` is tagged **`v0.1.0-results`**.

That claim is checked, not asserted. `../diff_fret_analysis/scripts/regression_gate.py`
captures 76 arrays spanning every live path — potential, generator, photophysics,
likelihood, objective and priors, init, `fit`, `fit_multi`, CRB, reconstruction and
the HMC target — from the published `case_01/sim0` inputs, and diffs two captures
with `np.array_equal`. Across this release: 73 bit-identical, 2 within `1e-12`
(the two `torch.linalg.lstsq` calls, which are not reproducible on identical inputs
even single-threaded — a LAPACK property, not a change here).

### Removed: the neural-network landscape parameterisation

* `MLPPotential`, `_ACTIVATIONS`, and the `_BasePotential` base class (with only
  the spline left, every one of its methods was overridden).
* `PotentialConfig.kind`, `.hidden`, `.activation`. **Callers that passed
  `kind="spline"` simply drop the argument** — the spline is now the only
  parameterisation. This is the one breaking API change.
* The unreachable non-spline arms that removal exposed: the `else` branch of
  `objective.gauge_offset`, the grid-curvature branch of `objective.prior_penalty`
  (and with it `objective.curvature_penalty`, the grid form — only
  `curvature_penalty_spline` was ever reachable from a fit), the `is_spline`
  branch of `sample.build_log_prob`, the Adam fallback in
  `init.warmstart_potential`, and the generic probe fallback in
  `fisher._knot_basis`.

### Removed: orphaned helpers (no importer anywhere)

* `dynamics.py` in full (`em_transition_logp`, the secondary complete-data
  objective).
* `utils.softplus`, `inv_softplus`, `to_log`, `log_dot`, `as_tensor`, `LOG2PI`.
  Only `set_seed` remains. `utils` is now imported explicitly by `__init__`:
  until now `dfl.utils` was bound only as a side effect of `dynamics` importing
  it, so removing that module would have broken `dfl.utils.set_seed` for callers.
* `forward.build_propagator` (the potential-object wrapper; every caller uses
  `build_propagator_from_u`).

### Kept deliberately

`sample.py` (HMC/NUTS) and the GP prior, `infer.fit_multi`, and the test-only
cross-checks `forward.reference_loglik`, `generator.assert_generator_valid` and
`forward.marginal_loglik` — the last three are what make the fast
eigendecomposition path trustworthy.

### Docs

`PriorConfig.curvature_weight` documented the *grid* penalty's
`integral (u'')^2 dx` normalisation, which never described the spline penalty that
actually runs (a bare sum over interior knots, so a weight is only meaningful at
the `n_knots` it was selected at). Corrected.

### Known issue, not introduced here

`infer.fit_multi` calls `objective.prior_penalty` without `rates`, which that
function refuses when a background prior is configured — so `fit_multi` raises
with any `PriorConfig` carrying `bg_g_mean`/`bg_r_mean`. Pre-existing and
untouched, because fixing it would change behaviour; harmless so far only because
nothing calls `fit_multi` with such a prior.

## 0.1.0

Initial packaged release. Produced everything in `../diff_fret_analysis/results/`.
