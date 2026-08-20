# Changelog

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
