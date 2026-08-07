"""Backward filter / trajectory reconstruction (manuscript eqs 75-78).

Gates, in order of strength:
  1. gamma == an independent dense ``matrix_exp`` oracle in probability space;
  2. eq (77): ``<beta, rho>`` is constant in t and equals ``marginal_loglik``;
  3. flat mu -> gamma == stationary exactly (catches a transposed adjoint);
  4. inserting output events cannot move gamma at the photon times;
  5. the path sampler's marginals reproduce gamma;
  6. the reconstruction actually follows the colours.
"""

import torch

import diff_fret_likelihood as dfl
from diff_fret_likelihood.forward import _BasePotential_on_grid
from diff_fret_likelihood.generator import smoluchowski, stationary
from diff_fret_likelihood.photophysics import emission_rates


def _setup(G=6, n_knots=4, theta=(1.0, -0.5, 0.7, -1.2), D=8.0, K=8, seed=3,
           rates=None, gap_scale=0.02):
    """Small grid + spline potential + a short random photon stream."""
    torch.manual_seed(seed)
    grid = torch.linspace(4.0, 8.0, G)
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    if rates is None:
        rates = dfl.EffectiveRates.from_physics(300.0, 0.85, 0.85, 25.0, 50.0)
    pot = dfl.build_potential(dfl.PotentialConfig(kind="spline", n_knots=n_knots), grid)
    with torch.no_grad():
        pot.theta.copy_(torch.tensor(theta[:n_knots]))
    times = torch.cumsum(torch.rand(K) * gap_scale, 0)
    colors = torch.randint(0, 2, (K,))
    return dict(grid=grid, pot=pot, C=C, R0=consts.R0, rates=rates,
                D=torch.tensor(D), times=times, colors=colors, T=float(times[-1]))


def _dense_gamma(L, mu_G, mu_R, times, colors, T, p0):
    """Oracle: gamma at every photon time by dense ``matrix_exp`` in probability space.

    Runs the forward density rho and the backward filter beta independently, in the
    UNSYMMETRISED basis and with no ``eigh`` -- the role ``reference_loglik`` plays for the
    likelihood.  beta uses the adjoint ``A.T`` (eq 76) and terminal condition beta(T)=1.
    """
    A = L - torch.diag(mu_G + mu_R)
    K, G = times.shape[0], L.shape[0]
    emit = lambda k: mu_G if int(colors[k]) == 0 else mu_R

    rho, r, t_prev = [], p0.clone(), 0.0
    for k in range(K):
        r = torch.linalg.matrix_exp(A * (float(times[k]) - t_prev)) @ r
        r = r * emit(k)                                       # rho(t_k^+)
        rho.append(r.clone())
        t_prev = float(times[k])

    beta = [None] * K
    b = torch.linalg.matrix_exp(A.T * (float(T) - t_prev)) @ torch.ones(G, dtype=L.dtype)
    beta[K - 1] = b.clone()                                   # beta(t_K^+)
    for k in range(K - 1, 0, -1):
        b = b * emit(k)                                       # beta(t_k^-)
        b = torch.linalg.matrix_exp(A.T * (float(times[k]) - float(times[k - 1]))) @ b
        beta[k - 1] = b.clone()                               # beta(t_{k-1}^+)

    g = torch.stack([beta[k] * rho[k] for k in range(K)])
    return g / g.sum(-1, keepdim=True)


def _dense_pair(L, mu_G, mu_R, times, colors, T, p0, j):
    """Oracle: exact two-slice smoothing ``p(x_j = i, x_{j+1} = m | D)``, returned as [m, i].

    Unlike the marginals, this pins the TRANSITION kernel and the gap it is evaluated at:

        p(x_j, x_{j+1} | D) prop rho_j(i) [e^{(L-Lambda) g_{j+1}}]_{mi} mu_{c_{j+1}}(m) beta_{j+1}(m)

    so an off-by-one in the sampler's gap index shows up here even when the marginals do
    not (a near-identity kernel makes all short gaps look alike).
    """
    A = L - torch.diag(mu_G + mu_R)
    K, G = times.shape[0], L.shape[0]
    emit = lambda k: mu_G if int(colors[k]) == 0 else mu_R

    r, t_prev = p0.clone(), 0.0
    for k in range(j + 1):
        r = torch.linalg.matrix_exp(A * (float(times[k]) - t_prev)) @ r
        r = r * emit(k)                                       # rho_j
        t_prev = float(times[k])

    b = torch.linalg.matrix_exp(
        A.T * (float(T) - float(times[K - 1]))) @ torch.ones(G, dtype=L.dtype)
    for k in range(K - 1, j + 1, -1):                         # down to beta(t_{j+1}^+)
        b = b * emit(k)
        b = torch.linalg.matrix_exp(A.T * (float(times[k]) - float(times[k - 1]))) @ b

    P = torch.linalg.matrix_exp(A * (float(times[j + 1]) - float(times[j])))   # P[m, i]
    joint = (b * emit(j + 1)).unsqueeze(1) * P * r.unsqueeze(0)
    return joint / joint.sum()


# ---------------------------------------------------------------------------
# 1. primary gate: independent dense oracle
# ---------------------------------------------------------------------------
def test_gamma_matches_dense_oracle():
    s = _setup()
    u = _BasePotential_on_grid(s["pot"], s["grid"])
    L = smoluchowski(u, s["D"], float(s["grid"][1] - s["grid"][0]))
    mu_G, mu_R = emission_rates(s["grid"], s["rates"], s["C"], s["R0"])

    oracle = _dense_gamma(L, mu_G, mu_R, s["times"], s["colors"], s["T"], stationary(u))
    res = dfl.reconstruct_trace(s["times"], s["colors"], s["T"], s["pot"], s["D"],
                                s["rates"], s["grid"], s["C"], s["R0"])

    assert res.gamma.shape == oracle.shape
    assert torch.allclose(res.gamma, oracle, atol=1e-10), \
        f"max dev {float((res.gamma - oracle).abs().max()):.3e}"
    # well-formed posterior
    assert (res.gamma >= 0).all()
    assert torch.allclose(res.gamma.sum(-1), torch.ones(s["times"].shape[0]), atol=1e-12)
    # summaries are the moments of gamma
    assert torch.allclose(res.x_mean, res.gamma @ s["grid"], atol=1e-12)
    assert (res.x_sd > 0).all()
    assert res.paths.shape == (0, s["times"].shape[0])


# ---------------------------------------------------------------------------
# 2. eq (77): the likelihood can be evaluated by stopping anywhere
# ---------------------------------------------------------------------------
def test_evidence_constant_and_matches_marginal_loglik():
    s = _setup(G=12, K=20)
    t_out = torch.linspace(0.0, s["T"], 17)
    res = dfl.reconstruct_trace(s["times"], s["colors"], s["T"], s["pot"], s["D"],
                                s["rates"], s["grid"], s["C"], s["R0"], t_out=t_out)
    ll = dfl.marginal_loglik(s["times"], s["colors"], s["T"], s["pot"], s["D"],
                             s["rates"], s["grid"], s["C"], s["R0"])
    assert res.loglik_spread < 1e-8, f"<beta,rho> varies by {res.loglik_spread:.3e}"
    assert abs(res.loglik - float(ll)) < 1e-8, f"{res.loglik} vs {float(ll)}"


# ---------------------------------------------------------------------------
# 3. exact limit: position-independent emission carries no information
# ---------------------------------------------------------------------------
def test_flat_emission_gives_stationary_posterior():
    """a_g = a_r = 0 -> mu is constant -> gamma(t) == pi for every t, for ANY potential.

    Then ``e^{(L-mu)tau} = e^{-mu tau} e^{L tau}``, so rho stays proportional to pi and
    beta stays constant (``L.T 1 = 0``, i.e. the generator's columns sum to zero).  A
    transposed / non-adjoint backward sweep breaks this even though it can still produce a
    constant ``<beta,rho>``, so this is the test that pins the direction.
    """
    flat = dfl.EffectiveRates(torch.tensor(0.0), torch.tensor(0.0),
                              torch.tensor(25.0), torch.tensor(50.0))
    s = _setup(G=16, K=25, rates=flat)
    t_out = torch.linspace(0.0, s["T"], 13)
    res = dfl.reconstruct_trace(s["times"], s["colors"], s["T"], s["pot"], s["D"],
                                s["rates"], s["grid"], s["C"], s["R0"], t_out=t_out)
    pi = stationary(_BasePotential_on_grid(s["pot"], s["grid"]))
    assert torch.allclose(res.gamma, pi.expand_as(res.gamma), atol=1e-12), \
        f"max dev {float((res.gamma - pi).abs().max()):.3e}"


# ---------------------------------------------------------------------------
# 4. output events are exactly the identity
# ---------------------------------------------------------------------------
def test_output_lattice_invariance():
    """gamma at the photon times must not depend on what else was asked for."""
    s = _setup(G=10, K=12)
    base = dfl.reconstruct_trace(s["times"], s["colors"], s["T"], s["pot"], s["D"],
                                 s["rates"], s["grid"], s["C"], s["R0"])
    # a fine lattice that also contains every photon time (ties -> photon first)
    fine = torch.cat([torch.linspace(0.0, s["T"], 40), s["times"]])
    dense = dfl.reconstruct_trace(s["times"], s["colors"], s["T"], s["pot"], s["D"],
                                  s["rates"], s["grid"], s["C"], s["R0"], t_out=fine)
    # locate the photon times within the (sorted) output axis
    pos = torch.searchsorted(dense.t.contiguous(), s["times"].contiguous())
    assert torch.allclose(dense.t[pos], s["times"], atol=1e-12)
    assert torch.allclose(dense.gamma[pos], base.gamma, atol=1e-12), \
        f"max dev {float((dense.gamma[pos] - base.gamma).abs().max()):.3e}"
    assert abs(dense.loglik - base.loglik) < 1e-8


# ---------------------------------------------------------------------------
# 5. the path sampler reproduces gamma
# ---------------------------------------------------------------------------
def test_sample_paths_match_gamma_and_two_slice():
    """FFBS paths reproduce BOTH the one-slice (gamma) and two-slice smoothing laws.

    A transposed conditional kernel, a wrong ``s`` cancellation, or an initial draw from
    the filtering (rather than smoothing) law show up in the marginals.  The gap INDEX does
    not: with uniformly short gaps every kernel is near-identity, so the marginals cannot
    tell ``g_j`` from ``g_{j+1}``.  Hence the alternating short/long gaps below -- long
    enough to relax most of the way to stationarity, short enough to be near-identity --
    and the two-slice comparison, which is a direct function of the gap used.
    """
    torch.manual_seed(11)
    G = 5
    grid = torch.linspace(4.0, 8.0, G)
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300.0, 0.85, 0.85, 25.0, 50.0)
    pot = dfl.build_potential(dfl.PotentialConfig(kind="spline", n_knots=3), grid)
    with torch.no_grad():
        pot.theta.copy_(torch.tensor([0.8, -0.6, 0.5]))
    D = torch.tensor(8.0)
    K = 9
    gaps = torch.tensor([0.0005, 0.4] * K)[:K]          # near-identity / near-stationary
    times = torch.cumsum(gaps, 0)
    colors = torch.randint(0, 2, (K,))
    T = float(times[-1])

    P = 40000
    res = dfl.reconstruct_trace(times, colors, T, pot, D, rates, grid, C, consts.R0,
                                n_paths=P, seed=1)
    assert res.paths.shape == (P, K)
    assert bool((res.paths.unsqueeze(-1) == grid).any(-1).all())   # values are grid points

    hit = (res.paths.unsqueeze(-1) == grid).to(res.gamma.dtype)    # [P, K, G]
    dev = float((hit.mean(0) - res.gamma).abs().max())
    assert dev < 0.02, f"empirical marginals deviate from gamma by {dev:.3f}"

    # --- two-slice: pins the kernel AND the gap index ---
    u = _BasePotential_on_grid(pot, grid)
    L = smoluchowski(u, D, float(grid[1] - grid[0]))
    mu_G, mu_R = emission_rates(grid, rates, C, consts.R0)
    idx = (res.paths.unsqueeze(-1) == grid).to(torch.float64).argmax(-1)   # [P, K]
    for j in (0, 1, K - 2):
        want = _dense_pair(L, mu_G, mu_R, times, colors, T, stationary(u), j)
        emp = torch.zeros(G, G, dtype=torch.float64)
        emp.index_put_((idx[:, j + 1], idx[:, j]),
                       torch.full((P,), 1.0 / P, dtype=torch.float64), accumulate=True)
        d = float((emp - want).abs().max())
        assert d < 0.02, f"two-slice (j={j}) deviates by {d:.3f}"

    # reproducible, and sampling must not perturb the posterior itself
    again = dfl.reconstruct_trace(times, colors, T, pot, D, rates, grid, C, consts.R0,
                                  n_paths=P, seed=1)
    assert torch.equal(res.paths, again.paths)
    plain = dfl.reconstruct_trace(times, colors, T, pot, D, rates, grid, C, consts.R0)
    assert torch.allclose(res.gamma, plain.gamma, atol=1e-12)
    assert abs(res.loglik - plain.loglik) < 1e-9


# ---------------------------------------------------------------------------
# 6. it follows the data
# ---------------------------------------------------------------------------
def test_reconstruction_follows_the_colours():
    """Green photons -> low E -> large x;  red photons -> high E -> small x.

    E(x) = R0^6/(R0^6+x^6) decreases in x, so a run of donor (green) photons must place the
    molecule at large x and a run of acceptor (red) photons at small x.  Slow diffusion
    (D = 1) keeps the two stretches resolvable.
    """
    G = 24
    grid = torch.linspace(4.0, 8.0, G)
    consts = dfl.PhysicsConstants()
    C = consts.crosstalk_tensor()
    rates = dfl.EffectiveRates.from_physics(300.0, 0.85, 0.85, 25.0, 50.0)
    pot = dfl.build_potential(dfl.PotentialConfig(kind="spline", n_knots=4), grid)
    with torch.no_grad():
        pot.theta.zero_()                       # flat -> the data, not the prior, decides
    n = 20
    times = torch.arange(1, 2 * n + 1, dtype=torch.float64) * 0.01
    colors = torch.cat([torch.zeros(n, dtype=torch.long),
                        torch.ones(n, dtype=torch.long)])
    res = dfl.reconstruct_trace(times, colors, float(times[-1]), pot, torch.tensor(1.0),
                                rates, grid, C, consts.R0)

    E = dfl.fret_efficiency(grid, consts.R0)
    E_mean = res.gamma @ E
    green, red = slice(0, n), slice(n, 2 * n)
    assert res.x_mean[green].mean() > res.x_mean[red].mean() + 0.5, \
        f"green {float(res.x_mean[green].mean()):.2f} vs red {float(res.x_mean[red].mean()):.2f}"
    assert E_mean[green].mean() < E_mean[red].mean()
    # and the posterior is tighter than the prior it started from
    pi = stationary(_BasePotential_on_grid(pot, grid))
    sd_prior = float((pi @ grid.square() - (pi @ grid) ** 2).clamp_min(0).sqrt())
    assert float(res.x_sd.median()) < sd_prior


# ---------------------------------------------------------------------------
# 7. the batch wrapper
# ---------------------------------------------------------------------------
def test_reconstruct_batch_matches_per_trace():
    s = _setup(G=8, K=10)
    K = s["times"].shape[0]
    ipt = torch.zeros(2, K)
    ipt[0, 1:] = s["times"][1:] - s["times"][:-1]
    ipt[0, 0] = s["times"][0]
    ipt[1, :K - 3] = ipt[0, :K - 3]
    mask = torch.zeros(2, K, dtype=torch.bool)
    mask[0, :] = True
    mask[1, :K - 3] = True
    batch = dfl.simulate.Batch(
        ipt=ipt, colors=s["colors"].expand(2, K).contiguous(), mask=mask,
        lengths=torch.tensor([K, K - 3]), T=ipt.sum(1),
    )
    out = dfl.reconstruct_batch(batch, s["pot"], s["D"], s["rates"], s["grid"],
                                s["C"], s["R0"])
    assert len(out) == 2
    assert out[0].gamma.shape == (K, s["grid"].shape[0])
    assert out[1].gamma.shape == (K - 3, s["grid"].shape[0])
    single = dfl.reconstruct_trace(s["times"], s["colors"], float(s["times"][-1]),
                                   s["pot"], s["D"], s["rates"], s["grid"],
                                   s["C"], s["R0"])
    assert torch.allclose(out[0].gamma, single.gamma, atol=1e-12)
    assert len(dfl.reconstruct_batch(batch, s["pot"], s["D"], s["rates"], s["grid"],
                                     s["C"], s["R0"], indices=[1])) == 1


# ---------------------------------------------------------------------------
# 8. the guards actually fire
# ---------------------------------------------------------------------------
def test_input_guards():
    """Mis-tiled input must raise, not silently bias (cf. SPEC Remark 1)."""
    import pytest
    s = _setup(G=6, K=5)
    args = (s["pot"], s["D"], s["rates"], s["grid"], s["C"], s["R0"])
    dfl.reconstruct_trace(s["times"], s["colors"], s["T"], *args)          # valid

    with pytest.raises(AssertionError):                                    # T < t_K
        dfl.reconstruct_trace(s["times"], s["colors"], s["T"] * 0.5, *args)
    with pytest.raises(AssertionError):                                    # unsorted
        dfl.reconstruct_trace(s["times"].flip(0), s["colors"], s["T"], *args)
    with pytest.raises(AssertionError):                                    # t_out > T
        dfl.reconstruct_trace(s["times"], s["colors"], s["T"], *args,
                              t_out=torch.tensor([0.0, s["T"] * 2]))
    with pytest.raises(ValueError):                                        # empty t_out
        dfl.reconstruct_trace(s["times"], s["colors"], s["T"], *args,
                              t_out=torch.zeros(0))
    with pytest.raises(ValueError):                                        # no photons
        dfl.reconstruct_trace(torch.zeros(0), torch.zeros(0, dtype=torch.long),
                              0.0, *args)
