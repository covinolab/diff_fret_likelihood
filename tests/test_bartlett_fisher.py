"""Bartlett + Fisher score-at-truth test for the equilibrium likelihood.

A powered-null correctness test: simulate long *equilibrium* photon traces at a
known truth (double well, bound 4 nm / unbound 9 nm, 4 k_BT barrier, D=20) and
verify the marginal likelihood is the correct normalized density for that
data-generating process, on the parameterisation inference uses.

Two identities are checked at the true parameter ``phi*``:

  * **Bartlett-1** (score at truth is zero):  E_true[ d logL / d phi ] = 0.
    Reported as per-parameter  z = mean / SE  over N independent traces, plus a
    global Hotelling statistic.  A correct, unbiased likelihood has |z| ~ O(1).
  * **Bartlett-2 / Fisher** (information-matrix equality):
    Cov[ d logL / d phi ]  =  E[ -d^2 logL / d phi^2 ] .
    Reported as the full-matrix relative Frobenius distance and per-parameter
    diagonal ratios (~1).  This is the part that detects misspecification.

Parameterisation ``phi = [g_0..g_10 (force-spline knots), lnD, log a_g, log a_r,
log bg_g, log bg_r]`` (P=16): the landscape, the diffusion coefficient, and the
four photophysics rates are all scored together.

Because the simulator integrates a *cubic* spline through the force knots, the
map g -> U(x) is exactly affine (``U = B @ g + b0``); we reconstruct that basis
from the simulator's own FE integrator, so the likelihood landscape equals the
data-generating landscape to machine precision (zero representation error) and
Bartlett-1 is evaluated at the genuine truth, not a fitted point.

The (slow) simulation is cached to disk keyed on the truth config, so editing
the likelihood re-runs only the fast scoring/Fisher path -- this file is a
regression guard on the equilibrium likelihood.

Findings (validated up to ~5e6 photons):
  * Landscape and diffusion are UNBIASED: grid-free score_U(x) max|z| ~ 1-2 over
    the full grid, z(lnD) ~ 0, at all sizes.
  * The Fisher information-matrix equality HOLDS to a few percent (diagonal
    ratios ~1.0), photophysics included -- the likelihood is correctly specified.
  * The FOUR absolute photon-rate scores carry a small O(N_traces / sqrt(M))
    boundary term (M = total photons) from the photon-stream *windowing
    convention*: the batch path conditions on the first photon and drops the
    leading [0, t_1] and trailing [t_K, T] survival intervals (~1 unmodeled
    photon/trace).  It is grid-independent and p0-independent, and it is
    suppressed by LONGER traces (confirmed: at 4.8e6 photons the rate Hotelling
    p is 0.005 with 5000x150ms traces vs 0.16 with 1250x600ms traces).  It is a
    convention property, not a bulk defect; use longer traces for high-photon
    validation, or model the full window (survival gaps) if the absolute rate
    normalisation must be unbiased at any trace length.

Run modes:
    python -m diff_fret_likelihood.tests.test_bartlett_fisher --pilot   # calibration only
    python -m pytest diff_fret_likelihood/tests/test_bartlett_fisher.py -s
    python -m diff_fret_likelihood.tests.test_bartlett_fisher --full    # >=1000 traces + figure
"""

from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np
import torch

# Make the package + benchmark harness importable regardless of cwd.
_LA = "/home/dingeldein/Desktop/fret_sbi/likelihood_analysis"
if _LA not in sys.path:
    sys.path.insert(0, _LA)

import diff_fret_likelihood as dfl  # noqa: E402
from diff_fret_likelihood.config import DTYPE, GridConfig  # noqa: E402
from diff_fret_likelihood.forward import (  # noqa: E402
    build_propagator_from_u, marginal_loglik_batch,
)
from diff_fret_likelihood.generator import stationary  # noqa: E402
from diff_fret_likelihood.photophysics import EffectiveRates  # noqa: E402
from diff_fret_likelihood.simulate import (  # noqa: E402
    assemble_theta, ground_truth_landscape, simulate_traces, _ensure_path,
)
from benchmark.configs import (  # noqa: E402
    KNOT_X, K, X_MIN, X_MAX, PHOTOPHYSICS, consts_and_rates, make_cfg,
)

torch.set_default_dtype(DTYPE)

# Mixed-precision toggle for the score/likelihood recursion (fp32 matmuls, fp64
# log-normaliser -- mirrors forward._recur_step).  None -> full float64.  Set via
# BARTLETT_FP32=1 or by assigning the module global before calling run().
_BF_PDT = torch.float32 if os.environ.get("BARTLETT_FP32") == "1" else None

# --------------------------------------------------------------------------- #
# Truth configuration (validated in calibration)
# --------------------------------------------------------------------------- #
X_B, W = 6.5, 2.5                       # symmetric quartic: minima at 4 & 9 nm
A_QUARTIC = 4.0 / W ** 4                 # barrier = 4 kT above both wells
LOG10_D = float(np.log10(20.0))          # D = 20 nm^2/ms
TOTAL_TIME = float(os.environ.get("BARTLETT_TT", "150.0"))  # ms (~16 crossings)
N_MAX_PHOTONS = 30000                    # never truncate a 150 ms trace
X_INIT = [3.7, 9.3]
SPLINE_TYPE = "cubic"                    # -> g->U affine (exact basis)
PHOTOPHYSICS_NAME = "realistic"          # kD=6, eta=.85, backgrounds, crosstalk
N_GRID = 160

# scoring / sizes
N_TEST = int(os.environ.get("BARTLETT_N", "300"))
N_FULL = 1200
SEED_OFFSET = int(os.environ.get("BARTLETT_SEED", "0"))   # independent replicas
FISHER_SUBSET = 160                      # traces used for the E[-H] Hessian
SCORE_CHUNK = int(os.environ.get("BARTLETT_CHUNK", "32"))  # vmap chunk for scores
CACHE_DIR = os.path.join(_LA, "benchmark", "_bartlett")
FIG_DIR = os.path.join(_LA, "benchmark", "figures")

# parameter layout
IDX_LND = K
IDX_RATES = slice(K + 1, K + 5)
P_DIM = K + 5
RATE_NAMES = ["log_a_g", "log_a_r", "log_bg_g", "log_bg_r"]

# pass thresholds (generous margins; a real defect grows with sqrt(N))
THRESH = dict(maxz_gridfree=5.0, z_knot=4.5, z_lnD=4.0, z_rate=4.0,
              fisher_relfrob=0.30, diag_ratio=(0.6, 1.6))


# --------------------------------------------------------------------------- #
# Truth landscape (analytic quartic -> simulator force knots)
# --------------------------------------------------------------------------- #
def g_true_knots() -> np.ndarray:
    """Analytic force knots g_i = -U'(x_i) of the quartic double well."""
    x = KNOT_X
    return -4.0 * A_QUARTIC * (x - X_B) * ((x - X_B) ** 2 - W ** 2)


def build_cfg(total_time=TOTAL_TIME):
    return make_cfg("equilibrium", spline_type=SPLINE_TYPE, x_init=list(X_INIT),
                    total_time=total_time, N_max_photons=N_MAX_PHOTONS)


def build_theta(g_knots, log10_D=LOG10_D):
    pp = PHOTOPHYSICS[PHOTOPHYSICS_NAME]
    return assemble_theta(g_knots, log10_D, pp.R0, pp.kD, pp.k_gb, pp.k_rb,
                          pp.eta_g, pp.eta_r, pp.C_gr, pp.C_rg)


# --------------------------------------------------------------------------- #
# Exact affine force-knot basis:  U(x) = B @ g + b0
# --------------------------------------------------------------------------- #
def build_force_basis(cfg, theta, grid_np):
    """Return (B [G,K], b0 [G]) so that U(g) = B @ g + b0 reproduces the
    simulator's realised free energy exactly (cubic spline => affine in g)."""
    _ensure_path()
    from utils import eval_free_energy_from_config
    fe = eval_free_energy_from_config(cfg)

    def raw_U(g):
        th = np.asarray(theta, np.float64).copy()
        th[1:K + 1] = g
        _, U = fe(th, np.asarray(grid_np, np.float64))
        return U

    b0 = raw_U(np.zeros(K))
    B = np.empty((grid_np.size, K), dtype=np.float64)
    for k in range(K):
        e = np.zeros(K); e[k] = 1.0
        B[:, k] = raw_U(e) - b0
    return B, b0


class ForceKnotPotential:
    """Potential parameterised by force-spline knots: on_grid = B @ g + b0.

    ``g`` is a view into the leaf ``phi`` so autograd delivers d logL / d g.
    """

    def __init__(self, g, B, b0):
        self.g, self.B, self.b0 = g, B, b0

    def on_grid(self, grid):
        return self.B @ self.g + self.b0


# --------------------------------------------------------------------------- #
# Truth parameter vector phi*  and fixed calibration (C, R0, dx)
# --------------------------------------------------------------------------- #
def truth_phi(device):
    pp = PHOTOPHYSICS[PHOTOPHYSICS_NAME]
    rates = EffectiveRates.from_physics(pp.kD, pp.eta_g, pp.eta_r, pp.k_gb,
                                        pp.k_rb, device=device)
    g = torch.as_tensor(g_true_knots(), dtype=DTYPE, device=device)
    phi = torch.empty(P_DIM, dtype=DTYPE, device=device)
    phi[:K] = g
    phi[IDX_LND] = float(np.log(10.0 ** LOG10_D))
    phi[K + 1] = torch.log(rates.a_g)
    phi[K + 2] = torch.log(rates.a_r)
    phi[K + 3] = torch.log(rates.bg_g)
    phi[K + 4] = torch.log(rates.bg_r)
    return phi


def unpack_phi(phi):
    D = torch.exp(phi[IDX_LND])
    r = phi[IDX_RATES]
    rates = EffectiveRates(torch.exp(r[0]), torch.exp(r[1]),
                           torch.exp(r[2]), torch.exp(r[3]))
    return phi[:K], D, rates


# --------------------------------------------------------------------------- #
# Functional log-likelihoods (differentiable in phi)
# --------------------------------------------------------------------------- #
def _mp_recursion(prop, ipt, colors, mask, p0v, pdt):
    """Scaled forward recursion for one trace; optional fp32 propagation with an
    fp64 running log-normaliser (mirrors forward._recur_step)."""
    s = prop.s
    v = p0v / s
    c0 = v.abs().sum()
    v = v / c0
    log_norm = torch.log(c0)                          # stays float64
    lam, Q, muG, muR = prop.lam, prop.Q, prop.mu_G, prop.mu_R
    if pdt is not None:
        lam, Q, muG, muR = lam.to(pdt), Q.to(pdt), muG.to(pdt), muR.to(pdt)
        v = v.to(pdt)
    ones = torch.ones_like(muG)
    zero = torch.zeros((), dtype=v.dtype, device=v.device)
    for k in range(ipt.shape[0]):
        tau = torch.where(mask[k], ipt[k].to(v.dtype), zero)
        v = Q @ (torch.exp(lam * tau) * (Q.T @ v))    # e^{A tau} v
        emit = torch.where(colors[k] == 0, muG, muR)
        emit = torch.where(mask[k], emit, ones)
        v = v * emit
        c = v.abs().sum()
        c = torch.where(c > 0, c, torch.ones_like(c))
        v = v / c
        log_norm = log_norm + torch.log(c).to(log_norm.dtype)
    total = torch.dot(s, v.to(s.dtype))
    return torch.log(total.clamp_min(1e-300)) + log_norm


def single_logL(phi, ipt, colors, mask, grid, C, R0, B, b0, dx, jitter=1e-12,
                p0=None):
    """Marginal log-lik of ONE (padded) trace -- vmap/jacrev friendly."""
    g, D, rates = unpack_phi(phi)
    u = B @ g + b0
    u = u - u.min()
    prop = build_propagator_from_u(u, D, rates, grid, C, R0, dx, jitter)
    p0v = stationary(u) if p0 is None else p0
    return _mp_recursion(prop, ipt, colors, mask, p0v, _BF_PDT)


def batch_loglik_from_phi(phi, ipt, colors, mask, grid, C, R0, B, b0,
                          reduce="sum", p0=None):
    """Batched marginal log-lik through the tested package path."""
    g, D, rates = unpack_phi(phi)
    pot = ForceKnotPotential(g, B, b0)
    return marginal_loglik_batch(ipt, colors, mask, pot, D, rates, grid, C, R0,
                                 p0=p0, reduce=reduce, propagate_dtype=_BF_PDT)


def single_logL_u(u, ipt, colors, mask, D, rates, grid, C, R0, dx, jitter=1e-12,
                  p0=None):
    """Marginal log-lik of ONE (padded) trace given U(x) directly on the grid.

    For the representation-free grid-free functional score d logL / d U(x)."""
    prop = build_propagator_from_u(u, D, rates, grid, C, R0, dx, jitter)
    p0v = stationary(u) if p0 is None else p0
    return _mp_recursion(prop, ipt, colors, mask, p0v, _BF_PDT)


# --------------------------------------------------------------------------- #
# Cached parallel simulation (CPU, fork pool -- MUST run before any CUDA work)
# --------------------------------------------------------------------------- #
def _cache_key(g_knots, log10_D, n_traces):
    h = hashlib.sha1()
    for a in (np.round(g_knots, 8), np.array([log10_D, TOTAL_TIME, N_MAX_PHOTONS,
              N_GRID]), np.array(X_INIT)):
        h.update(np.ascontiguousarray(a, np.float64).tobytes())
    h.update(f"{SPLINE_TYPE}|{PHOTOPHYSICS_NAME}|{n_traces}|s{SEED_OFFSET}".encode())
    return h.hexdigest()[:16]


def _sim_worker(args):
    cfg, theta, n_sub, seed = args
    np.random.seed(seed)
    batch, _ = simulate_traces(cfg, theta, "equilibrium", n_traces=n_sub,
                               min_photons=50, max_photons=None, device="cpu",
                               verbose=False)
    out = []
    for i in range(batch.n_traces):
        n = int(batch.lengths[i])
        out.append((batch.ipt[i, :n].numpy().astype(np.float64),
                    batch.colors[i, :n].numpy().astype(np.int64)))
    return out


def simulate_cached(n_traces, n_workers=24, verbose=True):
    """Return (ipt, colors, mask, lengths) CPU tensors; simulate once + cache."""
    g_knots = g_true_knots()
    key = _cache_key(g_knots, LOG10_D, n_traces)
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"eq_{key}.pt")
    if os.path.exists(path):
        blob = torch.load(path, weights_only=False)
        if verbose:
            print(f"[cache] hit {path}  ({int(blob['lengths'].shape[0])} traces)")
        return blob["ipt"], blob["colors"], blob["mask"], blob["lengths"]

    if verbose:
        print(f"[cache] miss -> simulating {n_traces} equilibrium traces "
              f"(this is the slow, one-time step)...")
    import multiprocessing as mp
    cfg = build_cfg()
    theta = build_theta(g_knots)
    n_workers = max(1, min(n_workers, mp.cpu_count()))
    per = int(np.ceil(n_traces / n_workers))
    jobs = [(cfg, theta, per, 1234 + 7919 * w + 100003 * SEED_OFFSET)
            for w in range(n_workers)]
    t0 = time.perf_counter()
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        chunks = pool.map(_sim_worker, jobs)
    raw = [t for ch in chunks for t in ch][:n_traces]
    dt = time.perf_counter() - t0
    if verbose:
        print(f"[cache] simulated {len(raw)} traces in {dt:.1f}s "
              f"on {n_workers} workers")

    Kmax = max(len(i) for i, _ in raw)
    B = len(raw)
    ipt = torch.zeros(B, Kmax, dtype=DTYPE)
    colors = torch.zeros(B, Kmax, dtype=torch.int64)
    mask = torch.zeros(B, Kmax, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.int64)
    for b, (i, c) in enumerate(raw):
        n = len(i)
        ipt[b, :n] = torch.as_tensor(i, dtype=DTYPE)
        colors[b, :n] = torch.as_tensor(c)
        mask[b, :n] = True
        lengths[b] = n
    torch.save({"ipt": ipt, "colors": colors, "mask": mask, "lengths": lengths},
               path + ".tmp")
    os.replace(path + ".tmp", path)
    return ipt, colors, mask, lengths


# --------------------------------------------------------------------------- #
# Per-trace scores  (vmap/jacrev primary, autograd loop fallback)
# --------------------------------------------------------------------------- #
def per_trace_scores(phi, ipt, colors, mask, grid, C, R0, B, b0, dx,
                     chunk=SCORE_CHUNK, p0=None):
    """[N, P] per-trace scores d logL_i / d phi at ``phi``."""
    N = ipt.shape[0]
    try:
        from torch.func import vmap, jacrev

        def f(phi_, i, c, m):
            return single_logL(phi_, i, c, m, grid, C, R0, B, b0, dx, p0=p0)

        jac = vmap(jacrev(f, argnums=0), in_dims=(None, 0, 0, 0))
        outs = []
        for s0 in range(0, N, chunk):
            sl = slice(s0, min(s0 + chunk, N))
            outs.append(jac(phi, ipt[sl], colors[sl], mask[sl]).detach())
        return torch.cat(outs, 0)
    except Exception as exc:  # pragma: no cover - robustness fallback
        print(f"[scores] vmap/jacrev failed ({type(exc).__name__}: {exc}); "
              f"using autograd loop")
        rows = []
        for i in range(N):
            p = phi.clone().requires_grad_(True)
            ll = single_logL(p, ipt[i], colors[i], mask[i], grid, C, R0, B, b0,
                             dx, p0=p0)
            (gi,) = torch.autograd.grad(ll, p)
            rows.append(gi.detach())
        return torch.stack(rows, 0)


def mean_hessian(phi, ipt, colors, mask, grid, C, R0, B, b0, p0=None,
                 hb=int(os.environ.get("BARTLETT_HB", "24"))):
    """E[H] = (1/n) d^2 (sum_i logL_i) / d phi^2 via autograd double-backward.

    Hessian is linear in the summed log-lik, so we accumulate it over
    sub-batches of ``hb`` traces -- memory stays bounded regardless of n_grid
    or the Fisher subset size."""
    n = ipt.shape[0]
    Hsum = torch.zeros(P_DIM, P_DIM, dtype=DTYPE, device=phi.device)
    for s0 in range(0, n, hb):
        sl = slice(s0, min(s0 + hb, n))
        p = phi.clone().requires_grad_(True)
        ll = batch_loglik_from_phi(p, ipt[sl], colors[sl], mask[sl], grid, C, R0,
                                   B, b0, reduce="sum", p0=p0)
        (g1,) = torch.autograd.grad(ll, p, create_graph=True)
        for j in range(P_DIM):
            (row,) = torch.autograd.grad(g1[j], p, retain_graph=(j < P_DIM - 1))
            Hsum[j] += row.detach()
        del ll, g1
    return Hsum / n


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def bartlett1_stats(scores):
    """Per-parameter z = mean/SE and a rank-truncated Hotelling statistic."""
    S = scores.cpu().numpy()
    N, P = S.shape
    mean = S.mean(0)
    se = S.std(0, ddof=1) / np.sqrt(N)
    z = np.where(se > 0, mean / se, 0.0)

    cov = np.cov(S, rowvar=False)                       # [P,P]
    w, V = np.linalg.eigh(cov)
    keep = w > 1e-10 * w.max()
    dof = int(keep.sum())
    cinv = (V[:, keep] / w[keep]) @ V[:, keep].T        # rank-truncated pinv
    T2 = float(N * mean @ cinv @ mean)
    # Hotelling -> F
    from scipy import stats
    Fstat = T2 * (N - dof) / (dof * (N - 1)) if N > dof else np.nan
    pval = float(stats.f.sf(Fstat, dof, N - dof)) if np.isfinite(Fstat) else np.nan
    return dict(mean=mean, se=se, z=z, T2=T2, dof=dof, F=Fstat, p=pval)


def fisher_stats(scores, EH):
    """Compare Cov[score] to -E[H] (information-matrix equality)."""
    S = scores.cpu().numpy()
    cov = np.cov(S, rowvar=False)
    negEH = -EH.cpu().numpy()
    rel_frob = np.linalg.norm(cov - negEH) / (np.linalg.norm(negEH) + 1e-30)
    diag_ratio = np.diag(cov) / (np.diag(negEH) + 1e-30)
    return dict(cov=cov, negEH=negEH, rel_frob=float(rel_frob),
                diag_ratio=diag_ratio)


# --------------------------------------------------------------------------- #
# Grid-free functional score (diagnostic; representation-error-free arbiter)
# --------------------------------------------------------------------------- #
def gridfree_score(theta, cfg, ipt, colors, mask, grid, C, R0, rates, D,
                   dx, chunk=SCORE_CHUNK, p0=None):
    """max|z| of per-trace d logL / d U(x) over the full grid at the exact true
    U(x) -- representation-free arbiter of likelihood correctness."""
    gnp = grid.detach().cpu().numpy()
    U_np = ground_truth_landscape(cfg, theta, gnp)
    U_true = torch.as_tensor(U_np - U_np.min(), dtype=DTYPE, device=grid.device)
    N = ipt.shape[0]

    def f(u, i, c, m):
        return single_logL_u(u, i, c, m, D, rates, grid, C, R0, dx, p0=p0)

    try:
        from torch.func import vmap, jacrev
        jac = vmap(jacrev(f, argnums=0), in_dims=(None, 0, 0, 0))
        outs = []
        for s0 in range(0, N, chunk):
            sl = slice(s0, min(s0 + chunk, N))
            outs.append(jac(U_true, ipt[sl], colors[sl], mask[sl]).detach())
        per = torch.cat(outs, 0)
    except Exception as exc:  # pragma: no cover
        print(f"[gridfree] vmap failed ({type(exc).__name__}: {exc}); loop")
        rows = []
        for i in range(N):
            u = U_true.clone().requires_grad_(True)
            ll = f(u, ipt[i], colors[i], mask[i])
            (gu,) = torch.autograd.grad(ll, u)
            rows.append(gu.detach())
        per = torch.stack(rows, 0)

    S = per.cpu().numpy()                                # [N, G] per-trace
    mean = S.mean(0)
    se = S.std(0, ddof=1) / np.sqrt(N)
    z = np.where(se > 1e-300, mean / se, 0.0)
    occ = stationary(U_true).cpu().numpy()               # occupancy floor
    live = occ > 1e-4 * occ.max()
    z_live = np.where(live, z, 0.0)
    jmax = int(np.argmax(np.abs(z_live)))
    return dict(grid=gnp, U_true=U_true.cpu().numpy(), z=z, live=live,
                maxz=float(abs(z_live[jmax])), x_maxz=float(gnp[jmax]),
                gauge=float(abs(mean.sum())))


# --------------------------------------------------------------------------- #
# Self-checks (gate everything)
# --------------------------------------------------------------------------- #
def selfcheck_linearity(cfg, theta, grid_np, B, b0):
    _ensure_path()
    from utils import eval_free_energy_from_config
    fe = eval_free_energy_from_config(cfg)
    _, U_direct = fe(theta, grid_np)
    U_affine = B @ g_true_knots() + b0
    return float(np.abs(U_affine - U_direct).max())


def selfcheck_fd_score(phi, ipt, colors, mask, grid, C, R0, B, b0, dx,
                       probes, eps=1e-4, p0=None):
    """Central-difference d logL / d phi vs analytic, on a few components."""
    sub = slice(0, min(6, ipt.shape[0]))

    def LL(p):
        return float(batch_loglik_from_phi(p, ipt[sub], colors[sub], mask[sub],
                                           grid, C, R0, B, b0, reduce="sum",
                                           p0=p0))
    pa = phi.clone().requires_grad_(True)
    ll = batch_loglik_from_phi(pa, ipt[sub], colors[sub], mask[sub], grid, C, R0,
                               B, b0, reduce="sum", p0=p0)
    (ga,) = torch.autograd.grad(ll, pa)
    rows = []
    for j in probes:
        pp = phi.clone(); pp[j] += eps
        pm = phi.clone(); pm[j] -= eps
        fd = (LL(pp) - LL(pm)) / (2 * eps)
        an = float(ga[j])
        rows.append((j, an, fd, abs(an - fd) / (abs(fd) + 1e-12)))
    return rows


def selfcheck_equivalence(phi, ipt, colors, mask, grid, C, R0, B, b0, dx,
                          n=4, p0=None):
    """vmap-jacrev per-trace scores vs autograd loop on a small chunk."""
    v = per_trace_scores(phi, ipt[:n], colors[:n], mask[:n], grid, C, R0, B, b0,
                         dx, chunk=n, p0=p0)
    rows = []
    for i in range(n):
        p = phi.clone().requires_grad_(True)
        ll = single_logL(p, ipt[i], colors[i], mask[i], grid, C, R0, B, b0, dx,
                         p0=p0)
        (gi,) = torch.autograd.grad(ll, p)
        rows.append(gi.detach())
    loop = torch.stack(rows, 0)
    return float((v - loop).abs().max())


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(n_traces=N_TEST, device=None, make_fig=False, verbose=True, n_grid=N_GRID,
        do_fisher=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1. simulate on CPU (before CUDA), then move to device
    ipt, colors, mask, lengths = simulate_cached(n_traces, verbose=verbose)
    n_traces = ipt.shape[0]
    n_photons = int(lengths.sum())
    ipt = ipt.to(device); colors = colors.to(device); mask = mask.to(device)

    # 2. truth + exact basis + fixed calibration
    cfg = build_cfg()
    theta = build_theta(g_true_knots())
    grid = GridConfig(X_MIN, X_MAX, n_grid).build(device)
    dx = float(grid[1] - grid[0])
    gnp = grid.detach().cpu().numpy()
    B_np, b0_np = build_force_basis(cfg, theta, gnp)
    Bt = torch.as_tensor(B_np, dtype=DTYPE, device=device)
    b0t = torch.as_tensor(b0_np, dtype=DTYPE, device=device)
    pp = PHOTOPHYSICS[PHOTOPHYSICS_NAME]
    _, C, rates_true = consts_and_rates(pp, device=device)
    R0 = pp.R0
    phi = truth_phi(device)
    _, D_true, _ = unpack_phi(phi)

    if verbose:
        print(f"\n{'='*72}\n  BARTLETT + FISHER  |  eq double well  |  "
              f"device={device}\n{'='*72}")
        print(f"  traces={n_traces}  photons={n_photons}  "
              f"({n_photons/n_traces:.0f}/trace)  P={P_DIM}  n_grid={n_grid}")

    # 3. self-checks (gate)
    lin = selfcheck_linearity(cfg, theta, gnp, B_np, b0_np)
    probes = [K // 2, IDX_LND, K + 1, K + 3]           # a knot, lnD, a_g, bg_g
    fd = selfcheck_fd_score(phi, ipt, colors, mask, grid, C, R0, Bt, b0t, dx,
                            probes)
    equiv = selfcheck_equivalence(phi, ipt, colors, mask, grid, C, R0, Bt, b0t, dx)
    if verbose:
        print(f"\n  [self-check] basis linearity  max|B g + b0 - fe(g)| = {lin:.2e}")
        for j, an, fdv, re in fd:
            print(f"  [self-check] FD score phi[{j:2d}]  an={an:+.4e} "
                  f"fd={fdv:+.4e}  rel={re:.2e}")
        print(f"  [self-check] vmap==loop score  max abs diff = {equiv:.2e}")

    # 4. Bartlett-1: per-trace scores at truth
    scores = per_trace_scores(phi, ipt, colors, mask, grid, C, R0, Bt, b0t, dx)
    b1 = bartlett1_stats(scores)

    # 5. grid-free diagnostic (representation-free)
    gf = gridfree_score(theta, cfg, ipt, colors, mask, grid, C, R0, rates_true,
                        D_true, dx)

    res = dict(n_traces=n_traces, n_photons=n_photons, lin=lin, fd=fd,
               equiv=equiv, b1=b1, gf=gf, grid=gnp, n_grid=n_grid,
               scores=scores.cpu().numpy())

    if do_fisher:
        # 6. Bartlett-2 / Fisher
        nsub = min(FISHER_SUBSET, n_traces)
        EH = mean_hessian(phi, ipt[:nsub], colors[:nsub], mask[:nsub], grid, C,
                          R0, Bt, b0t)
        res["fi"] = fisher_stats(scores, EH)

        # 7. p0-sensitivity diagnostic (stationary vs uniform): a real likelihood
        #    defect is p0-invariant; a first-photon/window artifact moves with p0.
        nd = min(n_traces, 300)
        p0_unif = torch.ones(n_grid, dtype=DTYPE, device=device) / n_grid
        su = per_trace_scores(phi, ipt[:nd], colors[:nd], mask[:nd], grid, C, R0,
                              Bt, b0t, dx, p0=p0_unif).cpu().numpy()
        zu = np.where(su.std(0, ddof=1) > 0,
                      su.mean(0) / (su.std(0, ddof=1) / np.sqrt(nd)), 0.0)
        ss = scores.cpu().numpy()[:nd]
        zs = np.where(ss.std(0, ddof=1) > 0,
                      ss.mean(0) / (ss.std(0, ddof=1) / np.sqrt(nd)), 0.0)
        res["p0diag"] = dict(nd=nd, z_stat=zs, z_unif=zu)
    if verbose:
        report(res)
    if make_fig:
        _figure(res)
    return res


def report(res):
    b1, gf = res["b1"], res["gf"]
    z = b1["z"]
    print(f"\n  --- Bartlett-1 (score at truth = 0) ---")
    print(f"  grid-free  max|z|(full grid) = {gf['maxz']:.2f} at x={gf['x_maxz']:.2f} nm"
          f"   [gauge Sum_x score={gf['gauge']:.1e}]")
    jz = int(np.argmax(np.abs(z[:K])))
    print(f"  force-knots max|z| = {np.abs(z[:K]).max():.2f} (knot {jz}, x={KNOT_X[jz]:.2f})")
    print(f"  z(lnD) = {z[IDX_LND]:+.2f}")
    for i, nm in enumerate(RATE_NAMES):
        print(f"  z({nm:8s}) = {z[K+1+i]:+.2f}")
    print(f"  Hotelling: T2={b1['T2']:.1f}  dof={b1['dof']}  F p-value={b1['p']:.3f}")
    if "fi" in res:
        fi = res["fi"]
        print(f"\n  --- Bartlett-2 / Fisher (Var[s] = -E[H]) ---")
        print(f"  full-matrix relative Frobenius ||Cov(s)-(-E[H])||/||E[H]|| = {fi['rel_frob']:.3f}")
        dr = fi["diag_ratio"]
        print(f"  diag ratio  lnD = {dr[IDX_LND]:.2f}   " +
              "  ".join(f"{nm.split('_',1)[1]}={dr[K+1+i]:.2f}"
                        for i, nm in enumerate(RATE_NAMES)))
        kd = dr[:K]
        print(f"  diag ratio  force-knots: min={kd.min():.2f} max={kd.max():.2f} "
              f"median={np.median(kd):.2f}")
    if "p0diag" in res:
        pd = res["p0diag"]
        zs, zu = pd["z_stat"], pd["z_unif"]
        print(f"\n  --- p0 sensitivity (n={pd['nd']}; defect is p0-invariant) ---")
        print(f"  z(lnD)   stationary={zs[IDX_LND]:+.2f}  uniform={zu[IDX_LND]:+.2f}")
        for i, nm in enumerate(RATE_NAMES):
            print(f"  z({nm:8s}) stationary={zs[K+1+i]:+.2f}  uniform={zu[K+1+i]:+.2f}")


def verdict(res):
    """Return (ok: bool, failures: list[str])."""
    b1, gf, fi = res["b1"], res["gf"], res["fi"]
    z = b1["z"]
    f = []
    if res["lin"] > 1e-8:
        f.append(f"basis linearity {res['lin']:.1e} > 1e-8")
    if max(re for *_, re in res["fd"]) > 1e-3:
        f.append("FD score check > 1e-3")
    if res["equiv"] > 1e-6:
        f.append(f"vmap!=loop {res['equiv']:.1e} > 1e-6")
    if gf["gauge"] > 1e-5:
        f.append(f"grid-free gauge {gf['gauge']:.1e} > 1e-5")
    if gf["maxz"] > THRESH["maxz_gridfree"]:
        f.append(f"grid-free max|z| {gf['maxz']:.1f} > {THRESH['maxz_gridfree']}")
    if np.abs(z[:K]).max() > THRESH["z_knot"]:
        f.append(f"force-knot max|z| {np.abs(z[:K]).max():.1f} > {THRESH['z_knot']}")
    if abs(z[IDX_LND]) > THRESH["z_lnD"]:
        f.append(f"|z(lnD)| {abs(z[IDX_LND]):.1f} > {THRESH['z_lnD']}")
    if np.abs(z[K + 1:K + 5]).max() > THRESH["z_rate"]:
        f.append(f"rate max|z| {np.abs(z[K+1:K+5]).max():.1f} > {THRESH['z_rate']} "
                 f"(if Fisher rate ratios are ~1.0 and traces are short at high N, "
                 f"this is the O(N/sqrt(M)) windowing boundary effect -- use longer "
                 f"traces, e.g. BARTLETT_TT=600)")
    if fi["rel_frob"] > THRESH["fisher_relfrob"]:
        f.append(f"Fisher relFrob {fi['rel_frob']:.2f} > {THRESH['fisher_relfrob']}")
    lo, hi = THRESH["diag_ratio"]
    key_ratios = np.concatenate([[fi["diag_ratio"][IDX_LND]],
                                 fi["diag_ratio"][K + 1:K + 5]])
    if (key_ratios < lo).any() or (key_ratios > hi).any():
        f.append(f"Fisher diag ratio (lnD/rates) outside [{lo},{hi}]: "
                 f"{np.round(key_ratios,2).tolist()}")
    return (len(f) == 0), f


def _figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gf, b1, fi = res["gf"], res["b1"], res["fi"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    g = gf["grid"]
    a = ax[0]
    a.plot(g, gf["U_true"], color="#333", lw=2, label="true U(x)")
    a.set_xlabel("x (nm)"); a.set_ylabel("U (kT)", color="#333")
    a.set_ylim(-0.5, 8)
    a2 = a.twinx()
    z = gf["z"].copy(); z[~gf["live"]] = np.nan
    a2.plot(g, z, color="#d62728", lw=1.3, label="z = score/SE")
    for y in (-3, 0, 3):
        a2.axhline(y, color="#d62728", lw=.5, ls=":")
    a2.set_ylabel("grid-free z", color="#d62728")
    a.set_title(f"Bartlett-1 grid-free  max|z|={gf['maxz']:.1f}")
    a = ax[1]
    zk = b1["z"]
    lbl = [f"g{i}" for i in range(K)] + ["lnD"] + [n.split("_",1)[1] for n in RATE_NAMES]
    a.bar(range(P_DIM), zk, color="#1f77b4")
    for y in (-3, 3):
        a.axhline(y, color="k", lw=.6, ls=":")
    a.set_xticks(range(P_DIM)); a.set_xticklabels(lbl, rotation=90, fontsize=7)
    a.set_ylabel("z = mean/SE"); a.set_title("Bartlett-1 per-parameter z")
    a = ax[2]
    cov, negEH = fi["cov"], fi["negEH"]
    a.scatter(negEH.ravel(), cov.ravel(), s=8, alpha=.5)
    lim = [min(negEH.min(), cov.min()), max(negEH.max(), cov.max())]
    a.plot(lim, lim, "k--", lw=1)
    a.set_xlabel("-E[H] entries"); a.set_ylabel("Cov(score) entries")
    a.set_title(f"Fisher identity  relFrob={fi['rel_frob']:.2f}")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    p = os.path.join(FIG_DIR, "bartlett_fisher.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"\n  figure: {p}")


# --------------------------------------------------------------------------- #
# Grid-size sweep (does spatial discretisation bias the photophysics scores?)
# --------------------------------------------------------------------------- #
def grid_sweep(n_traces=N_FULL, grids=(96, 128, 160, 224, 320), device=None):
    """Re-score the SAME data across grid sizes.  Because the traces are held
    fixed, any *change* in the scores across grids is pure discretisation error
    (the sampling fluctuation is common to all grids)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*78}\n  GRID-SIZE SWEEP  |  {n_traces} traces held fixed  "
          f"(seed offset {SEED_OFFSET})\n{'='*78}")
    hdr = (f"  {'n_grid':>6} {'dx(nm)':>7} | {'gf max|z|':>9} {'z(lnD)':>7} | "
           f"{'z(a_g)':>7} {'z(a_r)':>7} {'z(bg_g)':>7} {'z(bg_r)':>7} | "
           f"{'meanScore(a_g)':>13}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    rows = []
    for ng in grids:
        res = run(n_traces=n_traces, device=device, n_grid=ng, do_fisher=False,
                  verbose=False)
        z = res["b1"]["z"]; m = res["b1"]["mean"]
        dx = (X_MAX - X_MIN) / (ng - 1)
        print(f"  {ng:6d} {dx:7.4f} | {res['gf']['maxz']:9.2f} {z[IDX_LND]:7.2f} | "
              f"{z[K+1]:7.2f} {z[K+2]:7.2f} {z[K+3]:7.2f} {z[K+4]:7.2f} | "
              f"{m[K+1]:13.4e}")
        rows.append((ng, dx, res["gf"]["maxz"], z.copy(), m.copy()))
    print("\n  Interpretation: rate-score z's STABLE across n_grid => the "
          "offset is a\n  sampling fluctuation (grid-independent).  A monotone "
          "drift with dx =>\n  discretisation bias (would extrapolate to 0 as "
          "n_grid -> inf).")
    return rows


# --------------------------------------------------------------------------- #
# pytest entry point
# --------------------------------------------------------------------------- #
def test_bartlett_fisher_equilibrium():
    res = run(n_traces=N_TEST, verbose=True)
    ok, failures = verdict(res)
    assert ok, "Bartlett/Fisher test failed:\n  - " + "\n  - ".join(failures)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _pilot():
    """Fast calibration: confirm cubic, basis linearity, crossings/photons."""
    cfg = build_cfg()
    theta = build_theta(g_true_knots())
    gnp = np.linspace(X_MIN, X_MAX, N_GRID)
    B, b0 = build_force_basis(cfg, theta, gnp)
    print("basis linearity:", selfcheck_linearity(cfg, theta, gnp, B, b0))
    U = ground_truth_landscape(cfg, theta, gnp)
    U = U - U.min()
    inner = (gnp > 3.5) & (gnp < 11.5)
    print("interior barrier (kT):", round(float(U[(gnp > 5) & (gnp < 8)].max()), 3))
    np.random.seed(0)
    batch, meta = simulate_traces(cfg, theta, "equilibrium", n_traces=4,
                                  min_photons=50, max_photons=None, device="cpu")
    print("photons/trace:", [int(x) for x in batch.lengths], meta["seconds"], "s")


if __name__ == "__main__":
    if "--pilot" in sys.argv:
        _pilot()
    elif "--gridsweep" in sys.argv:
        n = N_FULL if "--full" in sys.argv else N_TEST
        grid_sweep(n_traces=n)
    else:
        n = N_FULL if "--full" in sys.argv else N_TEST
        ng = int(os.environ.get("BARTLETT_GRID", str(N_GRID)))
        res = run(n_traces=n, make_fig=True, verbose=True, n_grid=ng)
        ok, failures = verdict(res)
        print(f"\n{'='*72}")
        print("  VERDICT:", "PASS" if ok else "FAIL")
        for fl in failures:
            print("   -", fl)
        sys.exit(0 if ok else 1)
