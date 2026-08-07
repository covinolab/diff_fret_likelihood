"""PRIMARY evaluator: marginal photon-stream log-likelihood (SPEC section 7.4).

Continuous-potential generalisation of the Gopich-Szabo photon-by-photon
likelihood.  The latent reaction coordinate is integrated out on a spatial
grid; the observation is a *marked* (coloured) point process whose intensity in
each channel is ``mu_c(x)``.  For photons at times ``t_1<...<t_K`` with colours
``c_k`` and window ``[0, T]`` (SPEC section 4.6):

    p = 1^T e^{(L-Lambda)(T-t_K)} V_{c_K} e^{(L-Lambda) tau_K} ... V_{c_1}
            e^{(L-Lambda) tau_1} p_0 ,   tau_k = t_k - t_{k-1}, Lambda=diag(mu).

Working in the symmetric basis ``s_i = e^{-u_i/2}`` (``L = diag(s) L_sym
diag(s)^{-1}``, ``V_c`` and ``Lambda`` diagonal hence invariant), the whole
product collapses to

    p = s^T [ prod e^{A tau} V_c ] (p_0 / s),   A = L_sym - diag(mu),

with ``A`` symmetric NSD.  One ``eigh(A)`` gives ``A = Q diag(lam) Q^T``; each
gap is then two mat-vecs ``v <- Q (exp(lam tau) * (Q^T v))`` (SPEC 7.4 step 4),
all gaps sharing the spectrum.  A running log-normaliser keeps everything in
log-space (SPEC 7.4 step 8).

Data / units convention.  The ``smFRET_sbi`` wrapper stores inter-photon times
``ipt`` (ms) with ``ipt[0]=0`` (the absolute time of the first photon is
dropped) and no separate window length.  We therefore condition on the first
photon: ``t_1 = 0`` (leading gap 0) and ``T = t_K`` (trailing gap 0).  The
likelihood is over the observed inter-photon gaps, which is standard for
photon-by-photon smFRET.  Explicit ``t_1`` / ``T`` can be supplied when known.
"""

from __future__ import annotations

import warnings

import torch

from .config import DTYPE
from .generator import smoluchowski, symmetrize, stationary, min_gauge
from .photophysics import EffectiveRates, emission_rates

# warn only ONCE per fallback kind (this runs inside the HMC hot loop; a bare
# ``warnings.warn`` under an "always" filter would spam thousands of times).
_EIGH_WARNED = set()


def _warn_once(kind: str, msg: str) -> None:
    if kind not in _EIGH_WARNED:
        _EIGH_WARNED.add(kind)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# Prepared propagator (eigendecomposition of the tilted symmetric generator)
# ---------------------------------------------------------------------------
def _robust_eigh(A: torch.Tensor):
    """``torch.linalg.eigh`` with a gradient-preserving CPU fallback.

    cuSOLVER (CUDA ``eigh``) can fail to converge (error 194) on ill-conditioned
    symmetric matrices -- e.g. the steep landscapes an HMC sampler transiently
    proposes -- where CPU LAPACK succeeds.  On failure we retry on the CPU (often
    the matrix is fine and cuSOLVER was merely flaky, so we recover the *correct*
    decomposition rather than crashing / spuriously rejecting), then as a last
    resort add escalating symmetric jitter.  No ``detach``: gradients to the
    landscape / D still flow (HMC needs them).
    """
    try:
        return torch.linalg.eigh(A)
    except torch.linalg.LinAlgError:
        pass
    A_cpu = A.cpu() if A.is_cuda else A
    if A.is_cuda:  # LAPACK is markedly more robust than cuSOLVER
        try:
            lam, Q = torch.linalg.eigh(A_cpu)
            _warn_once("cpu", "eigh failed on CUDA (cuSOLVER); fell back to the "
                       "slower CPU LAPACK path. This is triggered by ill-conditioned "
                       "generators (e.g. steep landscapes proposed during HMC). If it "
                       "recurs, reduce step_size, tighten gp_sigma, or sample on CPU. "
                       "(warned once)")
            return lam.to(A.device), Q.to(A.device)
        except torch.linalg.LinAlgError:
            pass
    eye = torch.eye(A_cpu.shape[0], dtype=A_cpu.dtype, device=A_cpu.device)
    jit = 1e-9 * max(1.0, float(A_cpu.detach().abs().max()))
    last_err: Exception | None = None
    for _ in range(7):
        try:
            lam, Q = torch.linalg.eigh(A_cpu + jit * eye)
            _warn_once("jitter", "eigh needed escalating jitter (up to "
                       f"{jit:.1e}) to decompose an ill-conditioned generator; the "
                       "landscape at this parameter is very steep -- results here are "
                       "approximate. Consider a tighter gp_sigma / smaller step_size. "
                       "(warned once)")
            return lam.to(A.device), Q.to(A.device)
        except torch.linalg.LinAlgError as err:
            last_err = err
            jit *= 10.0
    # genuinely undecomposable even with escalating jitter: re-raise the real
    # LinAlgError (a bare ``raise`` here has no active exception -> confusing
    # "No active exception to reraise").
    raise last_err if last_err is not None else RuntimeError("eigh failed to decompose A")


class Propagator:
    """Holds ``(lam, Q, s)`` and the emission rates on the grid for one eval."""

    def __init__(self, lam, Q, s, mu_G, mu_R, jitter_used):
        self.lam = lam          # [G] eigenvalues (<= 0)
        self.Q = Q              # [G, G] orthonormal eigenvectors
        self.s = s              # [G] = e^{-u/2}
        self.mu_G = mu_G        # [G]
        self.mu_R = mu_R        # [G]
        self.jitter_used = jitter_used

    def propagate(self, v: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Apply ``e^{A tau}`` to ``v``.

        ``v`` is ``[G]`` with scalar ``tau`` or ``[G, B]`` with ``tau`` ``[B]``.
        """
        if v.dim() == 1:
            return self.Q @ (torch.exp(self.lam * tau) * (self.Q.T @ v))
        # batched: v [G, B], tau [B]
        Vt = self.Q.T @ v                                    # [G, B]
        Vt = Vt * torch.exp(self.lam.unsqueeze(1) * tau.unsqueeze(0))
        return self.Q @ Vt


def build_propagator(
    potential,
    D: torch.Tensor,
    rates: EffectiveRates,
    grid: torch.Tensor,
    C: torch.Tensor,
    R0: float,
    dx: float,
    jitter: float = 1e-12,
    check_spectrum: bool = False,
) -> Propagator:
    u_grid = _BasePotential_on_grid(potential, grid)
    return build_propagator_from_u(
        u_grid, D, rates, grid, C, R0, dx, jitter, check_spectrum
    )


def build_propagator_from_u(
    u_grid: torch.Tensor,
    D: torch.Tensor,
    rates: EffectiveRates,
    grid: torch.Tensor,
    C: torch.Tensor,
    R0: float,
    dx: float,
    jitter: float = 1e-12,
    check_spectrum: bool = False,
) -> Propagator:
    """Build the eigendecomposition propagator from a gauge-fixed ``u_grid``."""
    mu_G, mu_R = emission_rates(grid, rates, C, R0)
    mu = mu_G + mu_R
    L = smoluchowski(u_grid, D, dx)
    L_sym, s = symmetrize(L, u_grid)
    A = L_sym - torch.diag(mu)
    if jitter:
        A = A + jitter * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    lam, Q = _robust_eigh(A)
    if check_spectrum:
        scale = max(1.0, float(A.abs().max()))
        if lam.max() > 1e-6 * scale:
            raise RuntimeError(f"tilted generator has positive eigenvalue "
                               f"{float(lam.max()):.3e} (should be <= 0)")
    return Propagator(lam, Q, s, mu_G, mu_R, jitter)


def _BasePotential_on_grid(potential, grid):
    """Potential on the grid with the ``exp(-u)`` overflow guard applied."""
    return min_gauge(potential.on_grid(grid))


# ---------------------------------------------------------------------------
# Single-trace marginal log-likelihood
# ---------------------------------------------------------------------------
def marginal_loglik(
    times: torch.Tensor,
    colors: torch.Tensor,
    T: float | None,
    potential,
    D: torch.Tensor,
    rates: EffectiveRates,
    grid: torch.Tensor,
    C: torch.Tensor,
    R0: float,
    p0: torch.Tensor | None = None,
    jitter: float = 1e-12,
) -> torch.Tensor:
    """Marginal log-likelihood of one trace (scalar tensor).

    ``times``  : [K] absolute arrival times (ms), non-decreasing, may start at 0.
    ``colors`` : [K] int64 in {0=green, 1=red}.
    ``T``      : window length (ms); default ``times[-1]`` (condition on first
                 photon, no trailing gap).
    """
    # SPEC Remark 1: the inter-photon gaps (plus the trailing gap) must tile
    # [0, T] exactly once.  Guard against unsorted arrival times or a window
    # shorter than the last photon -- either would silently mis-tile the survival
    # integral and bias log L.  (Single-trace entry point only; the batched hot
    # loop and ``_forward_recursion_single`` take pre-checked/pre-differenced
    # gaps and are left untouched for speed.)
    if times.numel() > 1:
        assert bool((times[1:] - times[:-1] >= -1e-9).all()), \
            "marginal_loglik: `times` must be non-decreasing (gaps tile [0,T])"
    if T is not None and times.numel() > 0:
        assert float(T) >= float(times[-1]) - 1e-9, \
            "marginal_loglik: T must be >= times[-1] (trailing gap >= 0)"
    u_grid = _BasePotential_on_grid(potential, grid)
    dx = float(grid[1] - grid[0]) if grid.shape[0] > 1 else 1.0
    prop = build_propagator_from_u(u_grid, D, rates, grid, C, R0, dx, jitter)
    return _forward_recursion_single(times, colors, T, prop, u_grid, p0)


def _forward_recursion_single(times, colors, T, prop: Propagator, u_grid, p0):
    device = u_grid.device
    K = times.shape[0]
    if T is None:
        T = float(times[-1]) if K > 0 else 0.0

    s = prop.s
    if p0 is None:
        p0 = stationary(u_grid)
    v = p0 / s                                   # map to symmetric basis
    c0 = v.abs().sum()
    v = v / c0
    log_norm = torch.log(c0)

    t_prev = torch.zeros((), dtype=DTYPE, device=device)
    for k in range(K):
        tau = times[k] - t_prev
        v = prop.propagate(v, tau)
        emit = prop.mu_G if int(colors[k]) == 0 else prop.mu_R
        v = v * emit
        c = v.abs().sum()
        v = v / c
        log_norm = log_norm + torch.log(c)
        t_prev = times[k]

    # trailing survival gap
    tau_final = torch.as_tensor(T, dtype=DTYPE, device=device) - t_prev
    v = prop.propagate(v, tau_final)

    total = torch.dot(s, v)
    return torch.log(total.clamp_min(1e-300)) + log_norm


# ---------------------------------------------------------------------------
# Photon recursion step (extracted so it can be torch.compile'd)
# ---------------------------------------------------------------------------
def _recur_step(V, log_norm, lam, Q, muG, muR, gap_k, colk, maskk, ones_col):
    """One photon step of the batched forward recursion.

    All recurrent state ``(V, log_norm)`` is threaded through so the whole step
    is a pure tensor->tensor map -- friendly to ``torch.compile`` /
    CUDA-graph capture (the Python ``for k`` loop stays outside).  Math is
    identical to the inline lockstep recursion:

        V <- diag(emit_k) . e^{A tau_k} V ,   log_norm += log ||.||_1

    ``V``:[G,B]  ``log_norm``:[B]  ``lam``:[G]  ``Q``:[G,G]  ``muG``/``muR``:[G,1]
    ``gap_k``:[B]  ``colk``/``maskk``:[1,B]  ``ones_col``:[G,B].
    """
    Vt = Q.T @ V                                          # to eigenbasis
    Vt = Vt * torch.exp(lam.unsqueeze(1) * gap_k.unsqueeze(0))
    V = Q @ Vt                                            # e^{A tau} V
    emit = torch.where(colk == 0, muG, muR)               # [G,B]
    emit = torch.where(maskk, emit, ones_col)             # padded -> 1
    V = V * emit
    c = V.abs().sum(dim=0)                                # [B]
    c = torch.where(c > 0, c, torch.ones_like(c))
    V = V / c.unsqueeze(0)
    log_norm = log_norm + torch.log(c).to(log_norm.dtype)
    return V, log_norm


def _recur_chunk(V, log_norm, lam, Q, muG, muR, gaps, cols, masks, ones_col):
    """Apply ``L`` consecutive photon steps (``L = gaps.shape[1]``).

    The inner loop is unrolled at ``torch.compile`` trace time (fixed ``L``), so
    the whole chunk compiles ONCE and, under ``mode="reduce-overhead"``, replays
    as a single CUDA graph -- collapsing ~L kernel launches into one replay.
    Calling the compiled *single* step in a Python loop instead makes Dynamo
    recompile per photon (per-column storage offsets differ) -> compile blow-up.
    ``gaps``:[B,L]  ``cols``/``masks``:[B,L].
    """
    L = gaps.shape[1]
    for j in range(L):
        V, log_norm = _recur_step(
            V, log_norm, lam, Q, muG, muR,
            gaps[:, j], cols[:, j].unsqueeze(0), masks[:, j].unsqueeze(0),
            ones_col,
        )
    return V, log_norm


# Default photons-per-compiled-chunk. Big enough to amortise launch overhead,
# small enough that the unrolled graph compiles quickly (once, then cached).
_CHUNK = 64

# Cache compiled chunks by (mode, L): ``torch.compile`` returns a fresh wrapper
# each call, so compiling once keeps the artifact warm across the many
# evaluations of a fit / HMC chain.
_COMPILED_CHUNKS: dict = {}


def _get_compiled_chunk(compile_mode, L):
    key = (compile_mode, L)
    fn = _COMPILED_CHUNKS.get(key)
    if fn is None:
        fn = torch.compile(_recur_chunk, mode=compile_mode)
        _COMPILED_CHUNKS[key] = fn
    return fn


# ---------------------------------------------------------------------------
# Batched marginal log-likelihood (shared parameters across traces)
# ---------------------------------------------------------------------------
def marginal_loglik_batch(
    ipt: torch.Tensor,
    colors: torch.Tensor,
    mask: torch.Tensor,
    potential,
    D: torch.Tensor,
    rates: EffectiveRates,
    grid: torch.Tensor,
    C: torch.Tensor,
    R0: float,
    p0: torch.Tensor | None = None,
    jitter: float = 1e-12,
    reduce: str = "sum",
    compile_mode: str | None = None,
    propagate_dtype: "torch.dtype | None" = None,
) -> torch.Tensor:
    """Batched marginal log-likelihood over independent traces.

    ``ipt``    : [B, Kmax] inter-photon gaps (ms); ``ipt[:,0]`` is the leading
                 gap (0 from the simulator wrapper).  Padded entries ignored.
    ``colors`` : [B, Kmax] int64 in {0, 1}.
    ``mask``   : [B, Kmax] bool, True for valid photons.
    ``reduce`` : 'sum' -> total log-lik; 'none' -> per-trace [B].
    ``compile_mode`` : ``None`` (eager, default) or a ``torch.compile`` mode for
                 the per-photon step, e.g. ``"reduce-overhead"`` (CUDA-graphs) --
                 numerically transparent, see ``tests/test_compile.py``.
    ``propagate_dtype`` : ``None``/float64 (default) or ``torch.float32`` to run
                 the per-photon recursion in mixed precision (fp32 matmuls, fp64
                 log-normaliser).  Big GPU win where fp64 is throttled; accuracy
                 gated by ``tests/test_fp32.py`` + ``tests/test_bartlett_fisher.py``.

    All traces share the parameters, hence one eigendecomposition; the
    photon-by-photon recursion runs in lockstep with masking (padded steps are
    the identity).  ``T = t_K`` per trace (no trailing gap).
    """
    dx = float(grid[1] - grid[0]) if grid.shape[0] > 1 else 1.0
    u_grid = _BasePotential_on_grid(potential, grid)
    prop = build_propagator_from_u(u_grid, D, rates, grid, C, R0, dx, jitter)

    G = grid.shape[0]
    B, Kmax = ipt.shape
    device = grid.device
    s = prop.s

    if p0 is None:
        p0 = stationary(u_grid)
    v0 = (p0 / s).unsqueeze(1).expand(G, B).contiguous()      # [G, B]

    gap = ipt.clone()
    gap = torch.where(mask, gap, torch.zeros_like(gap))        # padded -> 0
    ones_col = torch.ones(G, B, dtype=DTYPE, device=device)

    V = v0
    c0 = V.abs().sum(dim=0)
    V = V / c0.unsqueeze(0)
    log_norm = torch.log(c0)                                   # [B]

    muG = prop.mu_G.unsqueeze(1)     # [G,1]
    muR = prop.mu_R.unsqueeze(1)     # [G,1]

    lam, Q = prop.lam, prop.Q

    # Optional mixed precision: run the per-photon recursion in propagate_dtype
    # (e.g. float32 -- a large win on GPUs where fp64 is throttled) while the
    # eigh (already done, fp64) and the running log-normaliser stay float64.
    if propagate_dtype is not None and propagate_dtype != DTYPE:
        lam = lam.to(propagate_dtype)
        Q = Q.to(propagate_dtype)
        muG = muG.to(propagate_dtype)
        muR = muR.to(propagate_dtype)
        ones_col = ones_col.to(propagate_dtype)
        gap = gap.to(propagate_dtype)
        V = V.to(propagate_dtype)                              # log_norm stays fp64

    if compile_mode is None:
        # eager: byte-identical to the legacy lockstep loop
        for k in range(Kmax):
            V, log_norm = _recur_step(
                V, log_norm, lam, Q, muG, muR,
                gap[:, k], colors[:, k].unsqueeze(0), mask[:, k].unsqueeze(0),
                ones_col,
            )
    else:
        # compiled: process fixed-size chunks (one compile, bounded unroll).
        # Pad Kmax up to a multiple of L with mask=False steps -- those are the
        # identity (gap 0 -> e^{A.0}=I; emit 1) and are absorbed exactly by the
        # running log-normaliser, so the result is unchanged.
        L = _CHUNK
        pad = (-Kmax) % L
        if pad:
            gap = torch.nn.functional.pad(gap, (0, pad))
            colors = torch.nn.functional.pad(colors, (0, pad))
            mask = torch.nn.functional.pad(mask, (0, pad))
        chunk = _get_compiled_chunk(compile_mode, L)
        for k0 in range(0, gap.shape[1], L):
            V, log_norm = chunk(
                V, log_norm, lam, Q, muG, muR,
                gap[:, k0:k0 + L].contiguous(),
                colors[:, k0:k0 + L].contiguous(),
                mask[:, k0:k0 + L].contiguous(),
                ones_col,
            )

    total = s @ V.to(s.dtype)                                  # [B] (back to fp64)
    per_trace = torch.log(total.clamp_min(1e-300)) + log_norm
    if reduce == "sum":
        return per_trace.sum()
    if reduce == "none":
        return per_trace
    raise ValueError(f"unknown reduce {reduce!r}")


# ---------------------------------------------------------------------------
# Reference implementation (slow; probability space; for tests only)
# ---------------------------------------------------------------------------
def reference_loglik(
    L: torch.Tensor,
    mu_G: torch.Tensor,
    mu_R: torch.Tensor,
    times: torch.Tensor,
    colors: torch.Tensor,
    T: float | None,
    p0: torch.Tensor,
) -> torch.Tensor:
    """Independent forward value via ``matrix_exp`` in probability space.

    No symmetrisation, no ``eigh`` -- validates the fast path (SPEC tests 4-5).
    """
    device = L.device
    K = times.shape[0]
    if T is None:
        T = float(times[-1]) if K > 0 else 0.0
    mu = mu_G + mu_R
    A = L - torch.diag(mu)
    G = L.shape[0]

    v = p0.clone()
    c0 = v.abs().sum()
    v = v / c0
    log_norm = torch.log(c0)
    t_prev = torch.zeros((), dtype=DTYPE, device=device)
    for k in range(K):
        tau = times[k] - t_prev
        v = torch.linalg.matrix_exp(A * tau) @ v
        emit = mu_G if int(colors[k]) == 0 else mu_R
        v = v * emit
        c = v.abs().sum()
        v = v / c
        log_norm = log_norm + torch.log(c)
        t_prev = times[k]
    tau_final = torch.as_tensor(T, dtype=DTYPE, device=device) - t_prev
    v = torch.linalg.matrix_exp(A * tau_final) @ v
    ones = torch.ones(G, dtype=DTYPE, device=device)
    total = torch.dot(ones, v)
    return torch.log(total.clamp_min(1e-300)) + log_norm
