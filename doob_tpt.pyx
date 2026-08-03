# doob_tpt.pyx
#
# Transition-path-time (TPT) sampler for 1-D overdamped Langevin dynamics on a
# cubic-spline potential, via Doob's h-transform with the committor as the
# conditioning function. The unconditioned step is the overdamped-Langevin core
# lifted verbatim from smFRET_simulator; the only addition is the reactive drift.
#
# Shares gsl_utils.pxd with the smFRET module (same GSL symbols).

import cython
cimport gsl_utils
from libc.math cimport sqrt, exp
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp

cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def doob_tpt(
        cnp.ndarray[double] x_knots,
        cnp.ndarray[double] y_knots,
        double D,
        double epsilon,
        double dt,
        long N_max_steps,
        int N_grid=2000,
        ):
    """
    Sample one transition-path time for overdamped Langevin motion on a
    cubic-spline potential U(x) (units kT = 1).

    Unconditioned dynamics
        dx = -D U'(x) dt + sqrt(2 D dt) * eta                       (eta ~ N(0,1))

    Committor (backward Kolmogorov, reference states = spline-domain edges)
        q(x) = int_{x0}^{x} e^{U} dy / int_{x0}^{xL} e^{U} dy ,   q'(x) ~ e^{U(x)}

    Doob h-transformed (reactive) dynamics
        dx = [ -D U'(x) + 2 D q'(x)/q(x) ] dt + sqrt(2 D dt) * eta

    The path starts on the q = epsilon isocommittor and ends on q = 1 - epsilon.
    The lower interface is reflecting (a last-crossing transition path never
    re-enters the reactant side, and this keeps the walker off the q -> 0 drift
    singularity); the upper interface is absorbing. Both interfaces -> the true
    TPT as epsilon -> 0. Drop the reflection for the pure conditioned first-passage.

    q(x) depends on U only, so the committor grid is D-independent and, at fixed
    geometry, TPT scales as 1/D -- a cheap correctness check.

    Parameters
    ----------
    x_knots, y_knots : double[:]   spline knots; y_knots are potential *values* U.
    D                : double      diffusion coefficient.
    epsilon          : double      isocommittor cut in (0, 0.5); interfaces at eps, 1-eps.
    dt               : double      integration time step.
    N_max_steps      : long        step budget for one path.
    N_grid           : int         grid resolution for the committor integral.

    Returns
    -------
    float : the transition-path time, or -1.0 if q = 1 - epsilon is not reached
            within N_max_steps.
    """
    cdef int N_knots = x_knots.shape[0]
    cdef int j, idx
    cdef long step
    cdef double x_lo = x_knots[0]
    cdef double x_hi = x_knots[N_knots - 1]
    cdef double dxg = (x_hi - x_lo) / (N_grid - 1)

    # --- cubic-spline potential U (same construction as the smFRET simulator) ---
    cdef double *x_k = <double *> malloc(N_knots * sizeof(double))
    cdef double *y_k = <double *> malloc(N_knots * sizeof(double))
    for j in range(N_knots):
        x_k[j] = x_knots[j]
        y_k[j] = y_knots[j]
    cdef gsl_utils.gsl_interp_accel *acc = gsl_utils.gsl_interp_accel_alloc()
    cdef gsl_utils.gsl_spline *spline = gsl_utils.gsl_spline_alloc(
        gsl_utils.gsl_interp_cspline, N_knots)
    gsl_utils.gsl_spline_init(spline, x_k, y_k, N_knots)

    # --- committor on a grid:  C[j] = int exp(U - Umax) dx ,  q = C / C[-1] ---
    #     (Umax shift keeps exp() bounded; it cancels in every q'/q ratio.)
    cdef double *C = <double *> malloc(N_grid * sizeof(double))
    cdef double Ug, Umax, w_prev, w_curr
    gsl_utils.gsl_spline_eval_e(spline, x_lo, acc, &Umax)
    for j in range(1, N_grid):
        gsl_utils.gsl_spline_eval_e(spline, x_lo + j * dxg, acc, &Ug)
        if Ug > Umax:
            Umax = Ug
    C[0] = 0.0
    gsl_utils.gsl_spline_eval_e(spline, x_lo, acc, &Ug)
    w_prev = exp(Ug - Umax)
    for j in range(1, N_grid):
        gsl_utils.gsl_spline_eval_e(spline, x_lo + j * dxg, acc, &Ug)
        w_curr = exp(Ug - Umax)
        C[j] = C[j - 1] + 0.5 * (w_prev + w_curr) * dxg
        w_prev = w_curr
    cdef double C_tot = C[N_grid - 1]

    # --- isocommittor interfaces  x_A (q = eps)  and  x_B (q = 1 - eps) ---
    cdef double c_lo = epsilon * C_tot
    cdef double c_hi = (1.0 - epsilon) * C_tot
    cdef double x_A = x_lo, x_B = x_hi
    for j in range(1, N_grid):
        if C[j - 1] < c_lo and c_lo <= C[j]:
            x_A = x_lo + (j - 1 + (c_lo - C[j - 1]) / (C[j] - C[j - 1])) * dxg
        if C[j - 1] < c_hi and c_hi <= C[j]:
            x_B = x_lo + (j - 1 + (c_hi - C[j - 1]) / (C[j] - C[j - 1])) * dxg
            break

    # --- random number generator (mirrors smFRET_simulator) ---
    cdef gsl_utils.gsl_rng_type *rng_type
    cdef gsl_utils.gsl_rng *rng
    cdef long seed = np.random.randint(low=1, high=2**63)
    gsl_utils.gsl_rng_env_setup()
    rng_type = gsl_utils.gsl_rng_default
    rng = gsl_utils.gsl_rng_alloc(rng_type)
    gsl_utils.gsl_rng_set(rng, seed)

    # --- integrate the reactive (h-transformed) trajectory ---
    cdef double B = sqrt(2.0 * D * dt)          # noise amplitude, matches A=D*dt
    cdef double x = x_A, x_new, Up, Ux, Cx, drift, g, frac
    cdef double tpt = -1.0
    cdef int status = 0

    for step in range(N_max_steps):
        status = gsl_utils.gsl_spline_eval_deriv_e(spline, x, acc, &Up)   # U'(x)
        if status != 0:
            break
        gsl_utils.gsl_spline_eval_e(spline, x, acc, &Ux)                  # U(x)

        # C(x) = int_{x0}^{x} exp(U - Umax) dy  (linear interp on the grid)
        g = (x - x_lo) / dxg
        idx = <int> g
        if idx < 0:
            idx = 0
        elif idx > N_grid - 2:
            idx = N_grid - 2
        frac = g - idx
        Cx = C[idx] + frac * (C[idx + 1] - C[idx])

        # h-transformed drift:  -D U' + 2 D q'/q ,   q'/q = exp(U - Umax) / C(x)
        drift = -D * Up + 2.0 * D * exp(Ux - Umax) / Cx
        x_new = x + drift * dt + B * gsl_utils.gsl_ran_gaussian_ziggurat(rng, 1.0)

        if x_new < x_A:            # reflect at the reactant interface
            x_new = 2.0 * x_A - x_new
        if x_new >= x_B:           # absorb at the product interface
            tpt = (step + 1) * dt
            break

        x = x_new

    gsl_utils.gsl_spline_free(spline)
    gsl_utils.gsl_interp_accel_free(acc)
    gsl_utils.gsl_rng_free(rng)
    free(x_k)
    free(y_k)
    free(C)

    return tpt
