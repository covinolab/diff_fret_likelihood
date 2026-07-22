import cython
cimport gsl_utils
from libc.math cimport sqrt, pow, exp
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp

cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double draw_boltzmann_x0(
        gsl_utils.gsl_spline *spline,
        gsl_utils.gsl_interp_accel *acc,
        gsl_utils.gsl_rng *rng,
        cnp.ndarray[double] x_knots,
        int N_grid=2000,
        ):
    """
    One draw from the equilibrium distribution of the spline potential,
        p_eq(x) ∝ exp(-U(x)),   kT = 1,
    where U is the *value* of `spline` (the main loop uses its derivative).

    Support is the spline's own domain [x_knots[0], x_knots[-1]] — that is
    where U is defined; GSL errors outside it and the integrator aborts a
    trajectory that leaves it. So this *is* the Boltzmann distribution for
    this potential, not a range imposed on it.

    Inverse-CDF via trapezoid integration, no heap allocation.
    """
    cdef int j, N = x_knots.shape[0]
    cdef double x_lo = x_knots[0]
    cdef double dx = (x_knots[N - 1] - x_lo) / (N_grid - 1)
    cdef double Ug, Umin, Z = 0.0, u_draw
    cdef double c_prev, c_curr = 0.0, w_prev, w_curr, frac

    # min of U (log-sum-exp shift so exp() stays finite in deep wells)
    gsl_utils.gsl_spline_eval_e(spline, x_lo, acc, &Umin)
    for j in range(1, N_grid):
        gsl_utils.gsl_spline_eval_e(spline, x_lo + j * dx, acc, &Ug)
        if Ug < Umin:
            Umin = Ug

    # total mass Z = ∫ exp(-(U - Umin)) dx
    gsl_utils.gsl_spline_eval_e(spline, x_lo, acc, &Ug)
    w_prev = exp(-(Ug - Umin))
    for j in range(1, N_grid):
        gsl_utils.gsl_spline_eval_e(spline, x_lo + j * dx, acc, &Ug)
        w_curr = exp(-(Ug - Umin))
        Z += 0.5 * (w_prev + w_curr) * dx
        w_prev = w_curr

    # invert the CDF at u ~ Uniform(0, Z)
    u_draw = gsl_utils.gsl_rng_uniform(rng) * Z
    gsl_utils.gsl_spline_eval_e(spline, x_lo, acc, &Ug)
    w_prev = exp(-(Ug - Umin))
    for j in range(1, N_grid):
        gsl_utils.gsl_spline_eval_e(spline, x_lo + j * dx, acc, &Ug)
        w_curr = exp(-(Ug - Umin))
        c_prev = c_curr
        c_curr += 0.5 * (w_prev + w_curr) * dx
        if c_curr >= u_draw:
            if c_curr > c_prev:
                frac = (u_draw - c_prev) / (c_curr - c_prev)
            else:
                frac = 0.0
            return x_lo + (j - 1 + frac) * dx
        w_prev = w_curr

    return x_knots[N - 1]


@cython.boundscheck(False)
@cython.wraparound(False)
def eval_spline(
        cnp.ndarray[double] x_knots,
        cnp.ndarray[double] y_knots,
        cnp.ndarray[double] x_eval,
        str spline_type='cubic'
        ):
    """Cython wrapper for GSL spline interpolation
    
    Parameters
    ----------
    x_knots : array_like
        x values of the knots
    y_knots : array_like
        y values of the knots
    x_eval : array_like
        x values where the spline is evaluated
    
    Returns
    -------
    y_eval : array_like
        y values of the spline at x_eval
    """

    cdef int N_knots = len(x_knots)
    cdef int N_eval = len(x_eval)

    cdef double *x_k = <double *> malloc(N_knots * sizeof(double))
    cdef double *y_k = <double *> malloc(N_knots * sizeof(double))

    for i  from 0 <= i < N_knots:
        x_k[i] = x_knots[i]
        y_k[i] = y_knots[i]

    cdef gsl_utils.gsl_interp_accel *acc
    acc = gsl_utils.gsl_interp_accel_alloc()
    cdef gsl_utils.gsl_spline *spline

    if spline_type == "cubic":
        spline = gsl_utils.gsl_spline_alloc(gsl_utils.gsl_interp_cspline, N_knots)
    elif spline_type == "steffen":
        spline = gsl_utils.gsl_spline_alloc(gsl_utils.gsl_interp_steffen, N_knots)
    elif spline_type == "akima":
        spline = gsl_utils.gsl_spline_alloc(gsl_utils.gsl_interp_akima, N_knots)
    else:
        raise ValueError(f"Unsupported spline type: {spline_type}")

    gsl_utils.gsl_spline_init(spline, x_k, y_k, N_knots)

    cdef cnp.ndarray[double] y_eval = np.empty(N_eval, dtype=np.double)
    cdef double tmp_y
    cdef int status

    for i  from 0 <= i < N_eval:
        status = gsl_utils.gsl_spline_eval_e(spline, x_eval[i], acc, &tmp_y)
        if status != 0:
            break
        y_eval[i] = tmp_y

    gsl_utils.gsl_spline_free (spline)
    gsl_utils.gsl_interp_accel_free (acc)
    free(x_k)
    free(y_k)

    if status != 0:
        return None

    return y_eval


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def smFRET_simulator(
        double Dx,
        cnp.ndarray[double] x_knots,
        cnp.ndarray[double] y_knots,
        double R0,
        double kD,
        double k_gb,
        double k_rb,
        double eta_g,
        double eta_r,
        double C_gg,
        double C_rr,
        double C_gr,
        double C_rg,
        double T,
        int N_max_photons,
        double dt
        ):
    """
    Spline-based potential integrator with complete FRET photophysics.

    Implements:
    1. Langevin dynamics: dx = D*F*dt + sqrt(2*D*dt)*η
       where F = -dU/dx from cubic spline potential U(x)
    2. FRET efficiency: E(r) = R0^6 / (R0^6 + r^6)
    3. Cross-talk matrix for photon detection:
       - φ_g(r) = C_gg*(1-E(r)) + C_rg*E(r)
       - φ_r(r) = C_gr*(1-E(r)) + C_rr*E(r)
    4. Poisson photon statistics:
       - n_g ~ Poisson(η_g * (k_D * φ_g(r) + k_gb) * dt)
       - n_r ~ Poisson(η_r * (k_D * φ_r(r) + k_rb) * dt)
       Within each `dt` step, n_g/n_r arrival times are drawn uniformly on
       [i*dt, (i+1)*dt] — statistically exact under the homogeneous-Poisson
       assumption the simulator already commits to per step.

    Parameters
    ----------
    N_max_photons : int
        Per-channel budget for arrival-time arrays. If exceeded mid-trace,
        the simulation aborts and returns (None, None).
    dt : float
        Integration time step.
    T : float
        Total simulation time.

    Returns
    -------
    G_times : np.ndarray
        Green-channel photon arrival times (variable length ≤ N_max_photons).
    R_times : np.ndarray
        Red-channel photon arrival times (variable length ≤ N_max_photons).
    """

    # Initialize loop variables
    cdef long i
    cdef int N_knots = len(x_knots)
    cdef long N = <long>(T / dt)

    # Initialize random number generator
    cdef gsl_utils.gsl_rng_type * rng_type
    cdef gsl_utils.gsl_rng * rng
    cdef long seed = np.random.randint(low=1, high=2**63)

    gsl_utils.gsl_rng_env_setup()
    rng_type = gsl_utils.gsl_rng_default
    rng = gsl_utils.gsl_rng_alloc(rng_type)
    gsl_utils.gsl_rng_set(rng, seed)

    # Initialize spline knots as C arrays
    cdef double *x_k = <double *> malloc(N_knots * sizeof(double))
    cdef double *y_k = <double *> malloc(N_knots * sizeof(double))

    for i in range(N_knots):
        x_k[i] = x_knots[i]
        y_k[i] = y_knots[i]

    # Setup spline interpolation
    cdef gsl_utils.gsl_interp_accel *acc
    acc = gsl_utils.gsl_interp_accel_alloc()
    cdef gsl_utils.gsl_spline *spline

    spline = gsl_utils.gsl_spline_alloc(gsl_utils.gsl_interp_cspline, N_knots)

    gsl_utils.gsl_spline_init(spline, x_k, y_k, N_knots)

    # Langevin dynamics coefficients (overdamped limit, kT=1)
    cdef double Ax = Dx * dt          # Drift coefficient
    cdef double Bx = sqrt(2.0 * Ax)   # Diffusion coefficient

    # Pre-compute R0^6 for FRET efficiency
    cdef double R0_6 = pow(R0, 6.0)

    # State variables
    cdef double xold = draw_boltzmann_x0(spline, acc, rng, x_knots)
    cdef double xnew, Fx, spline_val
    cdef double r, r_6, E, phi_g, phi_r, lambda_g, lambda_r
    cdef unsigned int n_g, n_r, k_emit
    cdef int status = 0
    cdef bint budget_exceeded = False   # must init: read at return even if never tripped

    # Output arrays: per-photon arrival times (variable length, capped at N_max_photons).
    cdef cnp.ndarray[double] G_times = np.empty(N_max_photons, dtype=np.double)
    cdef cnp.ndarray[double] R_times = np.empty(N_max_photons, dtype=np.double)
    cdef long n_g_total = 0
    cdef long n_r_total = 0

    # Main integration loop
    for i in range(1, N):

        # Evaluate force directly from gradient spline: F = g(x)
        status = gsl_utils.gsl_spline_eval_deriv_e(spline, xold, acc, &spline_val)
        if status != 0:
            break

        Fx = -spline_val

        # Langevin integration step
        xnew = xold + Ax * Fx + Bx * gsl_utils.gsl_ran_gaussian_ziggurat(rng, 1.0)

        # Use position as distance r for FRET (ensure positive)
        r = xnew
        if r < 0.0:
            r = 0.0
        r_6 = pow(r, 6.0)

        # FRET efficiency: E(r) = R0^6 / (R0^6 + r^6)
        E = R0_6 / (R0_6 + r_6)

        # Cross-talk corrected detection probabilities
        phi_g = C_gg * (1.0 - E) + C_rg * E
        phi_r = C_gr * (1.0 - E) + C_rr * E

        # Poisson rates for photon emission
        lambda_g = eta_g * (kD * phi_g + k_gb) * dt
        lambda_r = eta_r * (kD * phi_r + k_rb) * dt

        # Sample photon counts from Poisson distribution
        n_g = gsl_utils.gsl_ran_poisson(rng, lambda_g)
        n_r = gsl_utils.gsl_ran_poisson(rng, lambda_r)

        # Emit arrival times for each photon in this step.
        # Times are uniform on [i*dt, (i+1)*dt] (homogeneous Poisson within dt).
        if n_g > 0:
            if n_g_total + <long>n_g > N_max_photons:
                budget_exceeded = True
                break
            for k_emit in range(n_g):
                G_times[n_g_total] = (<double>i + gsl_utils.gsl_rng_uniform(rng)) * dt
                n_g_total += 1
        if n_r > 0:
            if n_r_total + <long>n_r > N_max_photons:
                budget_exceeded = True
                break
            for k_emit in range(n_r):
                R_times[n_r_total] = (<double>i + gsl_utils.gsl_rng_uniform(rng)) * dt
                n_r_total += 1

        xold = xnew

    # Free allocated memory
    gsl_utils.gsl_spline_free(spline)
    gsl_utils.gsl_interp_accel_free(acc)
    gsl_utils.gsl_rng_free(rng)
    free(x_k)
    free(y_k)

    if status != 0 or budget_exceeded:
        return None, None

    return G_times[:n_g_total], R_times[:n_r_total]