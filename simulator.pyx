import cython
cimport gsl_utils
from libc.math cimport sqrt, pow, exp, ceil, isfinite
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
    One draw from the equilibrium distribution of the spline potential.

    Support is the spline's own domain [x_knots[0], x_knots[-1]] — that is
    where U is defined; GSL errors outside it and the integrator aborts a
    trajectory that leaves it.

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


# Estimate of the maximum photon count per trace, used to size the arrival-time vectors
# Z standard deviations above the mean bound, plus a constant.  Z = 10 puts the
# per-trace overflow probability below 1e-15 (Poisson Chernoff) once mu is large;
# CAP_SLACK covers the small-mu regime, where 10*sqrt(mu) is only a few counts.
cdef double CAP_Z = 10.0
cdef long CAP_SLACK = 128


def photon_capacity(
        double T,
        double dt,
        double kD,
        double beta_g,
        double beta_r,
        double eta_g,
        double eta_r,
        double C_gg,
        double C_rr,
        double C_gr,
        double C_rg,
        ):
    """Arrival-time buffer size that ``smFRET_simulator`` cannot overflow.


    The emission rates depend on the walker's position only through the FRET
    efficiency ``E(r)``, and ``E`` is confined to ``[0, 1]``.  Since
    ``phi_g = C_gg (1-E) + C_rg E`` is a convex combination of ``C_gg`` and ``C_rg``,
    it is bounded by ``max(C_gg, C_rg)``; likewise ``phi_r <= max(C_gr, C_rr)``.  A
    channel's count over one trace is a sum of independent per-step Poissons, hence
    Poisson with a mean bounded by ``mu_max`` below -- whatever path the walker takes.
    Adding ``CAP_Z`` standard deviations plus ``CAP_SLACK`` therefore bounds the count
    with probability ``1 - 1e-15`` per trace, which is why the simulator can size its
    buffers itself instead of taking a budget from the caller.
    """
    # phi_g ranges over [min(C_gg, C_rg), max(C_gg, C_rg)] as E sweeps [0, 1], and
    # phi_r over [min(C_gr, C_rr), max(C_gr, C_rr)].  So the four corners below are
    # the extremes of the two per-step Poisson means -- all the walker can reach.
    cdef double lam[4]
    cdef double lam_min, lam_max, mu_max
    cdef int j

    if not isfinite(T) or not isfinite(dt) or dt <= 0.0 or T < 0.0:
        raise ValueError(
            f"photon_capacity: need finite T >= 0 and dt > 0, got T={T}, dt={dt}"
        )

    lam[0] = eta_g * kD * C_gg + beta_g          # photons / ms, green at E = 0
    lam[1] = eta_g * kD * C_rg + beta_g          #               green at E = 1
    lam[2] = eta_r * kD * C_gr + beta_r          #               red   at E = 0
    lam[3] = eta_r * kD * C_rr + beta_r          #               red   at E = 1

    lam_min = lam[0]
    lam_max = lam[0]
    for j in range(1, 4):
        if lam[j] < lam_min:
            lam_min = lam[j]
        if lam[j] > lam_max:
            lam_max = lam[j]

    if not isfinite(lam_min) or not isfinite(lam_max):
        raise ValueError(
            f"photon_capacity: emission rates are not finite (range [{lam_min}, "
            f"{lam_max}]); check kD, beta_g, beta_r, eta_g, eta_r and the crosstalk."
        )
    # A negative Poisson mean is undefined behaviour in gsl_ran_poisson, so refuse it
    # here rather than let the simulator walk into it at some interior FRET efficiency.
    if lam_min < 0.0:
        raise ValueError(
            f"photon_capacity: emission rate reaches {lam_min:.6g} photons/ms, but a "
            f"Poisson mean cannot be negative.  Check the signs of kD={kD}, "
            f"beta_g={beta_g}, beta_r={beta_r}, eta_g={eta_g}, eta_r={eta_r}, C_gg={C_gg}, "
            f"C_rr={C_rr}, C_gr={C_gr}, C_rg={C_rg}."
        )

    mu_max = T * lam_max
    return <long>ceil(mu_max + CAP_Z * sqrt(mu_max)) + CAP_SLACK


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def smFRET_simulator(
        double Dx,
        cnp.ndarray[double] x_knots,
        cnp.ndarray[double] y_knots,
        double R0,
        double kD,
        double beta_g,
        double beta_r,
        double eta_g,
        double eta_r,
        double C_gg,
        double C_rr,
        double C_gr,
        double C_rg,
        double T,
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
       - n_g ~ Poisson((η_g * k_D * φ_g(r) + β_g) * dt)
       - n_r ~ Poisson((η_r * k_D * φ_r(r) + β_r) * dt)
       ``beta_g``/``beta_r`` are DETECTED background rates (kHz), added outside the
       detector efficiency -- identical to the likelihood's ``mu_G = a_g φ_g + bg_g``
       with ``a_g = eta_g k_D`` (see ``photophysics.emission_rates``).
       Within each `dt` step, n_g/n_r arrival times are drawn uniformly on
       [i*dt, (i+1)*dt] — statistically exact under the homogeneous-Poisson
       assumption the simulator already commits to per step.

    The arrival-time buffers are sized internally by ``photon_capacity`` from the
    rate parameters, so there is no photon budget to supply and no way to lose a
    trace to one being too small.

    Parameters
    ----------
    dt : float
        Integration time step.
    T : float
        Total simulation time.

    Returns
    -------
    G_times, R_times : np.ndarray
        Green-/red-channel photon arrival times (variable length), or
        ``(None, None)`` if the walker left the spline domain ``[x_knots[0],
        x_knots[-1]]`` -- the one abort the caller is meant to retry.

    Raises
    ------
    RuntimeError
        If a buffer overflows.  ``photon_capacity`` is a hard upper bound, so an
        overflow means that bound is wrong; it is a bug, not a statistical event,
        and must not be silently retried.
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
    cdef int overflow_ch = 0            # 0 = green, 1 = red  (only read if it trips)
    cdef long overflow_step = -1
    cdef long overflow_need = -1

    # Output arrays: per-photon arrival times (variable length, capped at `cap`).
    # `cap` is a hard upper bound on either channel's count -- see photon_capacity.
    cdef long cap = photon_capacity(T, dt, kD, beta_g, beta_r, eta_g, eta_r,
                                    C_gg, C_rr, C_gr, C_rg)
    cdef cnp.ndarray[double] G_times = np.empty(cap, dtype=np.double)
    cdef cnp.ndarray[double] R_times = np.empty(cap, dtype=np.double)
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

        # Poisson rates for photon emission.  The backgrounds are DETECTED rates and
        # so sit outside eta -- the same convention the likelihood uses
        # (photophysics.emission_rates: mu_G = a_g phi_g + beta_g, a_g = eta_g kD).
        lambda_g = (eta_g * kD * phi_g + beta_g) * dt
        lambda_r = (eta_r * kD * phi_r + beta_r) * dt

        # Sample photon counts from Poisson distribution
        n_g = gsl_utils.gsl_ran_poisson(rng, lambda_g)
        n_r = gsl_utils.gsl_ran_poisson(rng, lambda_r)

        # Emit arrival times for each photon in this step.
        # Times are uniform on [i*dt, (i+1)*dt] (homogeneous Poisson within dt).
        if n_g > 0:
            if n_g_total + <long>n_g > cap:
                budget_exceeded = True
                overflow_ch = 0
                overflow_step = i
                overflow_need = n_g_total + <long>n_g
                break
            for k_emit in range(n_g):
                G_times[n_g_total] = (<double>i + gsl_utils.gsl_rng_uniform(rng)) * dt
                n_g_total += 1
        if n_r > 0:
            if n_r_total + <long>n_r > cap:
                budget_exceeded = True
                overflow_ch = 1
                overflow_step = i
                overflow_need = n_r_total + <long>n_r
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

    # Raise only after the frees above: an overflow means photon_capacity's bound is
    # wrong, which is a bug to surface loudly, not a trace for the caller to retry.
    if budget_exceeded:
        raise RuntimeError(
            f"smFRET_simulator: photon buffer overflow in the "
            f"{'green' if overflow_ch == 0 else 'red'} channel at step "
            f"{overflow_step} of {N} (needed {overflow_need} slots, capacity {cap}). "
            f"photon_capacity() is a hard upper bound on either channel's count, so "
            f"this is a bug in that bound, not a statistical event.  Parameters: "
            f"T={T}, dt={dt}, kD={kD}, beta_g={beta_g}, beta_r={beta_r}, eta_g={eta_g}, "
            f"eta_r={eta_r}, C_gg={C_gg}, C_rr={C_rr}, C_gr={C_gr}, C_rg={C_rg}."
        )

    if status != 0:              # walker left the spline domain -- the retryable abort
        return None, None

    return G_times[:n_g_total], R_times[:n_r_total]