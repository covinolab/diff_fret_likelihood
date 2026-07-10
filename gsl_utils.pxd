cdef extern from "gsl/gsl_interp.h":

    ctypedef struct gsl_interp_accel

    ctypedef struct gsl_interp_type
    ctypedef struct gsl_interp

    gsl_interp_type * gsl_interp_linear
    gsl_interp_type * gsl_interp_polynomial
    gsl_interp_type * gsl_interp_cspline
    gsl_interp_type * gsl_interp_cspline_periodic
    gsl_interp_type * gsl_interp_akima
    gsl_interp_type * gsl_interp_akima_periodic
    gsl_interp_type * gsl_interp_steffen

    gsl_interp_accel * gsl_interp_accel_alloc() nogil

    size_t gsl_interp_accel_find(gsl_interp_accel * a,  double x_array[], size_t size, double x) nogil

    int gsl_interp_accel_reset (gsl_interp_accel * a) nogil

    void gsl_interp_accel_free(gsl_interp_accel * a) nogil

    gsl_interp * gsl_interp_alloc( gsl_interp_type * T, size_t n) nogil

    int gsl_interp_init(gsl_interp * obj,  double xa[],  double ya[], size_t size) nogil

    char * gsl_interp_name( gsl_interp * interp) nogil 

    unsigned int gsl_interp_min_size( gsl_interp * interp) nogil

    int gsl_interp_eval_e( gsl_interp * obj, double xa[],  double ya[], double x, gsl_interp_accel * a, double * y) nogil

    double gsl_interp_eval( gsl_interp * obj, double xa[],  double ya[], double x, gsl_interp_accel * a) nogil

    int gsl_interp_eval_deriv_e( gsl_interp * obj,  double xa[],  double ya[], double x, gsl_interp_accel * a, double * d) nogil

    double gsl_interp_eval_deriv( gsl_interp * obj, double xa[],  double ya[], double x, gsl_interp_accel * a) nogil

    int gsl_interp_eval_deriv2_e( gsl_interp * obj, double xa[],  double ya[], double x, gsl_interp_accel * a, double * d2) nogil

    double gsl_interp_eval_deriv2( gsl_interp * obj, double xa[],  double ya[], double x, gsl_interp_accel * a) nogil

    int gsl_interp_eval_integ_e( gsl_interp * obj, double xa[],  double ya[], double a, double b, gsl_interp_accel * acc, double * result) nogil

    double gsl_interp_eval_integ( gsl_interp * obj, double xa[],  double ya[], double a, double b, gsl_interp_accel * acc) nogil

    void gsl_interp_free(gsl_interp * interp) nogil

    size_t gsl_interp_bsearch( double x_array[], double x, size_t index_lo, size_t index_hi) nogil


cdef extern from "gsl/gsl_spline.h":

    ctypedef struct gsl_spline

    gsl_spline * gsl_spline_alloc( gsl_interp_type * T, size_t size) nogil

    int gsl_spline_init(gsl_spline * spline,  double xa[],  double ya[], size_t size) nogil

    int gsl_spline_eval_e( gsl_spline * spline, double x, gsl_interp_accel * a, double * y) nogil

    double gsl_spline_eval( gsl_spline * spline, double x, gsl_interp_accel * a) nogil

    int gsl_spline_eval_deriv_e( gsl_spline * spline, double x, gsl_interp_accel * a, double * y) nogil

    double gsl_spline_eval_deriv( gsl_spline * spline, double x, gsl_interp_accel * a) nogil

    int gsl_spline_eval_deriv2_e( gsl_spline * spline, double x, gsl_interp_accel * a, double * y) nogil

    double gsl_spline_eval_deriv2( gsl_spline * spline, double x, gsl_interp_accel * a) nogil

    int gsl_spline_eval_integ_e( gsl_spline * spline, double a, double b, gsl_interp_accel * acc, double * y) nogil

    double gsl_spline_eval_integ( gsl_spline * spline, double a, double b, gsl_interp_accel * acc) nogil

    void gsl_spline_free(gsl_spline * spline) nogil


cdef extern from "gsl/gsl_rng.h":

    ctypedef struct gsl_rng_type
    ctypedef struct gsl_rng

    cdef gsl_rng_type *gsl_rng_default

    unsigned long int gsl_rng_default_seed

    gsl_rng *gsl_rng_alloc( gsl_rng_type * T) nogil
    void gsl_rng_free(gsl_rng * r) nogil
    void gsl_rng_set( gsl_rng * r, unsigned long int seed) nogil
    gsl_rng_type * gsl_rng_env_setup () nogil
    double gsl_rng_uniform( gsl_rng * r) nogil

    unsigned long int gsl_rng_uniform_int( gsl_rng * r, unsigned long int n) nogil


cdef extern from "gsl/gsl_randist.h":

    double gsl_ran_gaussian( gsl_rng * r,  double sigma) nogil
    double gsl_ran_gaussian_ziggurat( gsl_rng *r, double sigma) nogil
    
    unsigned int gsl_ran_poisson( gsl_rng * r, double mu) nogil