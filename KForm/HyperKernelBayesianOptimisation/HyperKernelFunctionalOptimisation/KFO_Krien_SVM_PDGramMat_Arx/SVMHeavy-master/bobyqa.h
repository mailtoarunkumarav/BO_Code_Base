//double nlopt_seconds(void)
//{
//   return 0; //FIXME
//}



#ifndef BOBYQA_H
#define BOBYQA_H 1

//#include "nlopt-util.h"
//#include "nlopt.h"

typedef enum {
     NLOPT_FAILURE = -1, /* generic failure code */
     NLOPT_INVALID_ARGS = -2,
     NLOPT_OUT_OF_MEMORY = -3,
     NLOPT_ROUNDOFF_LIMITED = -4,
     NLOPT_FORCED_STOP = -5,
     NLOPT_SUCCESS = 1, /* generic success code */
     NLOPT_STOPVAL_REACHED = 2,
     NLOPT_FTOL_REACHED = 3,
     NLOPT_XTOL_REACHED = 4,
     NLOPT_MAXEVAL_REACHED = 5,
     NLOPT_MAXTIME_REACHED = 6
} nlopt_result;

#define NLOPT_MINF_MAX_REACHED NLOPT_STOPVAL_REACHED

/* stopping criteria */
typedef struct {
     unsigned n; // dimensionality of problem
     double minf_max; // stop if objective drops below this (default -HUGE_VAL)
     double ftol_rel; // relative tolerance (zero to ignore)
     double ftol_abs; // absolute tolerance (zero to ignore)
     double xtol_rel; // relative tolerance in optimisation param values (0 to ignore)
     const double *xtol_abs; // absolute tolerance in optimisation param values (NULL to ignore)
     int nevals, maxeval; // max number of evaluations (0 for no limit)   nevals is a retained count of the number of evaluations done (set zero initially)
     double maxtime, start; // max training time, start appears to be unused
     int *force_stop; // set nz to force stop (appears to be unused?)
} nlopt_stopping;

typedef double (*nlopt_func)(unsigned n, const double *x,
                             double *gradient, /* NULL if not needed */
                             void *func_data);


extern nlopt_result bobyqa(int n, int npt, double *x, 
			   const double *lb, const double *ub,
			   const double *dx, 
			   nlopt_stopping *stop, double *minf,
			   nlopt_func f, void *f_data);

#endif /* BOBYQA_H */
