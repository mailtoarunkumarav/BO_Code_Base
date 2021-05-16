// Borrowed from nlopt, ripped apart and messed around to fit in 
// neldermead.h/neldermead.cc without needing extra includes.

/* Copyright (c) 2007-2014 Massachusetts Institute of Technology
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
 */

#ifndef neldermead_h
#define neldermead_h

#include <stdlib.h>
#include <stddef.h> /* for ptrdiff_t */
#include "basefn.h"


inline std::ostream &directstream(void);
inline std::ostream &directstream(void)
{
//    return std::cerr;
    static NullOStream devnullstream;

    return devnullstream;
//    return errstream();
}


// Stopping crit

typedef struct {
     unsigned n; // problem dimension
     double minf_max;
     double ftol_rel;
     double ftol_abs;
     double xtol_rel;
     const double *xtol_abs;
     int nevals; // 0
     int maxeval;
     double maxtime;
     svmvolatile int *force_stop;
} nlopt_stopping;

// Function type to be optimised

typedef double (*nlopt_func)(unsigned n, const double *x,
                             double *gradient, /* NULL if not needed */
                             void *func_data);

// Return codes

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

// Original optimiser algorithm

nlopt_result nldrmd_minimize(int n, nlopt_func f, void *f_data,
			     const double *lb, const double *ub, /* bounds */
			     double *x, /* in: initial guess, out: minimizer */
			     double *minf,
			     const double *xstep, /* initial step sizes */
			     nlopt_stopping *stop);

// Subplex optimiser

nlopt_result sbplx_minimize(int n, nlopt_func f, void *f_data,
			    const double *lb, const double *ub, /* bounds */
			    double *x, /* in: initial guess, out: minimizer */
			    double *minf,
			    const double *xstep0, /* initial step sizes */
			    nlopt_stopping *stop);


#endif
