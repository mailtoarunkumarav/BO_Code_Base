
//
// DIRect optimiser borrowed from nl_opt *FRONTEND*
// =====================================
//
// Use function direct_return_code direct_optimize(...)
//
//
// direct_objective_func f: function to be optimised
// void *f_data: passed to function to be optimised
// int dim: dimension of problem
// const double *lower_bounds: lower bound on x
// const double *upper_bounds: upper bound on x
// double *x: result x
// double *minf: set to fmin
// int max_feval: maximum function evaluations
// int max_iter: maximum iterations
// double maxtime: maximum optimisation time
// double magic_eps: epsilon value for DIRect algorithm (set 0)
// double magic_eps_abs: no idea (set 0)
// double volume_reltol: no idea (set 0)
// double sigma_reltol: no idea (set 0)
// volatile int &force_stop: set nonzero to force early termination
// double fglobal: DIRECT_UNKNOWN_FGLOBAL
// double fglobal_reltol: 0
// direct_algorithm algorithm: DIRECT_GABLONSKY (or DIRECT_ORIGINAL)
//
//
// DEBUGGING: redirect directstream
//
//
// The following are cut and pasted from the original nl_opt source (2016)
//
// ||=================================================================================||
// ||                                                                                 ||
// || ===AUTHORS===============================================================       ||
// || =========================================================================       ||
// ||                                                                                 ||
// || C conversion: Steven G. Johnson (stevenj@alum.mit.edu)                          ||
// || Original Fortran code: Joerg.M.Gablonsky (jmgablon@mailandnews.com)             ||
// ||                                                                                 ||
// ||                                                                                 ||
// || ===COPYING===============================================================       ||
// || =========================================================================       ||
// ||                                                                                 ||
// || This code is based on the DIRECT 2.0.4 Fortran code by Gablonsky et al. at      ||
// ||         http://www4.ncsu.edu/~ctk/SOFTWARE/DIRECTv204.tar.gz                    ||
// || The C version was initially converted via f2c and then cleaned up and           ||
// || reorganized by Steven G. Johnson (stevenj@alum.mit.edu), August 2007.           ||
// ||                                                                                 ||
// || ******** Copyright and license for the original Fortran DIRECT code ********    ||
// || Copyright (c) 1999, 2000, 2001 North Carolina State University                  ||
// ||                                                                                 ||
// || This program is distributed under the MIT License (see                          ||
// || http://www.opensource.org/licenses/mit-license.php):                            ||
// ||                                                                                 ||
// || Permission is hereby granted, free of charge, to any person obtaining a copy of ||
// || this software and associated documentation files (the "Software"), to deal in   ||
// || the Software without restriction, including without limitation the rights to    ||
// || use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies   ||
// || of the Software, and to permit persons to whom the Software is furnished to do  ||
// || so, subject to the following conditions:                                        ||
// ||                                                                                 ||
// || The above copyright notice and this permission notice shall be included in all  ||
// || copies or substantial portions of the Software.                                 ||
// ||                                                                                 ||
// || THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      ||
// || IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        ||
// || FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     ||
// || AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          ||
// || LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   ||
// || OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   ||
// || SOFTWARE.                                                                       ||
// ||                                                                                 ||
// ||                                                                                 ||
// || ===README================================================================       ||
// || =========================================================================       ||
// ||                                                                                 ||
// || The DIRECT algorithm (DIviding RECTangles) is a derivative-free global          ||
// || optimization algorithm invented by Jones et al.:                                ||
// ||                                                                                 ||
// ||         D. R. Jones, C. D. Perttunen, and B. E. Stuckmann,                      ||
// ||         "Lipschitzian optimization without the lipschitz constant,"             ||
// ||         J. Optimization Theory and Applications, vol. 79, p. 157 (1993).        ||
// ||                                                                                 ||
// || This is a deterministic-search algorithm based on systematic division           ||
// || of the search domain into smaller and smaller hyperrectangles.                  ||
// ||                                                                                 ||
// || The implementation is based on the 1998-2001 Fortran version by                 ||
// || J. M. Gablonsky at North Carolina State University, converted to C by           ||
// || Steven G. Johnson.  The Fortran source was downloaded from:                     ||
// ||                                                                                 ||
// ||         http://www4.ncsu.edu/~ctk/SOFTWARE/DIRECTv204.tar.gz                    ||
// ||                                                                                 ||
// || Gablonsky et al implemented a modified version of the original DIRECT           ||
// || algorithm, as described in:                                                     ||
// ||                                                                                 ||
// ||         J. M. Gablonsky and C. T. Kelley, "A locally-biased form                ||
// ||         of the DIRECT algorithm," J. Global Optimization 21 (1),                ||
// ||         p. 27-37 (2001).                                                        ||
// ||                                                                                 ||
// || Both the original Jones algorithm (NLOPT_GLOBAL_DIRECT) and the                 ||
// || Gablonsky modified version (NLOPT_GLOBAL_DIRECT_L) are implemented              ||
// || and available from the NLopt interface.  The Gablonsky version                  ||
// || makes the algorithm "more biased towards local search" so that it               ||
// || is more efficient for functions without too many local minima.                  ||
// ||                                                                                 ||
// || Also, Gablonsky et al. extended the algorithm to handle "hidden                 ||
// || constraints", i.e. arbitrary nonlinear constraints.  In NLopt, a                ||
// || hidden constraint is represented by returning NaN (or Inf, or                   ||
// || HUGE_VAL) from the objective function at any points violating the               ||
// || constraint.                                                                     ||
// ||                                                                                 ||
// || Further information on the DIRECT algorithm and Gablonsky's                     ||
// || implementation can be found in the included userguide.pdf file.                 ||
// ||                                                                                 ||
// ||=================================================================================||
//
//
// ||=================================================================================||
// ||                                                                                 ||
// || Example usage:                                                                  ||
// || #include <stdio.h>                                                              ||
// || #include <stdlib.h>                                                             ||
// ||                                                                                 ||
// || #include "direct_direct.h"                                                      ||
// ||                                                                                 ||
// || /* has two global minima at (0.09,-0.71) and (-0.09,0.71), plus                 ||
// ||    4 additional local minima */                                                 ||
// || static int cnt=0;                                                               ||
// || double tst_obj(int n, const double *xy, int *undefined_flag, void *unused)      ||
// || {                                                                               ||
// ||   double x, y, f;                                                               ||
// ||   x = xy[0];                                                                    ||
// ||   y = xy[1];                                                                    ||
// ||   f = ((x*x)*(4-2.1*(x*x)+((x*x)*(x*x))/3) + x*y + (y*y)*(-4+4*(y*y)));         ||
// ||   printf("feval:, %d, %g, %g, %g\n", ++cnt, x,y, f);                            ||
// ||   return f;                                                                     ||
// || }                                                                               ||
// ||                                                                                 ||
// || int main(int argc, char **argv)                                                 ||
// || {                                                                               ||
// ||   int n = 2;                                                                    ||
// ||   double x[2], l[2], u[2];                                                      ||
// ||   long int maxits = 0;                                                          ||
// ||   int info;                                                                     ||
// ||   double minf;                                                                  ||
// ||   int force_stop = 0;                                                           ||
// ||                                                                                 ||
// ||   maxits = argc < 2 ? 100 : atoi(argv[1]);                                      ||
// ||                                                                                 ||
// ||   l[0] = -3; l[1] = -3;                                                         ||
// ||   u[0] = 3; u[1] = 3;                                                           ||
// ||                                                                                 ||
// ||   info = direct_optimize(tst_obj, NULL, n, l, u, x, &minf,                      ||
// ||                          maxits, 500,                                           ||
// ||                          0, 0, 0, 0,                                            ||
// ||                          0.0, -1.0,                                             ||
// ||                          &force_stop,                                           ||
// ||                          DIRECT_UNKNOWN_FGLOBAL, 0,                             ||
// ||                          stdout, DIRECT_GABLONSKY);                             ||
// ||                                                                                 ||
// ||   printf("min f = %g at (%g,%g) after %d evals, return value %d\n",             ||
// ||          minf, x[0], x[1], cnt, info);                                          ||
// ||                                                                                 ||
// ||   return EXIT_SUCCESS;                                                          ||
// || }                                                                               ||
// ||                                                                                 ||
// ||=================================================================================||



#ifndef DIRECT_DIRECT_H
#define DIRECT_DIRECT_H

#include <math.h>
#include <stdio.h>
#include <iostream>
#include "basefn.h"

inline std::ostream &directstream(void);
inline std::ostream &directstream(void)
{
//    return std::cerr;
//    return errstream();
    static NullOStream devnullstream;

    return devnullstream;
}



typedef double (*direct_objective_func)(int n, const double *x,
					int *undefined_flag, 
					void *data);

typedef enum {
     DIRECT_ORIGINAL, DIRECT_GABLONSKY
} direct_algorithm;

typedef enum {
     DIRECT_INVALID_BOUNDS = -1,
     DIRECT_MAXFEVAL_TOOBIG = -2,
     DIRECT_INIT_FAILED = -3,
     DIRECT_SAMPLEPOINTS_FAILED = -4,
     DIRECT_SAMPLE_FAILED = -5,
     DIRECT_MAXFEVAL_EXCEEDED = 1,
     DIRECT_MAXITER_EXCEEDED = 2,
     DIRECT_GLOBAL_FOUND = 3,
     DIRECT_VOLTOL = 4,
     DIRECT_SIGMATOL = 5,
     DIRECT_MAXTIME_EXCEEDED = 6,

     DIRECT_OUT_OF_MEMORY = -100,
     DIRECT_INVALID_ARGS = -101,
     DIRECT_FORCED_STOP = -102
} direct_return_code;

#define DIRECT_UNKNOWN_FGLOBAL (-HUGE_VAL)
#define DIRECT_UNKNOWN_FGLOBAL_RELTOL (0.0)

direct_return_code direct_optimize(
     direct_objective_func f, void *f_data,
     int dim,
     const double *lower_bounds, const double *upper_bounds,

     double *x, double *minf, 

     int max_feval, int max_iter, 
     double maxtime,
     double magic_eps, double magic_eps_abs,
     double volume_reltol, double sigma_reltol,
     volatile int &force_stop,

     double fglobal,
     double fglobal_reltol,

     direct_algorithm algorithm);

#endif
