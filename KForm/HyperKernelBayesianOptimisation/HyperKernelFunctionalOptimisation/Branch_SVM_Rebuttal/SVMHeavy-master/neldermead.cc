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

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>
#include "basefn.h"

//#include "nlopt-util.h"
#include "neldermead.h"
//#include "redblack.h"


//#define TIMESTAMPTYPE double
#define TIMESTAMPTYPE time_used






// Original optimiser algorithm with some extra parameters

nlopt_result nldrmd_minimize_(int n, nlopt_func f, void *f_data,
                              const double *lb, const double *ub, /* bounds */
                              double *x,/* in: initial guess, out: minimizer */
                              double *minf,
                              const double *xstep, /* initial step sizes */
                              nlopt_stopping *stop,
                              const TIMESTAMPTYPE &starttime,
                              double psi, double *scratch, double *fdiff);









typedef void (*nlopt_mfunc)(unsigned m, double *result,
                            unsigned n, const double *x,
                             double *gradient, /* NULL if not needed */
                             void *func_data);

#define NLOPT_MINF_MAX_REACHED NLOPT_STOPVAL_REACHED

int nlopt_stop_forced(const nlopt_stopping *stop);
int nlopt_stop_evals(const nlopt_stopping *stop);
int nlopt_stop_time(const TIMESTAMPTYPE &starttime, const nlopt_stopping *stop);
int nlopt_stop_ftol(const nlopt_stopping *stop, double f, double oldf);
int nlopt_stop_x(const nlopt_stopping *stop,
                 const double *x, const double *oldx);

/* re-entrant qsort */
void nlopt_qsort_r(void *base_, size_t nmemb, size_t size, void *thunk,
                   int (*compar)(void *, const void *, const void *));

typedef struct {
     unsigned m; /* dimensional of constraint: mf maps R^n -> R^m */
     nlopt_func f; /* one-dimensional constraint, requires m == 1 */
     nlopt_mfunc mf;
     //nlopt_precond pre; /* preconditioner for f (NULL if none or if mf) */
     void *f_data;
     double *tol;
} nlopt_constraint;











typedef double *rb_key; /* key type ... double* is convenient for us,
                           but of course this could be cast to anything
                           desired (although void* would look more generic) */

typedef enum { RED, BLACK } rb_color;
typedef struct rb_node_s {
     struct rb_node_s *p, *r, *l; /* parent, right, left */
     rb_key k; /* key (and data) */
     rb_color c;
} rb_node;

typedef int (*rb_compare)(rb_key k1, rb_key k2);

typedef struct {
     rb_compare compare;
     rb_node *root;
     int N; /* number of nodes */
} rb_tree;

void rb_tree_init(rb_tree *t, rb_compare compare);
void rb_tree_destroy(rb_tree *t);
void rb_tree_destroy_with_keys(rb_tree *t);
rb_node *rb_tree_insert(rb_tree *t, rb_key k);
int rb_tree_check(rb_tree *t);
rb_node *rb_tree_find(rb_tree *t, rb_key k);
rb_node *rb_tree_find_le(rb_tree *t, rb_key k);
rb_node *rb_tree_find_lt(rb_tree *t, rb_key k);
rb_node *rb_tree_find_gt(rb_tree *t, rb_key k);
rb_node *rb_tree_resort(rb_tree *t, rb_node *n);
rb_node *rb_tree_min(rb_tree *t);
rb_node *rb_tree_max(rb_tree *t);
rb_node *rb_tree_succ(rb_node *n);
rb_node *rb_tree_pred(rb_node *n);
void rb_tree_shift_keys(rb_tree *t, ptrdiff_t kshift);

/* To change a key, use rb_tree_find+resort.  Removing a node
   currently wastes memory unless you change the allocation scheme
   in redblack.c */
rb_node *rb_tree_remove(rb_tree *t, rb_node *n);
















int nlopt_isinf(double x) {
     return fabs(x) >= HUGE_VAL * 0.99
#ifdef HAVE_ISINF
          || testisinf(x)
#endif
          ;
}







inline TIMESTAMPTYPE getstarttime(void);
inline TIMESTAMPTYPE getstarttime(void)
{
    return TIMECALL;
}

int nlopt_stop_time_(const TIMESTAMPTYPE &starttime, double xmtrtime)
{
//(void) starttime;
//(void) xmtrtime;
//    return 0;

    TIMESTAMPTYPE curr_time;
    int timeout = 0;
    double *uservars[] = { NULL };
    const char *varnames[] = { NULL };
    const char *vardescr[] = { NULL };

    if ( xmtrtime > 1 )
    {
        curr_time = TIMECALL;

        if ( TIMEDIFFSEC(curr_time,starttime) > xmtrtime )
        {
            timeout = 1;
        }
    }

    if ( !timeout )
    {
        timeout = kbquitdet("Nelder-Mead optimisation",uservars,varnames,vardescr);
    }

    return timeout;
}

int nlopt_stop_time(const TIMESTAMPTYPE &starttime, const nlopt_stopping *s)
{
     return nlopt_stop_time_(starttime, s->maxtime);
}



















/* utility routines to implement the various stopping criteria */

static int relstop(double vold, double vnew, double reltol, double abstol)
{
     if (nlopt_isinf(vold)) return 0;
     return(fabs(vnew - vold) < abstol 
	    || fabs(vnew - vold) < reltol * (fabs(vnew) + fabs(vold)) * 0.5
	    || (reltol > 0 && vnew == vold)); /* catch vnew == vold == 0 */
}

int nlopt_stop_ftol(const nlopt_stopping *s, double f, double oldf)
{
     return (relstop(oldf, f, s->ftol_rel, s->ftol_abs));
}

int nlopt_stop_f(const nlopt_stopping *s, double f, double oldf)
{
     return (f <= s->minf_max || nlopt_stop_ftol(s, f, oldf));
}

int nlopt_stop_x(const nlopt_stopping *s, const double *x, const double *oldx)
{
     unsigned i;
     for (i = 0; i < s->n; ++i)
	  if (!relstop(oldx[i], x[i], s->xtol_rel, s->xtol_abs[i]))
	       return 0;
     return 1;
}

int nlopt_stop_dx(const nlopt_stopping *s, const double *x, const double *dx)
{
     unsigned i;
     for (i = 0; i < s->n; ++i)
	  if (!relstop(x[i] - dx[i], x[i], s->xtol_rel, s->xtol_abs[i]))
	       return 0;
     return 1;
}

static double sc(double x, double smin, double smax)
{
     return smin + x * (smax - smin);
}

/* some of the algorithms rescale x to a unit hypercube, so we need to
   scale back before we can compare to the tolerances */
int nlopt_stop_xs(const nlopt_stopping *s,
		  const double *xs, const double *oldxs,
		  const double *scale_min, const double *scale_max)
{
     unsigned i;
     for (i = 0; i < s->n; ++i)
	  if (relstop(sc(oldxs[i], scale_min[i], scale_max[i]), 
		      sc(xs[i], scale_min[i], scale_max[i]),
		      s->xtol_rel, s->xtol_abs[i]))
	       return 1;
     return 0;
}

int nlopt_stop_evals(const nlopt_stopping *s)
{
     return (s->maxeval > 0 && s->nevals >= s->maxeval);
}

int nlopt_stop_evalstime(const TIMESTAMPTYPE &starttime, const nlopt_stopping *stop)
{
     return nlopt_stop_evals(stop) || nlopt_stop_time(starttime,stop);
}

int nlopt_stop_forced(const nlopt_stopping *stop)
{
     return stop->force_stop && *(stop->force_stop);
}

unsigned nlopt_count_constraints(unsigned p, const nlopt_constraint *c)
{
     unsigned i, count = 0;
     for (i = 0; i < p; ++i)
	  count += c[i].m;
     return count;
}

unsigned nlopt_max_constraint_dim(unsigned p, const nlopt_constraint *c)
{
     unsigned i, max_dim = 0;
     for (i = 0; i < p; ++i)
	  if (c[i].m > max_dim)
	       max_dim = c[i].m;
     return max_dim;
}

void nlopt_eval_constraint(double *result, double *grad,
			   const nlopt_constraint *c,
			   unsigned n, const double *x)
{
     if (c->f)
	  result[0] = c->f(n, x, grad, c->f_data);
     else
	  c->mf(c->m, result, n, x, grad, c->f_data);
}






























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

//#include <math.h>
//#include <stdlib.h>
//#include <string.h>

//#include "neldermead.h"
//#include "redblack.h"

/* Nelder-Mead simplex algorithm, used as a subroutine for the Rowan's
   subplex algorithm.  Modified to handle bound constraints ala
   Richardson and Kuester (1973), as mentioned below. */

/* heuristic "strategy" constants: */
static const double alpha = 1, beta = 0.5, gamm = 2, delta = 0.5;

/* sort order in red-black tree: keys [f(x), x] are sorted by f(x) */
static int simplex_compare(double *k1, double *k2)
{
     if (*k1 < *k2) return -1;
     if (*k1 > *k2) return +1;
     return k1 - k2; /* tie-breaker */
}

/* return 1 if a and b are approximately equal relative to floating-point
   precision, 0 otherwise */
static int close(double a, double b)
{
     return (fabs(a - b) <= 1e-13 * (fabs(a) + fabs(b)));
}

/* Perform the reflection xnew = c + scale * (c - xold),
   returning 0 if xnew == c or xnew == xold (coincident points), 1 otherwise.

   The reflected point xnew is "pinned" to the lower and upper bounds
   (lb and ub), as suggested by J. A. Richardson and J. L. Kuester,
   "The complex method for constrained optimization," Commun. ACM
   16(8), 487-489 (1973).  This is probably a suboptimal way to handle
   bound constraints, but I don't know a better way.  The main danger
   with this is that the simplex might collapse into a
   lower-dimensional hyperplane; this danger can be ameliorated by
   restarting (as in subplex), however. */
static int reflectpt(int n, double *xnew, 
		     const double *c, double scale, const double *xold,
		     const double *lb, const double *ub)
{
     int equalc = 1, equalold = 1, i;
     for (i = 0; i < n; ++i) {
	  double newx = c[i] + scale * (c[i] - xold[i]);
	  if (newx < lb[i]) newx = lb[i];
	  if (newx > ub[i]) newx = ub[i];
	  equalc = equalc && close(newx, c[i]);
	  equalold = equalold && close(newx, xold[i]);
	  xnew[i] = newx;
     }
     return !(equalc || equalold);
}

#define CHECK_EVAL(xc,fc) 						  \
 stop->nevals++;							  \
 if (nlopt_stop_forced(stop)) { ret=NLOPT_FORCED_STOP; goto done; }        \
 if ((fc) <= *minf) {							  \
   *minf = (fc); memcpy(x, (xc), n * sizeof(double));			  \
   if (*minf < stop->minf_max) { ret=NLOPT_MINF_MAX_REACHED; goto done; } \
 }									  \
 if (nlopt_stop_evals(stop)) { ret=NLOPT_MAXEVAL_REACHED; goto done; }	  \
 if (nlopt_stop_time(starttime,stop)) { ret=NLOPT_MAXTIME_REACHED; goto done; }

/* Internal version of nldrmd_minimize, intended to be used as
   a subroutine for the subplex method.  Three differences compared
   to nldrmd_minimize:

   *minf should contain the value of f(x)  (so that we don't have to
   re-evaluate f at the starting x).

   if psi > 0, then it *replaces* xtol and ftol in stop with the condition
   that the simplex diameter |xl - xh| must be reduced by a factor of psi 
   ... this is for when nldrmd is used within the subplex method; for
   ordinary termination tests, set psi = 0. 

   scratch should contain an array of length >= (n+1)*(n+1) + 2*n,
   used as scratch workspace. 

   On output, *fdiff will contain the difference between the high
   and low function values of the last simplex. */
nlopt_result nldrmd_minimize_(int n, nlopt_func f, void *f_data,
			     const double *lb, const double *ub, /* bounds */
			     double *x, /* in: initial guess, out: minimizer */
			     double *minf,
			     const double *xstep, /* initial step sizes */
			     nlopt_stopping *stop,
                             const TIMESTAMPTYPE &starttime,
			     double psi, double *scratch,
			     double *fdiff)
{
     double *pts; /* (n+1) x (n+1) array of n+1 points plus function val [0] */
     double *c; /* centroid * n */
     double *xcur; /* current point */
     rb_tree t; /* red-black tree of simplex, sorted by f(x) */
     int i, j;
     double ninv = 1.0 / n;
     nlopt_result ret = NLOPT_SUCCESS;
     double init_diam = 0;

     pts = scratch;
     c = scratch + (n+1)*(n+1);
     xcur = c + n;

     rb_tree_init(&t, simplex_compare);

     *fdiff = HUGE_VAL;

     /* initialize the simplex based on the starting xstep */
     memcpy(pts+1, x, sizeof(double)*n);
     pts[0] = *minf;
     if (*minf < stop->minf_max) { ret=NLOPT_MINF_MAX_REACHED; goto done; }
     for (i = 0; i < n; ++i) {
	  double *pt = pts + (i+1)*(n+1);
	  memcpy(pt+1, x, sizeof(double)*n);
	  pt[1+i] += xstep[i];
	  if (pt[1+i] > ub[i]) {
	       if (ub[i] - x[i] > fabs(xstep[i]) * 0.1)
		    pt[1+i] = ub[i];
	       else /* ub is too close to pt, go in other direction */
		    pt[1+i] = x[i] - fabs(xstep[i]);
	  }
	  if (pt[1+i] < lb[i]) {
	       if (x[i] - lb[i] > fabs(xstep[i]) * 0.1)
		    pt[1+i] = lb[i];
	       else {/* lb is too close to pt, go in other direction */
		    pt[1+i] = x[i] + fabs(xstep[i]);
		    if (pt[1+i] > ub[i]) /* go towards further of lb, ub */
			 pt[1+i] = 0.5 * ((ub[i] - x[i] > x[i] - lb[i] ?
					   ub[i] : lb[i]) + x[i]);
	       }
	  }
	  if (close(pt[1+i], x[i])) { ret=NLOPT_FAILURE; goto done; }
	  pt[0] = f(n, pt+1, NULL, f_data);
	  CHECK_EVAL(pt+1, pt[0]);
     }

 restart:
     for (i = 0; i < n + 1; ++i)
	  if (!rb_tree_insert(&t, pts + i*(n+1))) {
	       ret = NLOPT_OUT_OF_MEMORY;
	       goto done;
	  }

     while (1) {
	  rb_node *low = rb_tree_min(&t);
	  rb_node *high = rb_tree_max(&t);
	  double fl = low->k[0], *xl = low->k + 1;
	  double fh = high->k[0], *xh = high->k + 1;
	  double fr;

	  *fdiff = fh - fl;

	  if (init_diam == 0) /* initialize diam. for psi convergence test */
	       for (i = 0; i < n; ++i) init_diam += fabs(xl[i] - xh[i]);

	  if (psi <= 0 && nlopt_stop_ftol(stop, fl, fh)) {
	       ret = NLOPT_FTOL_REACHED;
	       goto done;
	  }

	  /* compute centroid ... if we cared about the perfomance of this,
	     we could do it iteratively by updating the centroid on
	     each step, but then we would have to be more careful about
	     accumulation of rounding errors... anyway n is unlikely to
	     be very large for Nelder-Mead in practical cases */
	  memset(c, 0, sizeof(double)*n);
	  for (i = 0; i < n + 1; ++i) {
	       double *xi = pts + i*(n+1) + 1;
	       if (xi != xh)
		    for (j = 0; j < n; ++j)
			 c[j] += xi[j];
	  }
	  for (i = 0; i < n; ++i) c[i] *= ninv;

	  /* x convergence check: find xcur = max radius from centroid */
	  memset(xcur, 0, sizeof(double)*n);
	  for (i = 0; i < n + 1; ++i) {
               double *xi = pts + i*(n+1) + 1;
	       for (j = 0; j < n; ++j) {
		    double dx = fabs(xi[j] - c[j]);
		    if (dx > xcur[j]) xcur[j] = dx;
	       }
	  }
	  for (i = 0; i < n; ++i) xcur[i] += c[i];
	  if (psi > 0) {
	       double diam = 0;
	       for (i = 0; i < n; ++i) diam += fabs(xl[i] - xh[i]);
	       if (diam < psi * init_diam) {
		    ret = NLOPT_XTOL_REACHED;
		    goto done;
	       }
	  }
	  else if (nlopt_stop_x(stop, c, xcur)) {
	       ret = NLOPT_XTOL_REACHED;
	       goto done;
	  }

	  /* reflection */
	  if (!reflectpt(n, xcur, c, alpha, xh, lb, ub)) { 
	       ret=NLOPT_XTOL_REACHED; goto done; 
	  }
	  fr = f(n, xcur, NULL, f_data);
	  CHECK_EVAL(xcur, fr);

	  if (fr < fl) { /* new best point, expand simplex */
	       if (!reflectpt(n, xh, c, gamm, xh, lb, ub)) {
		    ret=NLOPT_XTOL_REACHED; goto done; 
	       }
	       fh = f(n, xh, NULL, f_data);
	       CHECK_EVAL(xh, fh);
	       if (fh >= fr) { /* expanding didn't improve */
		    fh = fr;
		    memcpy(xh, xcur, sizeof(double)*n);
	       }
	  }
	  else if (fr < rb_tree_pred(high)->k[0]) { /* accept new point */
	       memcpy(xh, xcur, sizeof(double)*n);
	       fh = fr;
	  }
	  else { /* new worst point, contract */
	       double fc;
	       if (!reflectpt(n,xcur,c, fh <= fr ? -beta : beta, xh, lb,ub)) {
		    ret=NLOPT_XTOL_REACHED; goto done; 
	       }
	       fc = f(n, xcur, NULL, f_data);
	       CHECK_EVAL(xcur, fc);
	       if (fc < fr && fc < fh) { /* successful contraction */
		    memcpy(xh, xcur, sizeof(double)*n);
		    fh = fc;
	       }
	       else { /* failed contraction, shrink simplex */
		    rb_tree_destroy(&t);
		    rb_tree_init(&t, simplex_compare);
		    for (i = 0; i < n+1; ++i) {
			 double *pt = pts + i * (n+1);
			 if (pt+1 != xl) {
			      if (!reflectpt(n,pt+1, xl,-delta,pt+1, lb,ub)) {
				   ret = NLOPT_XTOL_REACHED;
				   goto done;
			      }
			      pt[0] = f(n, pt+1, NULL, f_data);
			      CHECK_EVAL(pt+1, pt[0]);
			 }
		    }
		    goto restart;
	       }
	  }

	  high->k[0] = fh;
	  rb_tree_resort(&t, high);
     }
     
done:
     rb_tree_destroy(&t);
     return ret;
}

nlopt_result nldrmd_minimize(int n, nlopt_func f, void *f_data,
			     const double *lb, const double *ub, /* bounds */
			     double *x, /* in: initial guess, out: minimizer */
			     double *minf,
			     const double *xstep, /* initial step sizes */
			     nlopt_stopping *stop)
{
     TIMESTAMPTYPE starttime = getstarttime();

     nlopt_result ret;
     double *scratch, fdiff;

     *minf = f(n, x, NULL, f_data);
     stop->nevals++;
     if (nlopt_stop_forced(stop)) return NLOPT_FORCED_STOP;
     if (*minf < stop->minf_max) return NLOPT_MINF_MAX_REACHED;
     if (nlopt_stop_evals(stop)) return NLOPT_MAXEVAL_REACHED;
     if (nlopt_stop_time(starttime,stop)) return NLOPT_MAXTIME_REACHED;

     scratch = (double*) malloc(sizeof(double) * ((n+1)*(n+1) + 2*n));
     if (!scratch) return NLOPT_OUT_OF_MEMORY;

     ret = nldrmd_minimize_(n, f, f_data, lb, ub, x, minf, xstep, stop, starttime,
			    0.0, scratch, &fdiff);
     free(scratch);
     return ret;
}



























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

//#include "nlopt-util.h"
//#include "neldermead.h"

/* Simple replacement for the BSD qsort_r function (re-entrant sorting),
   if it is not available.

   (glibc 2.8 included a qsort_r function as well, but totally
   *%&$#-ed things up by gratuitously changing the argument order, in
   such a way as to allow code using the BSD ordering to compile but
   die a flaming death at runtime.  Damn them all to Hell, I'll just
   use my own implementation.)

   (Actually, with glibc 2.3.6 on my Intel Core Duo, my implementation
   below seems to be significantly faster than qsort.  Go figure.)
*/

#ifndef HAVE_QSORT_R_damn_it_use_my_own
/* swap size bytes between a_ and b_ */
static void swap(void *a_, void *b_, size_t size)
{
     if (a_ == b_) return;
     {
          size_t i, nlong = size / sizeof(long);
          long *a = (long *) a_, *b = (long *) b_;
          for (i = 0; i < nlong; ++i) {
               long c = a[i];
               a[i] = b[i];
               b[i] = c;
          }
	  a_ = (void*) (a + nlong);
	  b_ = (void*) (b + nlong);
     }
     {
          size_t i;
          char *a = (char *) a_, *b = (char *) b_;
          size = size % sizeof(long);
          for (i = 0; i < size; ++i) {
               char c = a[i];
               a[i] = b[i];
               b[i] = c;
          }
     }
}
#endif /* HAVE_QSORT_R */

void nlopt_qsort_r(void *base_, size_t nmemb, size_t size, void *thunk,
		   int (*compar)(void *, const void *, const void *))
{
#ifdef HAVE_QSORT_R_damn_it_use_my_own
     /* Even if we could detect glibc vs. BSD by appropriate
	macrology, there is no way to make the calls compatible
	without writing a wrapper for the compar function...screw
	this. */
     qsort_r(base_, nmemb, size, thunk, compar);
#else
     char *base = (char *) base_;
     if (nmemb < 10) { /* use O(nmemb^2) algorithm for small enough nmemb */
	  size_t i, j;
	  for (i = 0; i+1 < nmemb; ++i)
	       for (j = i+1; j < nmemb; ++j)
		    if (compar(thunk, base+i*size, base+j*size) > 0)
			 swap(base+i*size, base+j*size, size);
     }
     else {
	  size_t i, pivot, npart;
	  /* pick median of first/middle/last elements as pivot */
	  {
	       const char *a = base, *b = base + (nmemb/2)*size, 
		    *c = base + (nmemb-1)*size;
	       pivot = compar(thunk,a,b) < 0
		    ? (compar(thunk,b,c) < 0 ? nmemb/2 :
		       (compar(thunk,a,c) < 0 ? nmemb-1 : 0))
		    : (compar(thunk,a,c) < 0 ? 0 :
		       (compar(thunk,b,c) < 0 ? nmemb-1 : nmemb/2));
	  }
	  /* partition array */
	  swap(base + pivot*size, base + (nmemb-1) * size, size);
	  pivot = (nmemb - 1) * size;
	  for (i = npart = 0; i < nmemb-1; ++i)
	       if (compar(thunk, base+i*size, base+pivot) <= 0)
		    swap(base+i*size, base+(npart++)*size, size);
	  swap(base+npart*size, base+pivot, size);
	  /* recursive sort of two partitions */
	  nlopt_qsort_r(base, npart, size, thunk, compar);
	  npart++; /* don't need to sort pivot */
	  nlopt_qsort_r(base+npart*size, nmemb-npart, size, thunk, compar);
     }
#endif /* !HAVE_QSORT_R */
}
























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

//#include <math.h>
//#include <stdlib.h>
//#include <stdio.h>
//#include <string.h>

//#include "neldermead.h"

/* subplex strategy constants: */
static const double psi = 0.25, omega = 0.1;
static const int nsmin = 2, nsmax = 5;

int sbplx_verbose = 0; /* for debugging */

/* qsort_r comparison function for sorting indices into delta-x array */
static int p_compare(void *dx_, const void *i_, const void *j_)
{
     const double *dx = (const double *) dx_;
     int i = *((const int *) i_), j = *((const int *) j_);
     double dxi = fabs(dx[i]), dxj = fabs(dx[j]);
     return (dxi > dxj ? -1 : (dxi < dxj ? +1 : 0));
}

typedef struct {
     const int *p; /* subspace index permutation */
     int is; /* starting index for this subspace */
     int n; /* dimension of underlying space */
     double *x; /* current x vector */
     nlopt_func f; void *f_data; /* the "actual" underlying function */
} subspace_data;

/* wrapper around objective function for subspace optimization */
static double subspace_func(unsigned ns, const double *xs, double *grad, void *data)
{
     subspace_data *d = (subspace_data *) data;
     int i, is = d->is;
     const int *p = d->p;
     double *x = d->x;

     (void) grad; /* should always be NULL here */
     for (i = is; i < is + ((int) ns); ++i) x[p[i]] = xs[i-is];
     return d->f(d->n, x, NULL, d->f_data);
}

nlopt_result sbplx_minimize(int n, nlopt_func f, void *f_data,
			    const double *lb, const double *ub, /* bounds */
			    double *x, /* in: initial guess, out: minimizer */
			    double *minf,
			    const double *xstep0, /* initial step sizes */
			    nlopt_stopping *stop)
{
     TIMESTAMPTYPE starttime = getstarttime();

     nlopt_result ret = NLOPT_SUCCESS;
     double *xstep, *xprev, *dx, *xs, *lbs, *ubs, *xsstep, *scratch;
     int *p; /* permuted indices of x sorted by decreasing magnitude |dx| */
     int i;
     subspace_data sd;
     //double fprev;

     *minf = f(n, x, NULL, f_data);
     stop->nevals++;
     if (nlopt_stop_forced(stop)) return NLOPT_FORCED_STOP;
     if (*minf < stop->minf_max) return NLOPT_MINF_MAX_REACHED;
     if (nlopt_stop_evals(stop)) return NLOPT_MAXEVAL_REACHED;
     if (nlopt_stop_time(starttime,stop)) return NLOPT_MAXTIME_REACHED;

     xstep = (double*)malloc(sizeof(double) * (n*3 + nsmax*4
					       + (nsmax+1)*(nsmax+1)+2*nsmax));
     if (!xstep) return NLOPT_OUT_OF_MEMORY;
     xprev = xstep + n; dx = xprev + n;
     xs = dx + n; xsstep = xs + nsmax; 
     lbs = xsstep + nsmax; ubs = lbs + nsmax;
     scratch = ubs + nsmax;
     p = (int *) malloc(sizeof(int) * n);
     if (!p) { free(xstep); return NLOPT_OUT_OF_MEMORY; }

     memcpy(xstep, xstep0, n * sizeof(double));
     memset(dx, 0, n * sizeof(double));

     sd.p = p;
     sd.n = n;
     sd.x = x;
     sd.f = f;
     sd.f_data = f_data;

     while (1) {
	  double normi = 0;
	  double normdx = 0;
	  int ns, nsubs = 0;
	  int nevals = stop->nevals;
	  double fdiff, fdiff_max = 0;

	  memcpy(xprev, x, n * sizeof(double));
	  //fprev = *minf;

	  /* sort indices into the progress vector dx by decreasing
	     order of magnitude |dx| */
	  for (i = 0; i < n; ++i) p[i] = i;
	  nlopt_qsort_r(p, (size_t) n, sizeof(int), dx, p_compare);

	  /* find the subspaces, and perform nelder-mead on each one */
	  for (i = 0; i < n; ++i) normdx += fabs(dx[i]); /* L1 norm */
	  for (i = 0; i + nsmin < n; i += ns) {
	       /* find subspace starting at index i */
	       int k, nk;
	       double ns_goodness = -HUGE_VAL, norm = normi;
	       nk = i+nsmax > n ? n : i+nsmax; /* max k for this subspace */
	       for (k = i; k < i+nsmin-1; ++k) norm += fabs(dx[p[k]]);
	       ns = nsmin;
	       for (k = i+nsmin-1; k < nk; ++k) {
		    double goodness;
		    norm += fabs(dx[p[k]]);
		    /* remaining subspaces must be big enough to partition */
		    if (n-(k+1) < nsmin) continue;
		    /* maximize figure of merit defined by Rowan thesis:
		       look for sudden drops in average |dx| */
		    if (k+1 < n)
			 goodness = norm/(k+1) - (normdx-norm)/(n-(k+1));
		    else
			 goodness = normdx/n;
		    if (goodness > ns_goodness) {
			 ns_goodness = goodness;
			 ns = (k+1)-i;
		    }
	       }
	       for (k = i; k < i+ns; ++k) normi += fabs(dx[p[k]]);
	       /* do nelder-mead on subspace of dimension ns starting w/i */
	       sd.is = i;
	       for (k = i; k < i+ns; ++k) {
		    xs[k-i] = x[p[k]];
		    xsstep[k-i] = xstep[p[k]];
		    lbs[k-i] = lb[p[k]];
		    ubs[k-i] = ub[p[k]];
	       }
	       ++nsubs;
	       nevals = stop->nevals;
	       ret = nldrmd_minimize_(ns, subspace_func, &sd, lbs,ubs,xs, minf,
				      xsstep, stop, starttime, psi, scratch, &fdiff);
	       if (fdiff > fdiff_max) fdiff_max = fdiff;
	       if (sbplx_verbose)
directstream() << stop->nevals - nevals << " NM iterations for (" <<  sd.is << "," << ns << ") subspace\n";
//		    printf("%d NM iterations for (%d,%d) subspace\n",
//			   stop->nevals - nevals, sd.is, ns);
	       for (k = i; k < i+ns; ++k) x[p[k]] = xs[k-i];
	       if (ret == NLOPT_FAILURE) { ret=NLOPT_XTOL_REACHED; goto done; }
	       if (ret != NLOPT_XTOL_REACHED) goto done;
	  }
	  /* nelder-mead on last subspace */
	  ns = n - i;
	  sd.is = i;
	  for (; i < n; ++i) {
	       xs[i-sd.is] = x[p[i]];
	       xsstep[i-sd.is] = xstep[p[i]];
	       lbs[i-sd.is] = lb[p[i]];
	       ubs[i-sd.is] = ub[p[i]];
	  }
	  ++nsubs;
	  nevals = stop->nevals;
	  ret = nldrmd_minimize_(ns, subspace_func, &sd, lbs,ubs,xs, minf,
				 xsstep, stop, starttime, psi, scratch, &fdiff);
	  if (fdiff > fdiff_max) fdiff_max = fdiff;
	  if (sbplx_verbose)
directstream() << stop->nevals - nevals << " NM iterations for (" <<  sd.is << "," << ns << ") subspace\n";
//	       printf("sbplx: %d NM iterations for (%d,%d) subspace\n",
//		      stop->nevals - nevals, sd.is, ns);
	  for (i = sd.is; i < n; ++i) x[p[i]] = xs[i-sd.is];
	  if (ret == NLOPT_FAILURE) { ret=NLOPT_XTOL_REACHED; goto done; }
	  if (ret != NLOPT_XTOL_REACHED) goto done;

	  /* termination tests: */
	  if (nlopt_stop_ftol(stop, *minf, *minf + fdiff_max)) {
               ret = NLOPT_FTOL_REACHED;
               goto done;
	  }
	  if (nlopt_stop_x(stop, x, xprev)) {
	       int j;
	       /* as explained in Rowan's thesis, it is important
		  to check |xstep| as well as |x-xprev|, since if
		  the step size is too large (in early iterations),
		  the inner Nelder-Mead may not make much progress */
	       for (j = 0; j < n; ++j)
		    if (fabs(xstep[j]) * psi > stop->xtol_abs[j]
			&& fabs(xstep[j]) * psi > stop->xtol_rel * fabs(x[j]))
			 break;
	       if (j == n) {
		    ret = NLOPT_XTOL_REACHED;
		    goto done;
	       }
	  }

	  /* compute change in optimal point */
	  for (i = 0; i < n; ++i) dx[i] = x[i] - xprev[i];

	  /* setting stepsizes */
	  {
	       double scale;
	       if (nsubs == 1)
		    scale = psi;
	       else {
		    double stepnorm = 0, dxnorm = 0;
		    for (i = 0; i < n; ++i) {
			 stepnorm += fabs(xstep[i]);
			 dxnorm += fabs(dx[i]);
		    }
		    scale = dxnorm / stepnorm;
		    if (scale < omega) scale = omega;
		    if (scale > 1/omega) scale = 1/omega;
	       }
	       if (sbplx_verbose)
directstream() << "sbplx: stepsize scale factor = " << scale << "\n";
//		    printf("sbplx: stepsize scale factor = %g\n", scale);
	       for (i = 0; i < n; ++i) 
		    xstep[i] = (dx[i] == 0) ? -(xstep[i] * scale)
                         : copysign(xstep[i] * scale, dx[i]);
	  }
     }

 done:
     free(p);
     free(xstep);
     return ret;
}
/*
This directory contains Nelder-Mead and variations thereof.  

Currently, I have implemented two algorithms, described below.

The code in this directory is under the same MIT license as the rest
of my code in NLopt (see ../COPYRIGHT).

Steven G. Johnson
November 2008

-----------------------------------------------------------------------

First, (almost) the original Nelder-Mead simplex algorithm
(NLOPT_LN_NELDERMEAD), as described in:

	J. A. Nelder and R. Mead, "A simplex method for function
	minimization," The Computer Journal 7, p. 308-313 (1965).

This method is simple and has demonstrated enduring popularity,
despite the later discovery that it fails to converge at all for some
functions.  Anecdotal evidence suggests that it often performs well
even for noisy and/or discontinuous objective functions.  I would tend
to recommend the Subplex method (below) instead, however.

The main variation is that I implemented explicit support for bound
constraints, using essentially the method described in:

	J. A. Richardson and J. L. Kuester, "The complex method for
	constrained optimization," Commun. ACM 16(8), 487-489 (1973).

	implementing the method described by:

	M. J. Box, "A new method of constrained optimization and a
	comparison with other methods," Computer J. 8 (1), 42-52 (1965).

Whenever a new point would lie outside the bound constraints, Box
advocates moving it "just inside" the constraints.  I couldn't see any
advantage to using a fixed distance inside the constraints, especially
if the optimum is on the constraint, so instead I move the point
exactly onto the constraint in that case.

The danger with implementing bound constraints in this way (or by
Box's method) is that you may collapse the simplex into a
lower-dimensional subspace.  I'm not aware of a better way, however.
In any case, this collapse of the simplex is ameliorated by
restarting, such as when Nelder-Mead is used within the Subplex
algorithm below.

-----------------------------------------------------------------------

Second, I re-implemented Tom Rowan's "Subplex" algorithm.  As Rowan
expressed a preference that other implementations of his algorithm use
a different name, I called my implementation "Sbplx" (NLOPT_LN_SBPLX).
Subplex (a variant of Nelder-Mead that uses Nelder-Mead on a sequence
of subspaces) is claimed to be much more efficient and robust than the
original Nelder-Mead, while retaining the latter's facility with
discontinuous objectives, and in my experience these claims seem to be
true.  (However, I'm not aware of any proof that Subplex is globally
convergent, and may fail for some objectives like Nelder-Mead; YMMV.)

I used the description of Rowan's algorithm in his PhD thesis:

     T. Rowan, "Functional Stability Analysis of Numerical Algorithms",
     Ph.D. thesis, Department of Computer Sciences, University of Texas
     at Austin, 1990.

I would have preferred to use Rowan's original implementation, posted
by him on Netlib:

     http://www.netlib.org/opt/subplex.tgz

Unfortunately, the legality of redistributing or modifying this code
is unclear.  Rowan didn't include any license statement at all with
the original code, which makes it technically illegal to redistribute.
I contacted Rowan about getting a clear open-source/free-software
license for it, and he was very agreeable, but he said he had to think
about the specific license choice and would get back to me.
Unfortunately, a year later I still haven't heard from him, and his
old email address no longer seems to work, so I don't know how to
contact him for permission.

Since the algorithm is not too complicated, however, I just rewrote
it.  There seem to be slight differences between the behavior of my
implementation and his (probably due to different choices of initial
subspace and other slight variations, where his paper was ambiguous),
but the number of iterations to converge on my test problems seems to
be quite close (within 10% for most problems).

The only major difference between my implementation and Rowan's, as
far as I can tell, is that I implemented explicit support for bound
constraints (via the method in the Box paper as described above).
This seems to be a big improvement in the case where the optimum lies
against one of the constraints.

-----------------------------------------------------------------------

Future possibilities:

	C. J. Price, I. D. Coope, and D. Byatt, "A convergent variant
	of the Nelder-Mead algorithm," J. Optim. Theory Appl. 113 (1),
	p. 5-19 (2002).

	A. Burmen, J. Puhan, and T. Tuma, "Grid restrained Nelder-Mead
	algorithm," Computational Optim. Appl. 34(3), 359-375 (2006).

Both of these are provably convergent variations of Nelder-Mead; the
latter authors claim that theirs is superior.
*/


































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

/* simple implementation of red-black trees optimized for use with DIRECT */

//#include <stddef.h>
//#include <stdlib.h>
//#include "redblack.h"

/* it is convenient to use an explicit node for NULL nodes ... we need
   to be careful never to change this node indirectly via one of our
   pointers!  */
rb_node nil = {&nil, &nil, &nil, 0, BLACK};
#define NIL (&nil)

void rb_tree_init(rb_tree *t, rb_compare compare) {
     t->compare = compare;
     t->root = NIL;
     t->N = 0;
}

static void destroy(rb_node *n)
{
     if (n != NIL) {
	  destroy(n->l); destroy(n->r);
	  free(n);
     }
}

void rb_tree_destroy(rb_tree *t)
{
     destroy(t->root);
     t->root = NIL;
}

void rb_tree_destroy_with_keys(rb_tree *t)
{
     rb_node *n = rb_tree_min(t);
     while (n) {
	  free(n->k); n->k = NULL;
	  n = rb_tree_succ(n);
     }
     rb_tree_destroy(t);
}

static void rotate_left(rb_node *p, rb_tree *t)
{
     rb_node *n = p->r; /* must be non-NIL */
     p->r = n->l;
     n->l = p;
     if (p->p != NIL) {
	  if (p == p->p->l) p->p->l = n;
	  else p->p->r = n;
     }
     else
	  t->root = n;
     n->p = p->p;
     p->p = n;
     if (p->r != NIL) p->r->p = p;
}

static void rotate_right(rb_node *p, rb_tree *t)
{
     rb_node *n = p->l; /* must be non-NIL */
     p->l = n->r;
     n->r = p;
     if (p->p != NIL) {
	  if (p == p->p->l) p->p->l = n;
	  else p->p->r = n;
     }
     else
	  t->root = n;
     n->p = p->p;
     p->p = n;
     if (p->l != NIL) p->l->p = p;
}

static void insert_node(rb_tree *t, rb_node *n)
{
     rb_compare compare = t->compare;
     rb_key k = n->k;
     rb_node *p = t->root;
     n->c = RED;
     n->p = n->l = n->r = NIL;
     t->N++;
     if (p == NIL) {
	  t->root = n;
	  n->c = BLACK;
	  return;
     }
     /* insert (RED) node into tree */
     while (1) {
	  if (compare(k, p->k) <= 0) { /* k <= p->k */
	       if (p->l != NIL)
		    p = p->l;
	       else {
		    p->l = n;
		    n->p = p;
		    break;
	       }
	  }
	  else {
	       if (p->r != NIL)
		    p = p->r;
	       else {
		    p->r = n;
		    n->p = p;
		    break;
	       }
	  }
     }
 fixtree:
     if (n->p->c == RED) { /* red cannot have red child */
	  rb_node *u = p == p->p->l ? p->p->r : p->p->l;
	  if (u != NIL && u->c == RED) {
	       p->c = u->c = BLACK;
	       n = p->p;
	       if ((p = n->p) != NIL) {
		    n->c = RED;
		    goto fixtree;
	       }
	  }
	  else {
	       if (n == p->r && p == p->p->l) {
		    rotate_left(p, t);
		    p = n; n = n->l;
	       }
	       else if (n == p->l && p == p->p->r) {
		    rotate_right(p, t);
		    p = n; n = n->r;
	       }
	       p->c = BLACK;
	       p->p->c = RED;
	       if (n == p->l && p == p->p->l)
		    rotate_right(p->p, t);
	       else if (n == p->r && p == p->p->r)
		    rotate_left(p->p, t);
	  }
	      
     }
}

rb_node *rb_tree_insert(rb_tree *t, rb_key k)
{
     rb_node *n = (rb_node *) malloc(sizeof(rb_node));
     if (!n) return NULL;
     n->k = k;
     insert_node(t, n);
     return n;
}

static int check_node(rb_node *n, int *nblack, rb_tree *t)
{
     int nbl, nbr;
     rb_compare compare = t->compare;
     if (n == NIL) { *nblack = 0; return 1; }
     if (n->r != NIL && n->r->p != n) return 0;
     if (n->r != NIL && compare(n->r->k, n->k) < 0)
	  return 0;
     if (n->l != NIL && n->l->p != n) return 0;
     if (n->l != NIL && compare(n->l->k, n->k) > 0)
	  return 0;
     if (n->c == RED) {
	  if (n->r != NIL && n->r->c == RED) return 0;
	  if (n->l != NIL && n->l->c == RED) return 0;
     }
     if (!(check_node(n->r, &nbl, t) && check_node(n->l, &nbr, t))) 
	  return 0;
     if (nbl != nbr) return 0;
     *nblack = nbl + (n->c == BLACK);
     return 1;
}
int rb_tree_check(rb_tree *t)
{
     int nblack;
     if (nil.c != BLACK) return 0;
     if (nil.p != NIL || nil.r != NIL || nil.l != NIL) return 0;
     if (t->root == NIL) return 1;
     if (t->root->c != BLACK) return 0;
     return check_node(t->root, &nblack, t);
}

rb_node *rb_tree_find(rb_tree *t, rb_key k)
{
     rb_compare compare = t->compare;
     rb_node *p = t->root;
     while (p != NIL) {
	  int comp = compare(k, p->k);
	  if (!comp) return p;
	  p = comp <= 0 ? p->l : p->r;
     }
     return NULL;
}

/* find greatest point in subtree p that is <= k */
static rb_node *find_le(rb_node *p, rb_key k, rb_tree *t)
{
     rb_compare compare = t->compare;
     while (p != NIL) {
	  if (compare(p->k, k) <= 0) { /* p->k <= k */
	       rb_node *r = find_le(p->r, k, t);
	       if (r) return r;
	       else return p;
	  }
	  else /* p->k > k */
	       p = p->l;
     }
     return NULL; /* k < everything in subtree */
}

/* find greatest point in t <= k */
rb_node *rb_tree_find_le(rb_tree *t, rb_key k)
{
     return find_le(t->root, k, t);
}

/* find greatest point in subtree p that is < k */
static rb_node *find_lt(rb_node *p, rb_key k, rb_tree *t)
{
     rb_compare compare = t->compare;
     while (p != NIL) {
	  if (compare(p->k, k) < 0) { /* p->k < k */
	       rb_node *r = find_lt(p->r, k, t);
	       if (r) return r;
	       else return p;
	  }
	  else /* p->k >= k */
	       p = p->l;
     }
     return NULL; /* k <= everything in subtree */
}

/* find greatest point in t < k */
rb_node *rb_tree_find_lt(rb_tree *t, rb_key k)
{
     return find_lt(t->root, k, t);
}

/* find least point in subtree p that is > k */
static rb_node *find_gt(rb_node *p, rb_key k, rb_tree *t)
{
     rb_compare compare = t->compare;
     while (p != NIL) {
	  if (compare(p->k, k) > 0) { /* p->k > k */
	       rb_node *l = find_gt(p->l, k, t);
	       if (l) return l;
	       else return p;
	  }
	  else /* p->k <= k */
	       p = p->r;
     }
     return NULL; /* k >= everything in subtree */
}

/* find least point in t > k */
rb_node *rb_tree_find_gt(rb_tree *t, rb_key k)
{
     return find_gt(t->root, k, t);
}

rb_node *rb_tree_min(rb_tree *t)
{
     rb_node *n = t->root;
     while (n != NIL && n->l != NIL)
	  n = n->l;
     return(n == NIL ? NULL : n);
}

rb_node *rb_tree_max(rb_tree *t)
{
     rb_node *n = t->root;
     while (n != NIL && n->r != NIL)
	  n = n->r;
     return(n == NIL ? NULL : n);
}

rb_node *rb_tree_succ(rb_node *n)
{
     if (!n) return NULL;
     if (n->r == NIL) {
	  rb_node *prev;
	  do {
	       prev = n;
	       n = n->p;
	  } while (prev == n->r && n != NIL);
	  return n == NIL ? NULL : n;
     }
     else {
	  n = n->r;
	  while (n->l != NIL)
	       n = n->l;
	  return n;
     }
}

rb_node *rb_tree_pred(rb_node *n)
{
     if (!n) return NULL;
     if (n->l == NIL) {
	  rb_node *prev;
	  do {
	       prev = n;
	       n = n->p;
	  } while (prev == n->l && n != NIL);
	  return n == NIL ? NULL : n;
     }
     else {
	  n = n->l;
	  while (n->r != NIL)
	       n = n->r;
	  return n;
     }
}

rb_node *rb_tree_remove(rb_tree *t, rb_node *n)
{
     rb_key k = n->k;
     rb_node *m, *mp;
     if (n->l != NIL && n->r != NIL) {
	  rb_node *lmax = n->l;
	  while (lmax->r != NIL) lmax = lmax->r;
	  n->k = lmax->k;
	  n = lmax;
     }
     m = n->l != NIL ? n->l : n->r;
     if (n->p != NIL) {
	  if (n->p->r == n) n->p->r = m;
	  else n->p->l = m;
     }
     else
	  t->root = m;
     mp = n->p;
     if (m != NIL) m->p = mp;
     if (n->c == BLACK) {
	  if (m->c == RED)
	       m->c = BLACK;
	  else {
	  deleteblack:
	       if (mp != NIL) {
		    rb_node *s = m == mp->l ? mp->r : mp->l;
		    if (s->c == RED) {
			 mp->c = RED;
			 s->c = BLACK;
			 if (m == mp->l) rotate_left(mp, t);
			 else rotate_right(mp, t);
			 s = m == mp->l ? mp->r : mp->l;
		    }
		    if (mp->c == BLACK && s->c == BLACK
			&& s->l->c == BLACK && s->r->c == BLACK) {
			 if (s != NIL) s->c = RED;
			 m = mp; mp = m->p;
			 goto deleteblack;
		    }
		    else if (mp->c == RED && s->c == BLACK &&
			     s->l->c == BLACK && s->r->c == BLACK) {
			 if (s != NIL) s->c = RED;
			 mp->c = BLACK;
		    }
		    else {
			 if (m == mp->l && s->c == BLACK &&
			     s->l->c == RED && s->r->c == BLACK) {
			      s->c = RED;
			      s->l->c = BLACK;
			      rotate_right(s, t);
			      s = m == mp->l ? mp->r : mp->l;
			 }
			 else if (m == mp->r && s->c == BLACK &&
				  s->r->c == RED && s->l->c == BLACK) {
			      s->c = RED;
			      s->r->c = BLACK;
			      rotate_left(s, t);
			      s = m == mp->l ? mp->r : mp->l;
			 }
			 s->c = mp->c;
			 mp->c = BLACK;
			 if (m == mp->l) {
			      s->r->c = BLACK;
			      rotate_left(mp, t);
			 }
			 else {
			      s->l->c = BLACK;
			      rotate_right(mp, t);
			 }
		    }
	       }
	  }
     }
     t->N--;
     n->k = k; /* n may have changed during remove */
     return n; /* the node that was deleted may be different from initial n */
}

rb_node *rb_tree_resort(rb_tree *t, rb_node *n)
{
     n = rb_tree_remove(t, n);
     insert_node(t, n);
     return n;
}

/* shift all key pointers by kshift ... this is useful when the keys
   are pointers into another array, that has been resized with realloc */
static void shift_keys(rb_node *n, ptrdiff_t kshift) /* assumes n != NIL */
{
     n->k += kshift;
     if (n->l != NIL) shift_keys(n->l, kshift);
     if (n->r != NIL) shift_keys(n->r, kshift);
}
void rb_tree_shift_keys(rb_tree *t, ptrdiff_t kshift)
{
     if (t->root != NIL) shift_keys(t->root, kshift);
}
