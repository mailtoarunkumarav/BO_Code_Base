
//
// DIRect optimiser borrowed from nl_opt
// =====================================
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


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "direct_direct.h"




//inline std::ostream &directstream(void);
//inline std::ostream &directstream(void)
//{
////    return std::cerr;
//    static NullOStream devnullstream;
//
//    return devnullstream;
//}


//#define TIMESTAMPTYPE double
#define TIMESTAMPTYPE time_used

inline int nlopt_stop_time_(TIMESTAMPTYPE &starttime, double xmtrtime);
inline int nlopt_stop_time_(TIMESTAMPTYPE &starttime, double xmtrtime)
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
        timeout = kbquitdet("DIRect optimisation",uservars,varnames,vardescr);
    }

    return timeout;
}

inline TIMESTAMPTYPE getstarttime(void);
inline TIMESTAMPTYPE getstarttime(void)
{
    return TIMECALL;
}

//#define DIRECT_UNKNOWN_FGLOBAL (-HUGE_VAL)
//#define DIRECT_UNKNOWN_FGLOBAL_RELTOL (0.0)


















typedef int integer;
typedef double doublereal;
typedef direct_objective_func fp;

//#define ASRT(c) if (!(c)) { fprintf(stderr, "DIRECT assertion failure at " __FILE__ ":%d -- " #c "\n", __LINE__); exit(EXIT_FAILURE); }
#define ASRT(c) if (!(c)) { directstream() << "DIRECT assertion failure at " << __FILE__ << ":" << __LINE__ << " -- " << #c << "\n"; throw(EXIT_FAILURE); }

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/* DIRsubrout.c */

void direct_dirheader_(
     FILE *logfile, integer *version,
     doublereal *x, integer *n, doublereal *eps, integer *maxf, integer *
     maxt, doublereal *l, doublereal *u, integer *algmethod, integer *
     maxfunc, const integer *maxdeep, doublereal *fglobal, doublereal *fglper,
     integer *ierror, doublereal *epsfix, integer *iepschange, doublereal *
     volper, doublereal *sigmaper);
void direct_dirinit_(
     doublereal *f, fp fcn, doublereal *c__,
     integer *length, integer *actdeep, integer *point, integer *anchor,
     integer *free, FILE *logfile, integer *arrayi,
     integer *maxi, integer *list2, doublereal *w, doublereal *x,
     doublereal *l, doublereal *u, doublereal *minf, integer *minpos,
     doublereal *thirds, doublereal *levels, integer *maxfunc, const integer *
     maxdeep, integer *n, integer *maxor, doublereal *fmax, integer *
     ifeasiblef, integer *iinfeasible, integer *ierror, void *fcndata,
     integer jones, TIMESTAMPTYPE &starttime, double maxtime, volatile int &force_stop);
void direct_dirinitlist_(
     integer *anchor, integer *free, integer *
     point, doublereal *f, integer *maxfunc, const integer *maxdeep);
void direct_dirpreprc_(doublereal *u, doublereal *l, integer *n, 
			      doublereal *xs1, doublereal *xs2, integer *oops);
void direct_dirchoose_(
     integer *anchor, integer *s, integer *actdeep,
     doublereal *f, doublereal *minf, doublereal epsrel, doublereal epsabs, doublereal *thirds,
     integer *maxpos, integer *length, integer *maxfunc, const integer *maxdeep,
     const integer *maxdiv, integer *n, FILE *logfile,
     integer *cheat, doublereal *kmax, integer *ifeasiblef, integer jones);
void direct_dirdoubleinsert_(
     integer *anchor, integer *s, integer *maxpos, integer *point, 
     doublereal *f, const integer *maxdeep, integer *maxfunc, 
     const integer *maxdiv, integer *ierror);
integer direct_dirgetmaxdeep_(integer *pos, integer *length, integer *maxfunc,
			      integer *n);
void direct_dirget_i__(
     integer *length, integer *pos, integer *arrayi, integer *maxi, 
     integer *n, integer *maxfunc);
void direct_dirsamplepoints_(
     doublereal *c__, integer *arrayi, 
     doublereal *delta, integer *sample, integer *start, integer *length, 
     FILE *logfile, doublereal *f, integer *free, 
     integer *maxi, integer *point, doublereal *x, doublereal *l,
     doublereal *minf, integer *minpos, doublereal *u, integer *n, 
     integer *maxfunc, const integer *maxdeep, integer *oops);
void direct_dirdivide_(
     integer *new__, integer *currentlength, 
     integer *length, integer *point, integer *arrayi, integer *sample, 
     integer *list2, doublereal *w, integer *maxi, doublereal *f, 
     integer *maxfunc, const integer *maxdeep, integer *n);
void direct_dirinsertlist_(
     integer *new__, integer *anchor, integer *point, doublereal *f, 
     integer *maxi, integer *length, integer *maxfunc, 
     const integer *maxdeep, integer *n, integer *samp, integer jones);
void direct_dirreplaceinf_(
     integer *free, integer *freeold, 
     doublereal *f, doublereal *c__, doublereal *thirds, integer *length, 
     integer *anchor, integer *point, doublereal *c1, doublereal *c2, 
     integer *maxfunc, const integer *maxdeep, integer *maxdim, integer *n, 
     FILE *logfile, doublereal *fmax, integer jones);
void direct_dirsummary_(
     FILE *logfile, doublereal *x, doublereal *l, doublereal *u, 
     integer *n, doublereal *minf, doublereal *fglobal, 
     integer *numfunc, integer *ierror);
integer direct_dirgetlevel_(
     integer *pos, integer *length, 
     integer *maxfunc, integer *n, integer jones);
void direct_dirinfcn_(
     fp fcn, doublereal *x, doublereal *c1, 
     doublereal *c2, integer *n, doublereal *f, integer *flag__, 
     void *fcn_data);

/* direct_serial.c / DIRparallel.c */
void direct_dirsamplef_(
     doublereal *c__, integer *arrayi, doublereal 
     *delta, integer *sample, integer *new__, integer *length, 
     FILE *logfile, doublereal *f, integer *free, integer *maxi, 
     integer *point, fp fcn, doublereal *x, doublereal *l, doublereal *
     minf, integer *minpos, doublereal *u, integer *n, integer *maxfunc, 
     const integer *maxdeep, integer *oops, doublereal *fmax, integer *
     ifeasiblef, integer *iinfesiblef, void *fcn_data, volatile int &force_stop);

/* DIRect.c */
void direct_direct_(
     fp fcn, doublereal *x, integer *n, doublereal *eps, doublereal epsabs,
     integer *maxf, integer *maxt, 
     TIMESTAMPTYPE &starttime, double maxtime, 
     volatile int &force_stop, doublereal *minf, doublereal *l, 
     doublereal *u, integer *algmethod, integer *ierror, FILE *logfile, 
     doublereal *fglobal, doublereal *fglper, doublereal *volper, 
     doublereal *sigmaper, void *fcn_data);
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================

//
// DIRect optimiser borrowed from nl_opt
// =====================================
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


/* DIRect-transp.f -- translated by f2c (version 20050501).
   
   f2c output hand-cleaned by SGJ (August 2007). 
*/

//#include <stdio.h>
//#include <stdlib.h>
//#include <math.h>
//#include "direct_internal.h"
//#include "direct_direct.h"

/* Common Block Declarations */

/* Table of constant values */

/* +-----------------------------------------------------------------------+ */
/* | Program       : Direct.f                                              | */
/* | Last modified : 07-16-2001                                            | */
/* | Written by    : Joerg Gablonsky (jmgablon@unity.ncsu.edu)             | */
/* |                 North Carolina State University                       | */
/* |                 Dept. of Mathematics                                  | */
/* | DIRECT is a method to solve problems of the form:                     | */
/* |              min f: Q --> R,                                          | */
/* | where f is the function to be minimized and Q is an n-dimensional     | */
/* | hyperrectangle given by the the following equation:                   | */
/* |                                                                       | */
/* |       Q={ x : l(i) <= x(i) <= u(i), i = 1,...,n }.                    | */
/* | Note: This version of DIRECT can also handle hidden constraints. By   | */
/* |       this we mean that the function may not be defined over the whole| */
/* |       hyperrectangle Q, but only over a subset, which is not given    | */
/* |       analytically.                                                   | */
/* |                                                                       | */
/* | We now give a brief outline of the algorithm:                         | */
/* |                                                                       | */
/* |   The algorithm starts with mapping the hyperrectangle Q to the       | */
/* |   n-dimensional unit hypercube. DIRECT then samples the function at   | */
/* |   the center of this hypercube and at 2n more points, 2 in each       | */
/* |   coordinate direction. Uisng these function values, DIRECT then      | */
/* |   divides the domain into hyperrectangles, each having exactly one of | */
/* |   the sampling points as its center. In each iteration, DIRECT chooses| */
/* |   some of the existing hyperrectangles to be further divided.         | */
/* |   We provide two different strategies of how to decide which          | */
/* |   hyperrectangles DIRECT divides and several different convergence    | */
/* |   criteria.                                                           | */
/* |                                                                       | */
/* |   DIRECT was designed to solve problems where the function f is       | */
/* |   Lipschitz continues. However, DIRECT has proven to be effective on  | */
/* |   more complex problems than these.                                   | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_direct_(fp fcn, doublereal *x, integer *n, doublereal *eps, doublereal epsabs, integer *maxf, integer *maxt, TIMESTAMPTYPE &starttime, double maxtime, volatile int &force_stop, doublereal *minf, doublereal *l, 
	doublereal *u, integer *algmethod, integer *ierror, FILE *logfile, 
	doublereal *fglobal, doublereal *fglper, doublereal *volper, 
	doublereal *sigmaper, void *fcn_data)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal d__1;

    /* changed by SGJ to be dynamically allocated ... would be
       even better to use realloc, below, to grow these as needed */
    integer MAXFUNC = *maxf <= 0 ? 101000 : (*maxf + 1000 + *maxf / 2);
    integer MAXDEEP = *maxt <= 0 ? MAXFUNC/5: *maxt + 1000;
    const integer MAXDIV = 5000;

    /* Local variables */
    integer increase;
    doublereal *c__ = 0	/* was [90000][64] */, *f = 0	/* 
	    was [90000][2] */;
    integer i__, j, *s = 0	/* was [3000][2] */, t;
    doublereal *w = 0;
    doublereal divfactor;
    integer ifeasiblef, iepschange, actmaxdeep;
    integer actdeep_div__, iinfesiblef;
    integer pos1, newtosample;
    integer ifree, help;
    doublereal *oldl = 0, fmax;
    integer maxi;
    doublereal kmax, *oldu = 0;
    integer oops, *list2 = 0	/* was [64][2] */, cheat;
    doublereal delta;
    integer mdeep, *point = 0, start;
    integer *anchor = 0, *length = 0	/* was [90000][64] */, *arrayi = 0;
    doublereal *levels = 0, *thirds = 0;
    integer writed;
    doublereal epsfix;
    integer oldpos, minpos, maxpos, tstart, actdeep, ifreeold, oldmaxf;
    integer numfunc, version;
    integer jones;

(void) writed;

    /* FIXME: change sizes dynamically? */
#define MY_ALLOC(p, t, n) p = (t *) malloc(sizeof(t) * (n)); \
                          if (!(p)) { *ierror = -100; goto cleanup; }

    /* Note that I've transposed c__, length, and f relative to the 
       original Fortran code.  e.g. length was length(maxfunc,n) 
       in Fortran [ or actually length(maxfunc, maxdims), but by
       using malloc I can just allocate n ], corresponding to
       length[n][maxfunc] in C, but I've changed the code to access
       it as length[maxfunc][n].  That is, the maxfunc direction
       is the discontiguous one.  This makes it easier to resize
       dynamically (by adding contiguous rows) using realloc, without
       having to move data around manually. */
    MY_ALLOC(c__, doublereal, MAXFUNC * (*n));
    MY_ALLOC(length, integer, MAXFUNC * (*n));
    MY_ALLOC(f, doublereal, MAXFUNC * 2);
    MY_ALLOC(point, integer, MAXFUNC);
    if (*maxf <= 0) *maxf = MAXFUNC - 1000;

    MY_ALLOC(s, integer, MAXDIV * 2);

    MY_ALLOC(anchor, integer, MAXDEEP + 2);
    MY_ALLOC(levels, doublereal, MAXDEEP + 1);
    MY_ALLOC(thirds, doublereal, MAXDEEP + 1);    
    if (*maxt <= 0) *maxt = MAXDEEP;

    MY_ALLOC(w, doublereal, (*n));
    MY_ALLOC(oldl, doublereal, (*n));
    MY_ALLOC(oldu, doublereal, (*n));
    MY_ALLOC(list2, integer, (*n) * 2);
    MY_ALLOC(arrayi, integer, (*n));

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE Direct                                                  | */
/* | On entry                                                              | */
/* |     fcn -- The argument containing the name of the user-supplied      | */
/* |            SUBROUTINE that returns values for the function to be      | */
/* |            minimized.                                                 | */
/* |       n -- The dimension of the problem.                              | */
/* |     eps -- Exceeding value. If eps > 0, we use the same epsilon for   | */
/* |            all iterations. If eps < 0, we use the update formula from | */
/* |            Jones:                                                     | */
/* |               eps = max(1.D-4*abs(minf),epsfix),                      | */
/* |            where epsfix = abs(eps), the absolute value of eps which is| */
/* |            passed to the function.                                    | */
/* |    maxf -- The maximum number of function evaluations.                | */
/* |    maxT -- The maximum number of iterations.                          | */
/* |            Direct stops when either the maximum number of iterations  | */
/* |            is reached or more than maxf function-evalutions were made.| */
/* |       l -- The lower bounds of the hyperbox.                          | */
/* |       u -- The upper bounds of the hyperbox.                          | */
/* |algmethod-- Choose the method, that is either use the original method  | */
/* |            as described by Jones et.al. (0) or use our modification(1)| */
/* | logfile -- File-Handle for the logfile. DIRECT expects this file to be| */
/* |            opened and closed by the user outside of DIRECT. We moved  | */
/* |            this to the outside so the user can add extra informations | */
/* |            to this file before and after the call to DIRECT.          | */
/* | fglobal -- Function value of the global optimum. If this value is not | */
/* |            known (that is, we solve a real problem, not a testproblem)| */
/* |            set this value to -1.D100 and fglper (see below) to 0.D0.  | */
/* |  fglper -- Terminate the optimization when the percent error          | */
/* |                100(f_min - fglobal)/max(1,abs(fglobal)) < fglper.     | */
/* |  volper -- Terminate the optimization when the volume of the          | */
/* |            hyperrectangle S with f(c(S)) = minf is less then volper   | */
/* |            percent of the volume of the original hyperrectangle.      | */
/* |sigmaper -- Terminate the optimization when the measure of the         | */
/* |            hyperrectangle S with f(c(S)) = minf is less then sigmaper.| */
/* |                                                                       | */
/* | User data that is passed through without being changed:               | */
/* |  fcn_data - opaque pointer to any user data                           | */
/* |                                                                       | */
/* | On return                                                             | */
/* |                                                                       | */
/* |       x -- The final point obtained in the optimization process.      | */
/* |            X should be a good approximation to the global minimum     | */
/* |            for the function within the hyper-box.                     | */
/* |                                                                       | */
/* |    minf -- The value of the function at x.                            | */
/* |  Ierror -- Error flag. If Ierror is lower 0, an error has occured. The| */
/* |            values of Ierror mean                                      | */
/* |            Fatal errors :                                             | */
/* |             -1   u(i) <= l(i) for some i.                             | */
/* |             -2   maxf is too large.                                   | */
/* |             -3   Initialization in DIRpreprc failed.                  | */
/* |             -4   Error in DIRSamplepoints, that is there was an error | */
/* |                  in the creation of the sample points.                | */
/* |             -5   Error in DIRSamplef, that is an error occured while  | */
/* |                  the function was sampled.                            | */
/* |             -6   Error in DIRDoubleInsert, that is an error occured   | */
/* |                  DIRECT tried to add all hyperrectangles with the same| */
/* |                  size and function value at the center. Either        | */
/* |                  increase maxdiv or use our modification (Jones = 1). | */
/* |            Termination values :                                       | */
/* |              1   Number of function evaluations done is larger then   | */
/* |                  maxf.                                                | */
/* |              2   Number of iterations is equal to maxT.               | */
/* |              3   The best function value found is within fglper of    | */
/* |                  the (known) global optimum, that is                  | */
/* |                   100(minf - fglobal/max(1,|fglobal|))  < fglper.     | */
/* |                  Note that this termination signal only occurs when   | */
/* |                  the global optimal value is known, that is, a test   | */
/* |                  function is optimized.                               | */
/* |              4   The volume of the hyperrectangle with minf at its    | */
/* |                  center is less than volper percent of the volume of  | */
/* |                  the original hyperrectangle.                         | */
/* |              5   The measure of the hyperrectangle with minf at its   | */
/* |                  center is less than sigmaper.                        | */
/* |                                                                       | */
/* | SUBROUTINEs used :                                                    | */
/* |                                                                       | */
/* | DIRheader, DIRInitSpecific, DIRInitList, DIRpreprc, DIRInit, DIRChoose| */
/* | DIRDoubleInsert, DIRGet_I, DIRSamplepoints, DIRSamplef, DIRDivide     | */
/* | DIRInsertList, DIRreplaceInf, DIRWritehistbox, DIRsummary, Findareas  | */
/* |                                                                       | */
/* | Functions used :                                                      | */
/* |                                                                       | */
/* | DIRgetMaxdeep, DIRgetlevel                                            | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | Parameters                                                            | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | The maximum of function evaluations allowed.                          | */
/* | The maximum dept of the algorithm.                                    | */
/* | The maximum number of divisions allowed.                              | */
/* | The maximal dimension of the problem.                                 | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | Global Variables.                                                     | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | EXTERNAL Variables.                                                   | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | User Variables.                                                       | */
/* | These can be used to pass user defined data to the function to be     | */
/* | optimized.                                                            | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | Place to define, if needed, some application-specific variables.      | */
/* | Note: You should try to use the arrays defined above for this.        | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | End of application - specific variables !                             | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | Internal variables :                                                  | */
/* |       f -- values of functions.                                       | */
/* |divfactor-- Factor used for termination with known global minimum.     | */
/* |  anchor -- anchors of lists with deepness i, -1 is anchor for list of | */
/* |            NaN - values.                                              | */
/* |       S -- List of potentially optimal points.                        | */
/* |   point -- lists                                                      | */
/* |    ifree -- first free position                                        | */
/* |       c -- midpoints of arrays                                        | */
/* |  thirds -- Precalculated values of 1/3^i.                             | */
/* |  levels -- Length of intervals.                                       | */
/* |  length -- Length of intervall (index)                                | */
/* |       t -- actual iteration                                           | */
/* |       j -- loop-variable                                              | */
/* | actdeep -- the actual minimal interval-length index                   | */
/* |  Minpos -- position of the actual minimum                             | */
/* |    file -- The filehandle for a datafile.                             | */
/* |  maxpos -- The number of intervalls, which are truncated.             | */
/* |    help -- A help variable.                                           | */
/* | numfunc -- The actual number of function evaluations.                 | */
/* |   file2 -- The filehandle for an other datafile.                      | */
/* |  ArrayI -- Array with the indexes of the sides with maximum length.   | */
/* |    maxi -- Number of directions with maximal side length.             | */
/* |    oops -- Flag which shows if anything went wrong in the             | */
/* |            initialisation.                                            | */
/* |   cheat -- Obsolete. If equal 1, we don't allow Ktilde > kmax.        | */
/* |  writed -- If writed=1, store final division to plot with Matlab.     | */
/* |   List2 -- List of indicies of intervalls, which are to be truncated. | */
/* |       i -- Another loop-variable.                                     | */
/* |actmaxdeep-- The actual maximum (minimum) of possible Interval length. | */
/* |  oldpos -- The old index of the minimum. Used to print only, if there | */
/* |            is a new minimum found.                                    | */
/* |  tstart -- The start of the outer loop.                               | */
/* |   start -- The postion of the starting point in the inner loop.       | */
/* | Newtosample -- The total number of points to sample in the inner loop.| */
/* |       w -- Array used to divide the intervalls                        | */
/* |    kmax -- Obsolete. If cheat = 1, Ktilde was not allowed to be larger| */
/* |            than kmax. If Ktilde > kmax, we set ktilde = kmax.         | */
/* |   delta -- The distance to new points from center of old hyperrec.    | */
/* |    pos1 -- Help variable used as an index.                            | */
/* | version -- Store the version number of DIRECT.                        | */
/* | oldmaxf -- Store the original function budget.                        | */
/* |increase -- Flag used to keep track if function budget was increased   | */
/* |            because no feasible point was found.                       | */
/* | ifreeold -- Keep track which index was free before. Used with          | */
/* |            SUBROUTINE DIRReplaceInf.                                  | */
/* |actdeep_div-- Keep track of the current depths for divisions.          | */
/* |    oldl -- Array used to store the original bounds of the domain.     | */
/* |    oldu -- Array used to store the original bounds of the domain.     | */
/* |  epsfix -- If eps < 0, we use Jones update formula. epsfix stores the | */
/* |            absolute value of epsilon.                                 | */
/* |iepschange-- flag iepschange to store if epsilon stays fixed or is     | */
/* |             changed.                                                  | */
/* |DIRgetMaxdeep-- Function to calculate the level of a hyperrectangle.   | */
/* |DIRgetlevel-- Function to calculate the level and stage of a hyperrec. | */
/* |    fmax -- Keep track of the maximum value of the function found.     | */
/* |Ifeasiblef-- Keep track if a feasible point has  been found so far.    | */
/* |             Ifeasiblef = 0 means a feasible point has been found,     | */
/* |             Ifeasiblef = 1 no feasible point has been found.          | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | JG 09/25/00 Version counter.                                          | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | JG 09/24/00 Add another actdeep to keep track of the current depths   | */
/* |             for divisions.                                            | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* |JG 01/13/01 Added epsfix for epsilon update. If eps < 0, we use Jones  | */
/* |            update formula. epsfix stores the absolute value of epsilon| */
/* |            then. Also added flag iepschange to store if epsilon stays | */
/* |            fixed or is changed.                                       | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 fmax is used to keep track of the maximum value found.    | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 Ifeasiblef is used to keep track if a feasible point has  | */
/* |             been found so far. Ifeasiblef = 0 means a feasible point  | */
/* |             has been found, Ifeasiblef = 1 if not.                    | */
/* | JG 03/09/01 IInfeasible is used to keep track if an infeasible point  | */
/* |             has been found. IInfeasible > 0 means a infeasible point  | */
/* |             has been found, IInfeasible = 0 if not.                   | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* |                            Start of code.                             | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
    /* Parameter adjustments */
    --u;
    --l;
    --x;

    /* Function Body */
    writed = 0;
    jones = *algmethod;
/* +-----------------------------------------------------------------------+ */
/* | Save the upper and lower bounds.                                      | */
/* +-----------------------------------------------------------------------+ */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	oldu[i__ - 1] = u[i__];
	oldl[i__ - 1] = l[i__];
/* L150: */
    }
/* +-----------------------------------------------------------------------+ */
/* | Set version.                                                          | */
/* +-----------------------------------------------------------------------+ */
    version = 204;
/* +-----------------------------------------------------------------------+ */
/* | Set parameters.                                                       | */
/* |    If cheat > 0, we do not allow \tilde{K} to be larger than kmax, and| */
/* |    set \tilde{K} to set value if necessary. Not used anymore.         | */
/* +-----------------------------------------------------------------------+ */
    cheat = 0;
    kmax = 1e10;
    mdeep = MAXDEEP;
/* +-----------------------------------------------------------------------+ */
/* | Write the header of the logfile.                                      | */
/* +-----------------------------------------------------------------------+ */
    direct_dirheader_(logfile, &version, &x[1], n, eps, maxf, maxt, &l[1], &u[1], 
	    algmethod, &MAXFUNC, &MAXDEEP, fglobal, fglper, ierror, &epsfix, &
		      iepschange, volper, sigmaper);
/* +-----------------------------------------------------------------------+ */
/* | If an error has occured while writing the header (we do some checking | */
/* | of variables there), return to the main program.                      | */
/* +-----------------------------------------------------------------------+ */
    if (*ierror < 0) {
	goto cleanup;
    }
/* +-----------------------------------------------------------------------+ */
/* | If the known global minimum is equal 0, we cannot divide by it.       | */
/* | Therefore we set it to 1. If not, we set the divisionfactor to the    | */
/* | absolute value of the global minimum.                                 | */
/* +-----------------------------------------------------------------------+ */
    if (*fglobal == 0.) {
	divfactor = 1.;
    } else {
	divfactor = fabs(*fglobal);
    }
/* +-----------------------------------------------------------------------+ */
/* | Save the budget given by the user. The variable maxf will be changed  | */
/* | if in the beginning no feasible points are found.                     | */
/* +-----------------------------------------------------------------------+ */
    oldmaxf = *maxf;
    increase = 0;
/* +-----------------------------------------------------------------------+ */
/* | Initialiase the lists.                                                | */
/* +-----------------------------------------------------------------------+ */
    direct_dirinitlist_(anchor, &ifree, point, f, &MAXFUNC, &MAXDEEP);
/* +-----------------------------------------------------------------------+ */
/* | Call the routine to initialise the mapping of x from the n-dimensional| */
/* | unit cube to the hypercube given by u and l. If an error occured,     | */
/* | give out a error message and return to the main program with the error| */
/* | flag set.                                                             | */
/* | JG 07/16/01 Changed call to remove unused data.                       | */
/* +-----------------------------------------------------------------------+ */
    direct_dirpreprc_(&u[1], &l[1], n, &l[1], &u[1], &oops);
    if (oops > 0) {
	if (logfile)
             directstream() << "WARNING: Initialization in DIRpreprc failed.\n";
	*ierror = -3;
	goto cleanup;
    }
    tstart = 2;
/* +-----------------------------------------------------------------------+ */
/* | Initialise the algorithm DIRECT.                                      | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | Added variable to keep track of the maximum value found.              | */
/* +-----------------------------------------------------------------------+ */
    direct_dirinit_(f, fcn, c__, length, &actdeep, point, anchor, &ifree,
	    logfile, arrayi, &maxi, list2, w, &x[1], &l[1], &u[1], 
	    minf, &minpos, thirds, levels, &MAXFUNC, &MAXDEEP, n, n, &
	    fmax, &ifeasiblef, &iinfesiblef, ierror, fcn_data, jones,
		    starttime, maxtime, force_stop);
/* +-----------------------------------------------------------------------+ */
/* | Added error checking.                                                 | */
/* +-----------------------------------------------------------------------+ */
    if (*ierror < 0) {
	if (*ierror == -4) {
	    if (logfile)
                 directstream() << "WARNING: Error occured in routine DIRsamplepoints.\n";
	    goto cleanup;
	}
	if (*ierror == -5) {
	    if (logfile)
                 directstream() << "WARNING: Error occured in routine DIRsamplef..\n";
	    goto cleanup;
	}
	if (*ierror == -102) goto L100;
    }
    else if (*ierror == DIRECT_MAXTIME_EXCEEDED) goto L100;
    numfunc = maxi + 1 + maxi;
    actmaxdeep = 1;
    oldpos = 0;
    tstart = 2;
/* +-----------------------------------------------------------------------+ */
/* | If no feasible point has been found, give out the iteration, the      | */
/* | number of function evaluations and a warning. Otherwise, give out     | */
/* | the iteration, the number of function evaluations done and minf.      | */
/* +-----------------------------------------------------------------------+ */
    if (ifeasiblef > 0) {
	 if (logfile)
              directstream() << "No feasible point found in " << tstart-1 << " iterations and " << numfunc << " function evaluations.\n";
    } else {
	 if (logfile)
              directstream() << tstart-1 << ", " << numfunc << ", " << *minf << ", " << fmax << "\n";
    }
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | Main loop!                                                            | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
    i__1 = *maxt;
    for (t = tstart; t <= i__1; ++t) {
/* +-----------------------------------------------------------------------+ */
/* | Choose the sample points. The indices of the sample points are stored | */
/* | in the list S.                                                        | */
/* +-----------------------------------------------------------------------+ */
	actdeep = actmaxdeep;
	direct_dirchoose_(anchor, s, &MAXDEEP, f, minf, *eps, epsabs, levels, &maxpos, length, 
		&MAXFUNC, &MAXDEEP, &MAXDIV, n, logfile, &cheat, &
		kmax, &ifeasiblef, jones);
/* +-----------------------------------------------------------------------+ */
/* | Add other hyperrectangles to S, which have the same level and the same| */
/* | function value at the center as the ones found above (that are stored | */
/* | in S). This is only done if we use the original DIRECT algorithm.     | */
/* | JG 07/16/01 Added Errorflag.                                          | */
/* +-----------------------------------------------------------------------+ */
	if (*algmethod == 0) {
	     direct_dirdoubleinsert_(anchor, s, &maxpos, point, f, &MAXDEEP, &MAXFUNC,
		     &MAXDIV, ierror);
	    if (*ierror == -6) {
		if (logfile)
                {
directstream() << "WARNING: Capacity of array S in DIRDoubleInsert reached. Increase maxdiv.\n";
directstream() << "This means that there are a lot of hyperrectangles with the same function\n";
directstream() << "value at the center. We suggest to use our modification instead (Jones = 1)\n";
                }
		goto cleanup;
	    }
	}
	oldpos = minpos;
/* +-----------------------------------------------------------------------+ */
/* | Initialise the number of sample points in this outer loop.            | */
/* +-----------------------------------------------------------------------+ */
	newtosample = 0;
	i__2 = maxpos;
	for (j = 1; j <= i__2; ++j) {
	    actdeep = s[j + MAXDIV-1];
/* +-----------------------------------------------------------------------+ */
/* | If the actual index is a point to sample, do it.                      | */
/* +-----------------------------------------------------------------------+ */
	    if (s[j - 1] > 0) {
/* +-----------------------------------------------------------------------+ */
/* | JG 09/24/00 Calculate the value delta used for sampling points.       | */
/* +-----------------------------------------------------------------------+ */
		actdeep_div__ = direct_dirgetmaxdeep_(&s[j - 1], length, &MAXFUNC, 
			n);
		delta = thirds[actdeep_div__ + 1];
		actdeep = s[j + MAXDIV-1];
/* +-----------------------------------------------------------------------+ */
/* | If the current dept of division is only one under the maximal allowed | */
/* | dept, stop the computation.                                           | */
/* +-----------------------------------------------------------------------+ */
		if (actdeep + 1 >= mdeep) {
		    if (logfile)
                         directstream() << "WARNING: Maximum number of levels reached. Increase maxdeep.\n";
		    *ierror = -6;
		    goto L100;
		}
		actmaxdeep = MAX(actdeep,actmaxdeep);
		help = s[j - 1];
		if (! (anchor[actdeep + 1] == help)) {
		    pos1 = anchor[actdeep + 1];
		    while(! (point[pos1 - 1] == help)) {
			pos1 = point[pos1 - 1];
		    }
		    point[pos1 - 1] = point[help - 1];
		} else {
		    anchor[actdeep + 1] = point[help - 1];
		}
		if (actdeep < 0) {
		    actdeep = (integer) f[(help << 1) - 2];
		}
/* +-----------------------------------------------------------------------+ */
/* | Get the Directions in which to decrease the intervall-length.         | */
/* +-----------------------------------------------------------------------+ */
		direct_dirget_i__(length, &help, arrayi, &maxi, n, &MAXFUNC);
/* +-----------------------------------------------------------------------+ */
/* | Sample the function. To do this, we first calculate the points where  | */
/* | we need to sample the function. After checking for errors, we then do | */
/* | the actual evaluation of the function, again followed by checking for | */
/* | errors.                                                               | */
/* +-----------------------------------------------------------------------+ */
		direct_dirsamplepoints_(c__, arrayi, &delta, &help, &start, length, 
			logfile, f, &ifree, &maxi, point, &x[
			1], &l[1], minf, &minpos, &u[1], n, &MAXFUNC, &
			MAXDEEP, &oops);
		if (oops > 0) {
		    if (logfile)
                         directstream() << "WARNING: Error occured in routine DIRsamplepoints.\n";
		    *ierror = -4;
		    goto cleanup;
		}
		newtosample += maxi;
/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 Added variable to keep track of the maximum value found.  | */
/* +-----------------------------------------------------------------------+ */
		direct_dirsamplef_(c__, arrayi, &delta, &help, &start, length,
			    logfile, f, &ifree, &maxi, point, fcn, &x[
			1], &l[1], minf, &minpos, &u[1], n, &MAXFUNC, &
			MAXDEEP, &oops, &fmax, &ifeasiblef, &iinfesiblef, 
				   fcn_data, force_stop);
                if (force_stop) {
		     *ierror = -102;
		     goto L100;
		}
                if (nlopt_stop_time_(starttime, maxtime)) {
                     *ierror = DIRECT_MAXTIME_EXCEEDED;
                     goto L100;
                }
		if (oops > 0) {
		    if (logfile)
                         directstream() << "WARNING: Error occured in routine DIRsamplef.\n";
		    *ierror = -5;
		    goto cleanup;
		}
/* +-----------------------------------------------------------------------+ */
/* | Divide the intervalls.                                                | */
/* +-----------------------------------------------------------------------+ */
		direct_dirdivide_(&start, &actdeep_div__, length, point, arrayi, &
			help, list2, w, &maxi, f, &MAXFUNC, &MAXDEEP, n);
/* +-----------------------------------------------------------------------+ */
/* | Insert the new intervalls into the list (sorted).                     | */
/* +-----------------------------------------------------------------------+ */
		direct_dirinsertlist_(&start, anchor, point, f, &maxi, length, &
			MAXFUNC, &MAXDEEP, n, &help, jones);
/* +-----------------------------------------------------------------------+ */
/* | Increase the number of function evaluations.                          | */
/* +-----------------------------------------------------------------------+ */
		numfunc = numfunc + maxi + maxi;
	    }
/* +-----------------------------------------------------------------------+ */
/* | End of main loop.                                                     | */
/* +-----------------------------------------------------------------------+ */
/* L20: */
	}
/* +-----------------------------------------------------------------------+ */
/* | If there is a new minimum, show the actual iteration, the number of   | */
/* | function evaluations, the minimum value of f (so far) and the position| */
/* | in the array.                                                         | */
/* +-----------------------------------------------------------------------+ */
	if (oldpos < minpos) {
	    if (logfile)
                 directstream() << t << ", " << numfunc << ", " << *minf << ", " << fmax << "\n";
	}
/* +-----------------------------------------------------------------------+ */
/* | If no feasible point has been found, give out the iteration, the      | */
/* | number of function evaluations and a warning.                         | */
/* +-----------------------------------------------------------------------+ */
	if (ifeasiblef > 0) {
	    if (logfile)
                 directstream() << "No feasible point found in " << t << " iterations and " << numfunc << " function evaluations\n";
	}
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* |                       Termination Checks                              | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 Calculate the index for the hyperrectangle at which       | */
/* |             minf is assumed. We then calculate the volume of this     | */
/* |             hyperrectangle and store it in delta. This delta can be   | */
/* |             used to stop DIRECT once the volume is below a certain    | */
/* |             percentage of the original volume. Since the original     | */
/* |             is 1 (scaled), we can stop once delta is below a certain  | */
/* |             percentage, given by volper.                              | */
/* +-----------------------------------------------------------------------+ */
	*ierror = jones;
	jones = 0;
	actdeep_div__ = direct_dirgetlevel_(&minpos, length, &MAXFUNC, n, jones);
	jones = *ierror;
/* +-----------------------------------------------------------------------+ */
/* | JG 07/16/01 Use precalculated values to calculate volume.             | */
/* +-----------------------------------------------------------------------+ */
	delta = thirds[actdeep_div__] * 100;
	if (delta <= *volper) {
	    *ierror = 4;
	    if (logfile)
                 directstream() << "DIRECT stopped: Volume of S_min is " << delta << "%% < " << *volper << "%% of the original volume.\n";
	    goto L100;
	}
/* +-----------------------------------------------------------------------+ */
/* | JG 01/23/01 Calculate the measure for the hyperrectangle at which     | */
/* |             minf is assumed. If this measure is smaller then sigmaper,| */
/* |             we stop DIRECT.                                           | */
/* +-----------------------------------------------------------------------+ */
	actdeep_div__ = direct_dirgetlevel_(&minpos, length, &MAXFUNC, n, jones);
	delta = levels[actdeep_div__];
	if (delta <= *sigmaper) {
	    *ierror = 5;
	    if (logfile)
                 directstream() << "DIRECT stopped: Measure of S_min = " << delta << " < " << *sigmaper << ".\n";
	    goto L100;
	}
/* +-----------------------------------------------------------------------+ */
/* | If the best found function value is within fglper of the (known)      | */
/* | global minimum value, terminate. This only makes sense if this optimal| */
/* | value is known, that is, in test problems.                            | */
/* +-----------------------------------------------------------------------+ */
	if ((*minf - *fglobal) * 100 / divfactor <= *fglper) {
	    *ierror = 3;
	    if (logfile)
                 directstream() << "DIRECT stopped: minf within fglper of global minimum.\n";
	    goto L100;
	}
/* +-----------------------------------------------------------------------+ */
/* | Find out if there are infeasible points which are near feasible ones. | */
/* | If this is the case, replace the function value at the center of the  | */
/* | hyper rectangle by the lowest function value of a nearby function.    | */
/* | If no infeasible points exist (IInfesiblef = 0), skip this.           | */
/* +-----------------------------------------------------------------------+ */
	if (iinfesiblef > 0) {
	     direct_dirreplaceinf_(&ifree, &ifreeold, f, c__, thirds, length, anchor, 
		    point, &u[1], &l[1], &MAXFUNC, &MAXDEEP, n, n, 
		    logfile, &fmax, jones);
	}
	ifreeold = ifree;
/* +-----------------------------------------------------------------------+ */
/* | If iepschange = 1, we use the epsilon change formula from Jones.      | */
/* +-----------------------------------------------------------------------+ */
	if (iepschange == 1) {
/* Computing MAX */
	    d__1 = fabs(*minf) * 1e-4;
	    *eps = MAX(d__1,epsfix);
	}
/* +-----------------------------------------------------------------------+ */
/* | If no feasible point has been found yet, set the maximum number of    | */
/* | function evaluations to the number of evaluations already done plus   | */
/* | the budget given by the user.                                         | */
/* | If the budget has already be increased, increase it again. If a       | */
/* | feasible point has been found, remark that and reset flag. No further | */
/* | increase is needed.                                                   | */
/* +-----------------------------------------------------------------------+ */
	if (increase == 1) {
	    *maxf = numfunc + oldmaxf;
	    if (ifeasiblef == 0) {
		if (logfile)
                     directstream() << "DIRECT found a feasible point.  The adjusted budget is now set to " << *maxf << ".\n";
		increase = 0;
	    }
	}
/* +-----------------------------------------------------------------------+ */
/* | Check if the number of function evaluations done is larger than the   | */
/* | allocated budget. If this is the case, check if a feasible point was  | */
/* | found. If this is a case, terminate. If no feasible point was found,  | */
/* | increase the budget and set flag increase.                            | */
/* +-----------------------------------------------------------------------+ */
	if (numfunc > *maxf) {
	    if (ifeasiblef == 0) {
		*ierror = 1;
		if (logfile)
                     directstream() << "DIRECT stopped: numfunc >= maxf.\n";
		goto L100;
	    } else {
		increase = 1;
		if (logfile)
                     {
directstream() << "DIRECT could not find a feasible point after " << numfunc << " function evaluations.\n";
directstream() << "DIRECT continues until a feasible point is found.\n";
                     }
		*maxf = numfunc + oldmaxf;
	    }
	}
/* L10: */
    }
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | End of main loop.                                                     | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | The algorithm stopped after maxT iterations.                          | */
/* +-----------------------------------------------------------------------+ */
    *ierror = 2;
    if (logfile)
         directstream() << "DIRECT stopped: maxT iterations.\n";

L100:
/* +-----------------------------------------------------------------------+ */
/* | Store the position of the minimum in x.                               | */
/* +-----------------------------------------------------------------------+ */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = c__[i__ + minpos * i__1 - i__1-1] * l[i__] + l[i__] * u[i__];
	u[i__] = oldu[i__ - 1];
	l[i__] = oldl[i__ - 1];
/* L50: */
    }
/* +-----------------------------------------------------------------------+ */
/* | Store the number of function evaluations in maxf.                     | */
/* +-----------------------------------------------------------------------+ */
    *maxf = numfunc;
/* +-----------------------------------------------------------------------+ */
/* | Give out a summary of the run.                                        | */
/* +-----------------------------------------------------------------------+ */
    direct_dirsummary_(logfile, &x[1], &l[1], &u[1], n, minf, fglobal, &numfunc, 
	    ierror);
/* +-----------------------------------------------------------------------+ */
/* | Format statements.                                                    | */
/* +-----------------------------------------------------------------------+ */

 cleanup:
#define MY_FREE(p) if (p) free(p)
    MY_FREE(c__);
    MY_FREE(f);
    MY_FREE(s);
    MY_FREE(w);
    MY_FREE(oldl);
    MY_FREE(oldu);
    MY_FREE(list2);
    MY_FREE(point);
    MY_FREE(anchor);
    MY_FREE(length);
    MY_FREE(arrayi);
    MY_FREE(levels);
    MY_FREE(thirds);
} /* direct_ */

// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================

//
// DIRect optimiser borrowed from nl_opt
// =====================================
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


/* DIRserial-transp.f -- translated by f2c (version 20050501).

   f2c output hand-cleaned by SGJ (August 2007).
*/

//#include "direct_internal.h"

/* +-----------------------------------------------------------------------+ */
/* | Program       : Direct.f (subfile DIRserial.f)                        | */
/* | Last modified : 04-12-2001                                            | */
/* | Written by    : Joerg Gablonsky                                       | */
/* | SUBROUTINEs, which differ depENDing on the serial or parallel version.| */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | SUBROUTINE for sampling.                                              | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirsamplef_(doublereal *c__, integer *arrayi, doublereal 
	*delta, integer *sample, integer *new__, integer *length, 
	FILE *logfile, doublereal *f, integer *free, integer *maxi, 
	integer *point, fp fcn, doublereal *x, doublereal *l, doublereal *
	minf, integer *minpos, doublereal *u, integer *n, integer *maxfunc, 
	const integer *maxdeep, integer *oops, doublereal *fmax, integer *
        ifeasiblef, integer *iinfesiblef, void *fcn_data, volatile int &force_stop)
{
/* go away compiler warnings */
(void) delta;
(void) sample;
(void) logfile;
(void) free;
(void) maxfunc;
(void) maxdeep;
(void) oops;


    /* System generated locals */
    integer length_dim1, length_offset, c_dim1, c_offset, i__1, i__2;
    doublereal d__1;

    /* Local variables */
    integer i__, j, helppoint, pos, kret;

/* +-----------------------------------------------------------------------+ */
/* | JG 07/16/01 fcn must be declared external.                            | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | JG 07/16/01 Removed fcn.                                              | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 Added variable to keep track of the maximum value found.  | */
/* |             Added variable to keep track IF feasible point was found. | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | Variables to pass user defined data to the function to be optimized.  | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | Set the pointer to the first function to be evaluated,                | */
/* | store this position also in helppoint.                                | */
/* +-----------------------------------------------------------------------+ */
    /* Parameter adjustments */
    --u;
    --l;
    --x;
    --arrayi;
    --point;
    f -= 3;
    length_dim1 = *n;
    length_offset = 1 + length_dim1;
    length -= length_offset;
    c_dim1 = *n;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;

    /* Function Body */
    pos = *new__;
    helppoint = pos;
/* +-----------------------------------------------------------------------+ */
/* | Iterate over all points, where the function should be                 | */
/* | evaluated.                                                            | */
/* +-----------------------------------------------------------------------+ */
    i__1 = *maxi + *maxi;
    for (j = 1; j <= i__1; ++j) {
/* +-----------------------------------------------------------------------+ */
/* | Copy the position into the helparrayy x.                              | */
/* +-----------------------------------------------------------------------+ */
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    x[i__] = c__[i__ + pos * c_dim1];
/* L60: */
	}
/* +-----------------------------------------------------------------------+ */
/* | Call the function.                                                    | */
/* +-----------------------------------------------------------------------+ */
        if (force_stop)  /* skip eval after forced stop */
	     f[(pos << 1) + 1] = *fmax;
	else
	     direct_dirinfcn_(fcn, &x[1], &l[1], &u[1], n, &f[(pos << 1) + 1], 
			      &kret, fcn_data);
        if (force_stop)
	     kret = -1; /* mark as invalid point */
/* +-----------------------------------------------------------------------+ */
/* | Remember IF an infeasible point has been found.                       | */
/* +-----------------------------------------------------------------------+ */
	*iinfesiblef = MAX(*iinfesiblef,kret);
	if (kret == 0) {
/* +-----------------------------------------------------------------------+ */
/* | IF the function evaluation was O.K., set the flag in                  | */
/* | f(2,pos). Also mark that a feasible point has been found.             | */
/* +-----------------------------------------------------------------------+ */
	    f[(pos << 1) + 2] = 0.;
	    *ifeasiblef = 0;
/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 Added variable to keep track of the maximum value found.  | */
/* +-----------------------------------------------------------------------+ */
/* Computing MAX */
	    d__1 = f[(pos << 1) + 1];
	    *fmax = MAX(d__1,*fmax);
	}
	if (kret >= 1) {
/* +-----------------------------------------------------------------------+ */
/* |  IF the function could not be evaluated at the given point,            | */
/* | set flag to mark this (f(2,pos) and store the maximum                 | */
/* | box-sidelength in f(1,pos).                                           | */
/* +-----------------------------------------------------------------------+ */
	    f[(pos << 1) + 2] = 2.;
	    f[(pos << 1) + 1] = *fmax;
	}
/* +-----------------------------------------------------------------------+ */
/* |  IF the function could not be evaluated due to a failure in            | */
/* | the setup, mark this.                                                 | */
/* +-----------------------------------------------------------------------+ */
	if (kret == -1) {
	    f[(pos << 1) + 2] = -1.;
	}
/* +-----------------------------------------------------------------------+ */
/* | Set the position to the next point, at which the function             | */
/* | should e evaluated.                                                   | */
/* +-----------------------------------------------------------------------+ */
	pos = point[pos];
/* L40: */
    }
    pos = helppoint;
/* +-----------------------------------------------------------------------+ */
/* | Iterate over all evaluated points and see, IF the minimal             | */
/* | value of the function has changed.  IF this has happEND,               | */
/* | store the minimal value and its position in the array.                | */
/* | Attention: Only valid values are checked!!                           | */
/* +-----------------------------------------------------------------------+ */
    i__1 = *maxi + *maxi;
    for (j = 1; j <= i__1; ++j) {
	if (f[(pos << 1) + 1] < *minf && f[(pos << 1) + 2] == 0.) {
	    *minf = f[(pos << 1) + 1];
	    *minpos = pos;
	}
	pos = point[pos];
/* L50: */
    }
} /* dirsamplef_ */
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================

//
// DIRect optimiser borrowed from nl_opt
// =====================================
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


/* DIRsubrout.f -- translated by f2c (version 20050501).

   f2c output hand-cleaned by SGJ (August 2007).
*/

//#include <math.h>
//#include "direct_internal.h"

/* Table of constant values */

static integer c__1 = 1;
static integer c__32 = 32;
static integer c__0 = 0;

/* +-----------------------------------------------------------------------+ */
/* | INTEGER Function DIRGetlevel                                          | */
/* | Returns the level of the hyperrectangle. Depending on the value of the| */
/* | global variable JONES. IF JONES equals 0, the level is given by       | */
/* |               kN + p, where the rectangle has p sides with a length of| */
/* |             1/3^(k+1), and N-p sides with a length of 1/3^k.          | */
/* | If JONES equals 1, the level is the power of 1/3 of the length of the | */
/* | longest side hyperrectangle.                                          | */
/* |                                                                       | */
/* | On Return :                                                           | */
/* |    the maximal length                                                 | */
/* |                                                                       | */
/* | pos     -- the position of the midpoint in the array length           | */
/* | length  -- the array with the dimensions                              | */
/* | maxfunc -- the leading dimension of length                            | */
/* | n	   -- the dimension of the problem                                  | */
/* |                                                                       | */
/* +-----------------------------------------------------------------------+ */
integer direct_dirgetlevel_(integer *pos, integer *length, integer *maxfunc, integer 
	*n, integer jones)
{
(void) maxfunc;
    /* System generated locals */
    integer length_dim1, length_offset, ret_val, i__1;

    /* Local variables */
    integer i__, k, p, help;

/* JG 09/15/00 Added variable JONES (see above) */
    /* Parameter adjustments */
    length_dim1 = *n;
    length_offset = 1 + length_dim1;
    length -= length_offset;

    /* Function Body */
    if (jones == 0) {
	help = length[*pos * length_dim1 + 1];
	k = help;
	p = 1;
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    if (length[i__ + *pos * length_dim1] < k) {
		k = length[i__ + *pos * length_dim1];
	    }
	    if (length[i__ + *pos * length_dim1] == help) {
		++p;
	    }
/* L100: */
	}
	if (k == help) {
	    ret_val = k * *n + *n - p;
	} else {
	    ret_val = k * *n + p;
	}
    } else {
	help = length[*pos * length_dim1 + 1];
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    if (length[i__ + *pos * length_dim1] < help) {
		help = length[i__ + *pos * length_dim1];
	    }
/* L10: */
	}
	ret_val = help;
    }
    return ret_val;
} /* dirgetlevel_ */

/* +-----------------------------------------------------------------------+ */
/* | Program       : Direct.f (subfile DIRsubrout.f)                       | */
/* | Last modified : 07-16-2001                                            | */
/* | Written by    : Joerg Gablonsky                                       | */
/* | Subroutines used by the algorithm DIRECT.                             | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRChoose                                               | */
/* |    Decide, which is the next sampling point.                          | */
/* |    Changed 09/25/00 JG                                                | */
/* |         Added maxdiv to call and changed S to size maxdiv.            | */
/* |    Changed 01/22/01 JG                                                | */
/* |         Added Ifeasiblef to call to keep track if a feasible point has| */
/* |         been found.                                                   | */
/* |    Changed 07/16/01 JG                                                | */
/* |         Changed if statement to prevent run-time errors.              |                                  
                 | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirchoose_(integer *anchor, integer *s, integer *actdeep,
	 doublereal *f, doublereal *minf, doublereal epsrel, doublereal epsabs, doublereal *thirds,
	 integer *maxpos, integer *length, integer *maxfunc, const integer *maxdeep,
	 const integer *maxdiv, integer *n, FILE *logfile,
	integer *cheat, doublereal *kmax, integer *ifeasiblef, integer jones)
{
    /* System generated locals */
    integer s_dim1, s_offset, length_dim1, length_offset, i__1;

    /* Local variables */
    integer i__, j, k;
    doublereal helplower;
    integer i___, j___;
    doublereal helpgreater;
    integer novaluedeep = 0;
    doublereal help2;
    integer novalue;

    /* Parameter adjustments */
    f -= 3;
    ++anchor;
    s_dim1 = *maxdiv;
    s_offset = 1 + s_dim1;
    s -= s_offset;
    length_dim1 = *n;
    length_offset = 1 + length_dim1;
    length -= length_offset;

    /* Function Body */
    helplower = HUGE_VAL;
    helpgreater = 0.;
    k = 1;
    if (*ifeasiblef >= 1) {
	i__1 = *actdeep;
	for (j = 0; j <= i__1; ++j) {
	    if (anchor[j] > 0) {
		s[k + s_dim1] = anchor[j];
		s[k + (s_dim1 << 1)] = direct_dirgetlevel_(&s[k + s_dim1], &length[
			length_offset], maxfunc, n, jones);
		goto L12;
	    }
/* L1001: */
	}
L12:
	++k;
	*maxpos = 1;
	return;
    } else {
	i__1 = *actdeep;
	for (j = 0; j <= i__1; ++j) {
	    if (anchor[j] > 0) {
		s[k + s_dim1] = anchor[j];
		s[k + (s_dim1 << 1)] = direct_dirgetlevel_(&s[k + s_dim1], &length[
			length_offset], maxfunc, n, jones);
		++k;
	    }
/* L10: */
	}
    }
    novalue = 0;
    if (anchor[-1] > 0) {
	novalue = anchor[-1];
	novaluedeep = direct_dirgetlevel_(&novalue, &length[length_offset], maxfunc, 
		n, jones);
    }
    *maxpos = k - 1;
    i__1 = *maxdeep;
    for (j = k - 1; j <= i__1; ++j) {
	s[k + s_dim1] = 0;
/* L11: */
    }
    for (j = *maxpos; j >= 1; --j) {
	helplower = HUGE_VAL;
	helpgreater = 0.;
	j___ = s[j + s_dim1];
	i__1 = j - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i___ = s[i__ + s_dim1];
/* +-----------------------------------------------------------------------+ */
/* | JG 07/16/01 Changed IF statement into two to prevent run-time errors  | */
/* |             which could occur if the compiler checks the second       | */
/* |             expression in an .AND. statement although the first       | */
/* |             statement is already not true.                            | */
/* +-----------------------------------------------------------------------+ */
	    if (i___ > 0 && ! (i__ == j)) {
		if (f[(i___ << 1) + 2] <= 1.) {
		    help2 = thirds[s[i__ + (s_dim1 << 1)]] - thirds[s[j + (
			    s_dim1 << 1)]];
		    help2 = (f[(i___ << 1) + 1] - f[(j___ << 1) + 1]) / help2;
		    if (help2 <= 0.) {
			 if (logfile)
                              directstream() << "thirds > 0, help2 <= 0\n";
			goto L60;
		    }
		    if (help2 < helplower) {
			 if (logfile)
                              directstream() << "helplower = " << help2 << "\n";
			helplower = help2;
		    }
		}
	    }
/* L30: */
	}
	i__1 = *maxpos;
	for (i__ = j + 1; i__ <= i__1; ++i__) {
	    i___ = s[i__ + s_dim1];
/* +-----------------------------------------------------------------------+ */
/* | JG 07/16/01 Changed IF statement into two to prevent run-time errors  | */
/* |             which could occur if the compiler checks the second       | */
/* |             expression in an .AND. statement although the first       | */
/* |             statement is already not true.                            | */
/* +-----------------------------------------------------------------------+ */
	    if (i___ > 0 && ! (i__ == j)) {
		if (f[(i___ << 1) + 2] <= 1.) {
		    help2 = thirds[s[i__ + (s_dim1 << 1)]] - thirds[s[j + (
			    s_dim1 << 1)]];
		    help2 = (f[(i___ << 1) + 1] - f[(j___ << 1) + 1]) / help2;
		    if (help2 <= 0.) {
			if (logfile)
                             directstream() << "thirds < 0, help2 <= 0\n";
			goto L60;
		    }
		    if (help2 > helpgreater) {
			if (logfile)
                              directstream() << "helpgreater = " << help2 << "\n";
			helpgreater = help2;
		    }
		}
	    }
/* L31: */
	}
	if (helpgreater <= helplower) {
	    if (*cheat == 1 && helplower > *kmax) {
		helplower = *kmax;
	    }
	    if (f[(j___ << 1) + 1] - helplower * thirds[s[j + (s_dim1 << 1)]] >
		     MIN(*minf - epsrel * fabs(*minf), 
			 *minf - epsabs)) {
		if (logfile)
                     directstream() << "> minf - epslminfl\n";
		goto L60;
	    }
	} else {
	    if (logfile)
                 directstream() << "helpgreater > helplower: " << helpgreater << "  " << helplower << "  " << helpgreater-helplower << "\n";
	    goto L60;
	}
	goto L40;
L60:
	s[j + s_dim1] = 0;
L40:
	;
    }
    if (novalue > 0) {
	++(*maxpos);
	s[*maxpos + s_dim1] = novalue;
	s[*maxpos + (s_dim1 << 1)] = novaluedeep;
    }
} /* dirchoose_ */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRDoubleInsert                                         | */
/* |      Routine to make sure that if there are several potential optimal | */
/* |      hyperrectangles of the same level (i.e. hyperrectangles that have| */
/* |      the same level and the same function value at the center), all of| */
/* |      them are divided. This is the way as originally described in     | */
/* |      Jones et.al.                                                     | */
/* | JG 07/16/01 Added errorflag to calling sequence. We check if more     | */
/* |             we reach the capacity of the array S. If this happens, we | */
/* |             return to the main program with an error.                 | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirdoubleinsert_(integer *anchor, integer *s, integer *
	maxpos, integer *point, doublereal *f, const integer *maxdeep, integer *
	maxfunc, const integer *maxdiv, integer *ierror)
{
(void) maxdeep;
(void) maxfunc;

    /* System generated locals */
    integer s_dim1, s_offset, i__1;

    /* Local variables */
    integer i__, oldmaxpos, pos, help, iflag, actdeep;

/* +-----------------------------------------------------------------------+ */
/* | JG 07/16/01 Added flag to prevent run time-errors on some systems.    | */
/* +-----------------------------------------------------------------------+ */
    /* Parameter adjustments */
    ++anchor;
    f -= 3;
    --point;
    s_dim1 = *maxdiv;
    s_offset = 1 + s_dim1;
    s -= s_offset;

    /* Function Body */
    oldmaxpos = *maxpos;
    i__1 = oldmaxpos;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (s[i__ + s_dim1] > 0) {
	    actdeep = s[i__ + (s_dim1 << 1)];
	    help = anchor[actdeep];
	    pos = point[help];
	    iflag = 0;
/* +-----------------------------------------------------------------------+ */
/* | JG 07/16/01 Added flag to prevent run time-errors on some systems. On | */
/* |             some systems the second conditions in an AND statement is | */
/* |             evaluated even if the first one is already not true.      | */
/* +-----------------------------------------------------------------------+ */
	    while(pos > 0 && iflag == 0) {
		if (f[(pos << 1) + 1] - f[(help << 1) + 1] <= 1e-13) {
		    if (*maxpos < *maxdiv) {
			++(*maxpos);
			s[*maxpos + s_dim1] = pos;
			s[*maxpos + (s_dim1 << 1)] = actdeep;
			pos = point[pos];
		    } else {
/* +-----------------------------------------------------------------------+ */
/* | JG 07/16/01 Maximum number of elements possible in S has been reached!| */
/* +-----------------------------------------------------------------------+ */
			*ierror = -6;
			return;
		    }
		} else {
		    iflag = 1;
		}
	    }
	}
/* L10: */
    }
} /* dirdoubleinsert_ */

/* +-----------------------------------------------------------------------+ */
/* | INTEGER Function GetmaxDeep                                           | */
/* | function to get the maximal length (1/length) of the n-dimensional    | */
/* | rectangle with midpoint pos.                                          | */
/* |                                                                       | */
/* | On Return :                                                           | */
/* |    the maximal length                                                 | */
/* |                                                                       | */
/* | pos     -- the position of the midpoint in the array length           | */
/* | length  -- the array with the dimensions                              | */
/* | maxfunc -- the leading dimension of length                            | */
/* | n	   -- the dimension of the problem                                  | */
/* |                                                                       | */
/* +-----------------------------------------------------------------------+ */
integer direct_dirgetmaxdeep_(integer *pos, integer *length, integer *maxfunc, 
	integer *n)
{
(void) maxfunc;

    /* System generated locals */
    integer length_dim1, length_offset, i__1, i__2, i__3;

    /* Local variables */
    integer i__, help;

    /* Parameter adjustments */
    length_dim1 = *n;
    length_offset = 1 + length_dim1;
    length -= length_offset;

    /* Function Body */
    help = length[*pos * length_dim1 + 1];
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
/* Computing MIN */
	i__2 = help, i__3 = length[i__ + *pos * length_dim1];
	help = MIN(i__2,i__3);
/* L10: */
    }
    return help;
} /* dirgetmaxdeep_ */

static integer isinbox_(doublereal *x, doublereal *a, doublereal *b, integer *n, 
	integer *lmaxdim)
{
(void) lmaxdim;

    /* System generated locals */
    integer ret_val, i__1;

    /* Local variables */
    integer outofbox, i__;

    /* Parameter adjustments */
    --b;
    --a;
    --x;

    /* Function Body */
    outofbox = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (a[i__] > x[i__] || b[i__] < x[i__]) {
	    outofbox = 0;
	    goto L1010;
	}
/* L1000: */
    }
L1010:
    ret_val = outofbox;
    return ret_val;
} /* isinbox_ */

/* +-----------------------------------------------------------------------+ */
/* | JG Added 09/25/00                                                     | */
/* |                                                                       | */
/* |                       SUBROUTINE DIRResortlist                        | */
/* |                                                                       | */
/* | Resort the list so that the infeasible point is in the list with the  | */
/* | replaced value.                                                       | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ static void dirresortlist_(integer *replace, integer *anchor, 
	doublereal *f, integer *point, integer *length, integer *n, integer *
	maxfunc, integer *maxdim, const integer *maxdeep, FILE *logfile,
					    integer jones)
{
(void) maxdim;
(void) maxdeep;

    /* System generated locals */
    integer length_dim1, length_offset, i__1;

    /* Local variables */
    integer i__, l, pos;
    integer start;

/* +-----------------------------------------------------------------------+ */
/* | Get the length of the hyper rectangle with infeasible mid point and   | */
/* | Index of the corresponding list.                                      | */
/* +-----------------------------------------------------------------------+ */
/* JG 09/25/00 Replaced with DIRgetlevel */
/*      l = DIRgetmaxDeep(replace,length,maxfunc,n) */
    /* Parameter adjustments */
    --point;
    f -= 3;
    length_dim1 = *n;
    length_offset = 1 + length_dim1;
    length -= length_offset;
    ++anchor;

    /* Function Body */
    l = direct_dirgetlevel_(replace, &length[length_offset], maxfunc, n, jones);
    start = anchor[l];
/* +-----------------------------------------------------------------------+ */
/* | If the hyper rectangle with infeasibel midpoint is already the start  | */
/* | of the list, give out message, nothing to do.                         | */
/* +-----------------------------------------------------------------------+ */
    if (*replace == start) {
/*         write(logfile,*) 'No resorting of list necessarry, since new ', */
/*     + 'point is already anchor of list .',l */
    } else {
/* +-----------------------------------------------------------------------+ */
/* | Take the hyper rectangle with infeasible midpoint out of the list.    | */
/* +-----------------------------------------------------------------------+ */
	pos = start;
	i__1 = *maxfunc;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (point[pos] == *replace) {
		point[pos] = point[*replace];
		goto L20;
	    } else {
		pos = point[pos];
	    }
	    if (pos == 0) {
		if (logfile)
                     directstream() << "Error in DIRREsortlist: We went through the whole list\nand could not find the point to replace!!\n";
		goto L20;
	    }
/* L10: */
	}
/* +-----------------------------------------------------------------------+ */
/* | If the anchor of the list has a higher value than the value of a      | */
/* | nearby point, put the infeasible point at the beginning of the list.  | */
/* +-----------------------------------------------------------------------+ */
L20:
	if (f[(start << 1) + 1] > f[(*replace << 1) + 1]) {
	    anchor[l] = *replace;
	    point[*replace] = start;
/*            write(logfile,*) 'Point is replacing current anchor for ' */
/*     +             , 'this list ',l,replace,start */
	} else {
/* +-----------------------------------------------------------------------+ */
/* | Insert the point into the list according to its (replaced) function   | */
/* | value.                                                                | */
/* +-----------------------------------------------------------------------+ */
	    pos = start;
	    i__1 = *maxfunc;
	    for (i__ = 1; i__ <= i__1; ++i__) {
/* +-----------------------------------------------------------------------+ */
/* | The point has to be added at the end of the list.                     | */
/* +-----------------------------------------------------------------------+ */
		if (point[pos] == 0) {
		    point[*replace] = point[pos];
		    point[pos] = *replace;
/*                  write(logfile,*) 'Point is added at the end of the ' */
/*     +             , 'list ',l, replace */
		    goto L40;
		} else {
		    if (f[(point[pos] << 1) + 1] > f[(*replace << 1) + 1]) {
			point[*replace] = point[pos];
			point[pos] = *replace;
/*                     write(logfile,*) 'There are points with a higher ' */
/*     +               ,'f-value in the list ',l,replace, pos */
			goto L40;
		    }
		    pos = point[pos];
		}
/* L30: */
	    }
L40:
	    pos = pos;
	}
    }
} /* dirresortlist_ */

/* +-----------------------------------------------------------------------+ */
/* | JG Added 09/25/00                                                     | */
/* |                       SUBROUTINE DIRreplaceInf                        | */
/* |                                                                       | */
/* | Find out if there are infeasible points which are near feasible ones. | */
/* | If this is the case, replace the function value at the center of the  | */
/* | hyper rectangle by the lowest function value of a nearby function.    | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirreplaceinf_(integer *free, integer *freeold, 
	doublereal *f, doublereal *c__, doublereal *thirds, integer *length, 
	integer *anchor, integer *point, doublereal *c1, doublereal *c2, 
	integer *maxfunc, const integer *maxdeep, integer *maxdim, integer *n, 
	FILE *logfile, doublereal *fmax, integer jones)
{
(void) freeold;

    /* System generated locals */
    integer c_dim1, c_offset, length_dim1, length_offset, i__1, i__2, i__3;
    doublereal d__1, d__2;

    /* Local variables */
    doublereal a[32], b[32];
    integer i__, j, k, l;
    doublereal x[32], sidelength;
    integer help;

/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 Added variable to keep track of the maximum value found.  | */
/* +-----------------------------------------------------------------------+ */
    /* Parameter adjustments */
    --point;
    f -= 3;
    ++anchor;
    length_dim1 = *maxdim;
    length_offset = 1 + length_dim1;
    length -= length_offset;
    c_dim1 = *maxdim;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --c2;
    --c1;

    /* Function Body */
    i__1 = *free - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (f[(i__ << 1) + 2] > 0.) {
/* +-----------------------------------------------------------------------+ */
/* | Get the maximum side length of the hyper rectangle and then set the   | */
/* | new side length to this lengths times the growth factor.              | */
/* +-----------------------------------------------------------------------+ */
	    help = direct_dirgetmaxdeep_(&i__, &length[length_offset], maxfunc, n);
	    sidelength = thirds[help] * 2.;
/* +-----------------------------------------------------------------------+ */
/* | Set the Center and the upper and lower bounds of the rectangles.      | */
/* +-----------------------------------------------------------------------+ */
	    i__2 = *n;
	    for (j = 1; j <= i__2; ++j) {
		sidelength = thirds[length[i__ + j * length_dim1]];
		a[j - 1] = c__[j + i__ * c_dim1] - sidelength;
		b[j - 1] = c__[j + i__ * c_dim1] + sidelength;
/* L20: */
	    }
/* +-----------------------------------------------------------------------+ */
/* | The function value is reset to 'Inf', since it may have been changed  | */
/* | in an earlier iteration and now the feasible point which was close    | */
/* | is not close anymore (since the hyper rectangle surrounding the       | */
/* | current point may have shrunk).                                       | */
/* +-----------------------------------------------------------------------+ */
	    f[(i__ << 1) + 1] = HUGE_VAL;
	    f[(i__ << 1) + 2] = 2.;
/* +-----------------------------------------------------------------------+ */
/* | Check if any feasible point is near this infeasible point.            | */
/* +-----------------------------------------------------------------------+ */
	    i__2 = *free - 1;
	    for (k = 1; k <= i__2; ++k) {
/* +-----------------------------------------------------------------------+ */
/* | If the point k is feasible, check if it is near.                      | */
/* +-----------------------------------------------------------------------+ */
		if (f[(k << 1) + 2] == 0.) {
/* +-----------------------------------------------------------------------+ */
/* | Copy the coordinates of the point k into x.                           | */
/* +-----------------------------------------------------------------------+ */
		    i__3 = *n;
		    for (l = 1; l <= i__3; ++l) {
			x[l - 1] = c__[l + k * c_dim1];
/* L40: */
		    }
/* +-----------------------------------------------------------------------+ */
/* | Check if the point k is near the infeasible point, if so, replace the | */
/* | value at */
/* +-----------------------------------------------------------------------+ */
		    if (isinbox_(x, a, b, n, &c__32) == 1) {
/* Computing MIN */
			 d__1 = f[(i__ << 1) + 1], d__2 = f[(k << 1) + 1];
			 f[(i__ << 1) + 1] = MIN(d__1,d__2);
			 f[(i__ << 1) + 2] = 1.; 
		    }
		}
/* L30: */
	    }
	    if (f[(i__ << 1) + 2] == 1.) {
		f[(i__ << 1) + 1] += (d__1 = f[(i__ << 1) + 1], fabs(d__1)) *
			1e-6f;
		i__2 = *n;
		for (l = 1; l <= i__2; ++l) {
		    x[l - 1] = c__[l + i__ * c_dim1] * c1[l] + c__[l + i__ *
			    c_dim1] * c2[l];
/* L200: */
		}
		dirresortlist_(&i__, &anchor[-1], &f[3], &point[1], 
			       &length[length_offset], n, 
			       maxfunc, maxdim, maxdeep, logfile, jones);
	    } else {
/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01                                                           | */
/* | Replaced fixed value for infeasible points with maximum value found,  | */
/* | increased by 1.                                                       | */
/* +-----------------------------------------------------------------------+ */
		if (! (*fmax == f[(i__ << 1) + 1])) {
/* Computing MAX */
		    d__1 = *fmax + 1., d__2 = f[(i__ << 1) + 1];
		    f[(i__ << 1) + 1] = MAX(d__1,d__2);
		}
	    }
	}
/* L10: */
    }
/* L1000: */
} /* dirreplaceinf_ */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRInsert                                               | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ static void dirinsert_(integer *start, integer *ins, integer *point, 
	doublereal *f, integer *maxfunc)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer i__, help;

/* JG 09/17/00 Rewrote this routine. */
/*      DO 10,i = 1,maxfunc */
/*        IF (f(ins,1) .LT. f(point(start),1)) THEN */
/*          help = point(start) */
/*          point(start) = ins */
/*          point(ins) = help */
/*          GOTO 20 */
/*        END IF */
/*        IF (point(start) .EQ. 0) THEN */
/*           point(start) = ins */
/*           point(ins) = 0 */
/*           GOTO 20 */
/*        END IF */
/*        start = point(start) */
/* 10    CONTINUE */
/* 20    END */
    /* Parameter adjustments */
    f -= 3;
    --point;

    /* Function Body */
    i__1 = *maxfunc;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (point[*start] == 0) {
	    point[*start] = *ins;
	    point[*ins] = 0;
	    return;
	} else if (f[(*ins << 1) + 1] < f[(point[*start] << 1) + 1]) {
	     help = point[*start];
	     point[*start] = *ins;
	     point[*ins] = help;
	     return;
	}
	*start = point[*start];
/* L10: */
    }
} /* dirinsert_ */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRInsertList                                           | */
/* |    Changed 02-24-2000                                                 | */
/* |      Got rid of the distinction between feasible and infeasible points| */
/* |      I could do this since infeasible points get set to a high        | */
/* |      function value, which may be replaced by a function value of a   | */
/* |      nearby function at the end of the main loop.                     | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirinsertlist_(integer *new__, integer *anchor, integer *
	point, doublereal *f, integer *maxi, integer *length, integer *
	maxfunc, const integer *maxdeep, integer *n, integer *samp,
					    integer jones)
{
(void) maxdeep;
    /* System generated locals */
    integer length_dim1, length_offset, i__1;

    /* Local variables */
    integer j;
    integer pos;
    integer pos1, pos2, deep;

/* JG 09/24/00 Changed this to Getlevel */
    /* Parameter adjustments */
    f -= 3;
    --point;
    ++anchor;
    length_dim1 = *n;
    length_offset = 1 + length_dim1;
    length -= length_offset;

    /* Function Body */
    i__1 = *maxi;
    for (j = 1; j <= i__1; ++j) {
	pos1 = *new__;
	pos2 = point[pos1];
	*new__ = point[pos2];
/* JG 09/24/00 Changed this to Getlevel */
/*        deep = DIRGetMaxdeep(pos1,length,maxfunc,n) */
	deep = direct_dirgetlevel_(&pos1, &length[length_offset], maxfunc, n, jones);
	if (anchor[deep] == 0) {
	    if (f[(pos2 << 1) + 1] < f[(pos1 << 1) + 1]) {
		anchor[deep] = pos2;
		point[pos2] = pos1;
		point[pos1] = 0;
	    } else {
		anchor[deep] = pos1;
		point[pos2] = 0;
	    }
	} else {
	    pos = anchor[deep];
	    if (f[(pos2 << 1) + 1] < f[(pos1 << 1) + 1]) {
		if (f[(pos2 << 1) + 1] < f[(pos << 1) + 1]) {
		    anchor[deep] = pos2;
/* JG 08/30/00 Fixed bug. Sorting was not correct when */
/*      f(1,pos2) < f(1,pos1) < f(1,pos) */
		    if (f[(pos1 << 1) + 1] < f[(pos << 1) + 1]) {
			point[pos2] = pos1;
			point[pos1] = pos;
		    } else {
			point[pos2] = pos;
			dirinsert_(&pos, &pos1, &point[1], &f[3], maxfunc);
		    }
		} else {
		    dirinsert_(&pos, &pos2, &point[1], &f[3], maxfunc);
		    dirinsert_(&pos, &pos1, &point[1], &f[3], maxfunc);
		}
	    } else {
		if (f[(pos1 << 1) + 1] < f[(pos << 1) + 1]) {
/* JG 08/30/00 Fixed bug. Sorting was not correct when */
/*      f(pos1,1) < f(pos2,1) < f(pos,1) */
		    anchor[deep] = pos1;
		    if (f[(pos << 1) + 1] < f[(pos2 << 1) + 1]) {
			point[pos1] = pos;
			dirinsert_(&pos, &pos2, &point[1], &f[3], maxfunc);
		    } else {
			point[pos1] = pos2;
			point[pos2] = pos;
		    }
		} else {
		    dirinsert_(&pos, &pos1, &point[1], &f[3], maxfunc);
		    dirinsert_(&pos, &pos2, &point[1], &f[3], maxfunc);
		}
	    }
	}
/* L10: */
    }
/* JG 09/24/00 Changed this to Getlevel */
/*      deep = DIRGetMaxdeep(samp,length,maxfunc,n) */
    deep = direct_dirgetlevel_(samp, &length[length_offset], maxfunc, n, jones);
    pos = anchor[deep];
    if (f[(*samp << 1) + 1] < f[(pos << 1) + 1]) {
	anchor[deep] = *samp;
	point[*samp] = pos;
    } else {
	dirinsert_(&pos, samp, &point[1], &f[3], maxfunc);
    }
} /* dirinsertlist_ */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRInsertList2  (Old way to do it.)                     | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ static void dirinsertlist_2__(integer *start, integer *j, integer *k,
	 integer *list2, doublereal *w, integer *maxi, integer *n)
{
    /* System generated locals */
    integer list2_dim1, list2_offset, i__1;

    /* Local variables */
    integer i__, pos;

    /* Parameter adjustments */
    --w;
    list2_dim1 = *n;
    list2_offset = 1 + list2_dim1;
    list2 -= list2_offset;

    /* Function Body */
    pos = *start;
    if (*start == 0) {
	list2[*j + list2_dim1] = 0;
	*start = *j;
	goto L50;
    }
    if (w[*start] > w[*j]) {
	list2[*j + list2_dim1] = *start;
	*start = *j;
    } else {
	i__1 = *maxi;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (list2[pos + list2_dim1] == 0) {
		list2[*j + list2_dim1] = 0;
		list2[pos + list2_dim1] = *j;
		goto L50;
	    } else {
		if (w[*j] < w[list2[pos + list2_dim1]]) {
		    list2[*j + list2_dim1] = list2[pos + list2_dim1];
		    list2[pos + list2_dim1] = *j;
		    goto L50;
		}
	    }
	    pos = list2[pos + list2_dim1];
/* L10: */
	}
    }
L50:
    list2[*j + (list2_dim1 << 1)] = *k;
} /* dirinsertlist_2__ */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRSearchmin                                            | */
/* |    Search for the minimum in the list.                                ! */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ static void dirsearchmin_(integer *start, integer *list2, integer *
	pos, integer *k, integer *n)
{
    /* System generated locals */
    integer list2_dim1, list2_offset;

    /* Parameter adjustments */
    list2_dim1 = *n;
    list2_offset = 1 + list2_dim1;
    list2 -= list2_offset;

    /* Function Body */
    *k = *start;
    *pos = list2[*start + (list2_dim1 << 1)];
    *start = list2[*start + list2_dim1];
} /* dirsearchmin_ */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRSamplepoints                                         | */
/* |    Subroutine to sample the new points.                               | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirsamplepoints_(doublereal *c__, integer *arrayi, 
	doublereal *delta, integer *sample, integer *start, integer *length, 
	FILE *logfile, doublereal *f, integer *free, 
	integer *maxi, integer *point, doublereal *x, doublereal *l,
	 doublereal *minf, integer *minpos, doublereal *u, integer *n, 
	integer *maxfunc, const integer *maxdeep, integer *oops)
{
(void) minf;
(void) minpos;
(void) maxfunc;
(void) maxdeep;

    /* System generated locals */
    integer length_dim1, length_offset, c_dim1, c_offset, i__1, i__2;

    /* Local variables */
    integer j, k, pos;


    /* Parameter adjustments */
    --u;
    --l;
    --x;
    --arrayi;
    --point;
    f -= 3;
    length_dim1 = *n;
    length_offset = 1 + length_dim1;
    length -= length_offset;
    c_dim1 = *n;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;

    /* Function Body */
    *oops = 0;
    pos = *free;
    *start = *free;
    i__1 = *maxi + *maxi;
    for (k = 1; k <= i__1; ++k) {
	i__2 = *n;
	for (j = 1; j <= i__2; ++j) {
	    length[j + *free * length_dim1] = length[j + *sample *
		    length_dim1];
	    c__[j + *free * c_dim1] = c__[j + *sample * c_dim1];
/* L20: */
	}
	pos = *free;
	*free = point[*free];
	if (*free == 0) {
	     if (logfile)
                  directstream() << "Error, no more free positions! Increase maxfunc!\n";
	    *oops = 1;
	    return;
	}
/* L10: */
    }
    point[pos] = 0;
    pos = *start;
    i__1 = *maxi;
    for (j = 1; j <= i__1; ++j) {
	c__[arrayi[j] + pos * c_dim1] = c__[arrayi[j] + *sample * c_dim1] + *
		delta;
	pos = point[pos];
	c__[arrayi[j] + pos * c_dim1] = c__[arrayi[j] + *sample * c_dim1] - *
		delta;
	pos = point[pos];
/* L30: */
    }
    ASRT(pos <= 0);
} /* dirsamplepoints_ */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRDivide                                               | */
/* |    Subroutine to divide the hyper rectangles according to the rules.  | */
/* |    Changed 02-24-2000                                                 | */
/* |      Replaced if statement by min (line 367)                          | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirdivide_(integer *new__, integer *currentlength, 
	integer *length, integer *point, integer *arrayi, integer *sample, 
	integer *list2, doublereal *w, integer *maxi, doublereal *f, integer *
	maxfunc, const integer *maxdeep, integer *n)
{
(void) maxfunc;
(void) maxdeep;

    /* System generated locals */
    integer length_dim1, length_offset, list2_dim1, list2_offset, i__1, i__2;
    doublereal d__1, d__2;

    /* Local variables */
    integer i__, j, k, pos, pos2;
    integer start;


    /* Parameter adjustments */
    f -= 3;
    --point;
    --w;
    list2_dim1 = *n;
    list2_offset = 1 + list2_dim1;
    list2 -= list2_offset;
    --arrayi;
    length_dim1 = *n;
    length_offset = 1 + length_dim1;
    length -= length_offset;

    /* Function Body */
    start = 0;
    pos = *new__;
    i__1 = *maxi;
    for (i__ = 1; i__ <= i__1; ++i__) {
	j = arrayi[i__];
	w[j] = f[(pos << 1) + 1];
	k = pos;
	pos = point[pos];
/* Computing MIN */
	d__1 = f[(pos << 1) + 1], d__2 = w[j];
	w[j] = MIN(d__1,d__2);
	pos = point[pos];
	dirinsertlist_2__(&start, &j, &k, &list2[list2_offset], &w[1], maxi, 
		n);
/* L10: */
    }
    ASRT(pos <= 0);
    i__1 = *maxi;
    for (j = 1; j <= i__1; ++j) {
	dirsearchmin_(&start, &list2[list2_offset], &pos, &k, n);
	pos2 = start;
	length[k + *sample * length_dim1] = *currentlength + 1; 
	i__2 = *maxi - j + 1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    length[k + pos * length_dim1] = *currentlength + 1; 
	    pos = point[pos];
	    length[k + pos * length_dim1] = *currentlength + 1;
/* JG 07/10/01 pos2 = 0 at the end of the 30-loop. Since we end */
/*             the loop now, we do not need to reassign pos and pos2. */
	    if (pos2 > 0) {
		pos = list2[pos2 + (list2_dim1 << 1)];
		pos2 = list2[pos2 + list2_dim1];
	    }
/* L30: */
	}
/* L20: */
    }
} /* dirdivide_ */

/* +-----------------------------------------------------------------------+ */
/* |                                                                       | */
/* |                       SUBROUTINE DIRINFCN                             | */
/* |                                                                       | */
/* | Subroutine DIRinfcn unscales the variable x for use in the            | */
/* | user-supplied function evaluation subroutine fcn. After fcn returns   | */
/* | to DIRinfcn, DIRinfcn then rescales x for use by DIRECT.              | */
/* |                                                                       | */
/* | On entry                                                              | */
/* |                                                                       | */
/* |        fcn -- The argument containing the name of the user-supplied   | */
/* |               subroutine that returns values for the function to be   | */
/* |               minimized.                                              | */
/* |                                                                       | */
/* |          x -- A double-precision vector of length n. The point at     | */
/* |               which the derivative is to be evaluated.                | */
/* |                                                                       | */
/* |        xs1 -- A double-precision vector of length n. Used for         | */
/* |               scaling and unscaling the vector x by DIRinfcn.         | */
/* |                                                                       | */
/* |        xs2 -- A double-precision vector of length n. Used for         | */
/* |               scaling and unscaling the vector x by DIRinfcn.         | */
/* |                                                                       | */
/* |          n -- An integer. The dimension of the problem.               | */
/* |       kret -- An Integer. If kret =  1, the point is infeasible,      | */
/* |                              kret = -1, bad problem set up,           | */
/* |                              kret =  0, feasible.                     | */
/* |                                                                       | */
/* | On return                                                             | */
/* |                                                                       | */
/* |          f -- A double-precision scalar.                              | */
/* |                                                                       | */
/* | Subroutines and Functions                                             | */
/* |                                                                       | */
/* | The subroutine whose name is passed through the argument fcn.         | */
/* |                                                                       | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirinfcn_(fp fcn, doublereal *x, doublereal *c1, 
	doublereal *c2, integer *n, doublereal *f, integer *flag__, 
				       void *fcn_data)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer i__;

/* +-----------------------------------------------------------------------+ */
/* | Variables to pass user defined data to the function to be optimized.  | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | Unscale the variable x.                                               | */
/* +-----------------------------------------------------------------------+ */
    /* Parameter adjustments */
    --c2;
    --c1;
    --x;

    /* Function Body */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = (x[i__] + c2[i__]) * c1[i__];
/* L20: */
    }
/* +-----------------------------------------------------------------------+ */
/* | Call the function-evaluation subroutine fcn.                          | */
/* +-----------------------------------------------------------------------+ */
    *flag__ = 0;
    *f = fcn(*n, &x[1], flag__, fcn_data);
/* +-----------------------------------------------------------------------+ */
/* | Rescale the variable x.                                               | */
/* +-----------------------------------------------------------------------+ */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] = x[i__] / c1[i__] - c2[i__];
/* L30: */
    }
} /* dirinfcn_ */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRGet_I                                                | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirget_i__(integer *length, integer *pos, integer *
	arrayi, integer *maxi, integer *n, integer *maxfunc)
{
(void) maxfunc;
    /* System generated locals */
    integer length_dim1, length_offset, i__1;

    /* Local variables */
    integer i__, j, help;

    /* Parameter adjustments */
    --arrayi;
    length_dim1 = *n;
    length_offset = 1 + length_dim1;
    length -= length_offset;

    /* Function Body */
    j = 1;
    help = length[*pos * length_dim1 + 1];
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	if (length[i__ + *pos * length_dim1] < help) {
	    help = length[i__ + *pos * length_dim1];
	}
/* L10: */
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (length[i__ + *pos * length_dim1] == help) {
	    arrayi[j] = i__;
	    ++j;
	}
/* L20: */
    }
    *maxi = j - 1;
} /* dirget_i__ */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRInit                                                 | */
/* |    Initialise all needed variables and do the first run of the        | */
/* |    algorithm.                                                         | */
/* |    Changed 02/24/2000                                                 | */
/* |       Changed fcn Double precision to fcn external!                   | */
/* |    Changed 09/15/2000                                                 | */
/* |       Added distinction between Jones way to characterize rectangles  | */
/* |       and our way. Common variable JONES controls which way we use.   | */
/* |          JONES = 0    Jones way (Distance from midpoint to corner)    | */
/* |          JONES = 1    Our way (Length of longest side)                | */
/* |    Changed 09/24/00                                                   | */
/* |       Added array levels. Levels contain the values to characterize   | */
/* |       the hyperrectangles.                                            | */
/* |    Changed 01/22/01                                                   | */
/* |       Added variable fmax to keep track of maximum value found.       | */
/* |       Added variable Ifeasiblef to keep track if feasibel point has   | */
/* |       been found.                                                     | */
/* |    Changed 01/23/01                                                   | */
/* |       Added variable Ierror to keep track of errors.                  | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirinit_(doublereal *f, fp fcn, doublereal *c__, 
	integer *length, integer *actdeep, integer *point, integer *anchor, 
	integer *free, FILE *logfile, integer *arrayi, 
	integer *maxi, integer *list2, doublereal *w, doublereal *x, 
	doublereal *l, doublereal *u, doublereal *minf, integer *minpos, 
	doublereal *thirds, doublereal *levels, integer *maxfunc, const integer *
	maxdeep, integer *n, integer *maxor, doublereal *fmax, integer *
	ifeasiblef, integer *iinfeasible, integer *ierror, void *fcndata,
        integer jones, TIMESTAMPTYPE &starttime, double maxtime, volatile int &force_stop)
{
(void) maxtime;

    /* System generated locals */
    integer c_dim1, c_offset, length_dim1, length_offset, list2_dim1, 
	    list2_offset, i__1, i__2;

    /* Local variables */
    integer i__, j;
    integer new__, help, oops;
    doublereal help2, delta;
    doublereal costmin;

(void) costmin;

/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 Added variable to keep track of the maximum value found.  | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 Added variable Ifeasiblef to keep track if feasibel point | */
/* |             has been found.                                           | */
/* | JG 01/23/01 Added variable Ierror to keep track of errors.            | */
/* | JG 03/09/01 Added IInfeasible to keep track if an infeasible point has| */
/* |             been found.                                               | */
/* +-----------------------------------------------------------------------+ */
/* JG 09/15/00 Added variable JONES (see above) */
/* +-----------------------------------------------------------------------+ */
/* | Variables to pass user defined data to the function to be optimized.  | */
/* +-----------------------------------------------------------------------+ */
    /* Parameter adjustments */
    --point;
    f -= 3;
    ++anchor;
    --u;
    --l;
    --x;
    --w;
    list2_dim1 = *maxor;
    list2_offset = 1 + list2_dim1;
    list2 -= list2_offset;
    --arrayi;
    length_dim1 = *n;
    length_offset = 1 + length_dim1;
    length -= length_offset;
    c_dim1 = *maxor;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;

    /* Function Body */
    *minf = HUGE_VAL;
    costmin = *minf;
/* JG 09/15/00 If Jones way of characterising rectangles is used, */
/*             initialise thirds to reflect this. */
    if (jones == 0) {
	i__1 = *n - 1;
	for (j = 0; j <= i__1; ++j) {
	    w[j + 1] = sqrt(*n - j + j / 9.) * .5;
/* L5: */
	}
	help2 = 1.;
	i__1 = *maxdeep / *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = *n - 1;
	    for (j = 0; j <= i__2; ++j) {
		levels[(i__ - 1) * *n + j] = w[j + 1] / help2;
/* L8: */
	    }
	    help2 *= 3.;
/* L10: */
	}
    } else {
/* JG 09/15/00 Initialiase levels to contain 1/j */
	help2 = 3.;
	i__1 = *maxdeep;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    levels[i__] = 1. / help2;
	    help2 *= 3.;
/* L11: */
	}
	levels[0] = 1.;
    }
    help2 = 3.;
    i__1 = *maxdeep;
    for (i__ = 1; i__ <= i__1; ++i__) {
	thirds[i__] = 1. / help2;
	help2 *= 3.;
/* L21: */
    }
    thirds[0] = 1.;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	c__[i__ + c_dim1] = .5;
	x[i__] = .5;
	length[i__ + length_dim1] = 0;
/* L20: */
    }
    direct_dirinfcn_(fcn, &x[1], &l[1], &u[1], n, &f[3], &help, fcndata);
    if (force_stop) {
	 *ierror = -102;
	 return;
    }
    f[4] = (doublereal) help;
    *iinfeasible = help;
    *fmax = f[3];
/* 09/25/00 Added this */
/*      if (f(1,1) .ge. 1.E+6) then */
    if (f[4] > 0.) {
	f[3] = HUGE_VAL;
	*fmax = f[3];
	*ifeasiblef = 1;
    } else {
	*ifeasiblef = 0;
    }
/* JG 09/25/00 Remove IF */
    *minf = f[3];
    costmin = f[3];
    *minpos = 1;
    *actdeep = 2;
    point[1] = 0;
    *free = 2;
    delta = thirds[1];
    if (nlopt_stop_time_(starttime, maxtime)) {
         *ierror = DIRECT_MAXTIME_EXCEEDED;
         return;
    }
    direct_dirget_i__(&length[length_offset], &c__1, &arrayi[1], maxi, n, maxfunc);
    new__ = *free;
    direct_dirsamplepoints_(&c__[c_offset], &arrayi[1], &delta, &c__1, &new__, &
	    length[length_offset], logfile, &f[3], free, maxi, &
	    point[1], &x[1], &l[1], minf, minpos, &u[1], n, 
	    maxfunc, maxdeep, &oops);
/* +-----------------------------------------------------------------------+ */
/* | JG 01/23/01 Added error checking.                                     | */
/* +-----------------------------------------------------------------------+ */
    if (oops > 0) {
	*ierror = -4;
	return;
    }
/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 Added variable to keep track of the maximum value found.  | */
/* |             Added variable to keep track if feasible point was found. | */
/* +-----------------------------------------------------------------------+ */
    direct_dirsamplef_(&c__[c_offset], &arrayi[1], &delta, &c__1, &new__, &length[
	    length_offset], logfile, &f[3], free, maxi, &point[
	    1], fcn, &x[1], &l[1], minf, minpos, &u[1], n, maxfunc, 
	    maxdeep, &oops, fmax, ifeasiblef, iinfeasible, fcndata,
	    force_stop);
    if (force_stop) {
	 *ierror = -102;
	 return;
    }
    if (nlopt_stop_time_(starttime, maxtime)) {
         *ierror = DIRECT_MAXTIME_EXCEEDED;
         return;
    }
/* +-----------------------------------------------------------------------+ */
/* | JG 01/23/01 Added error checking.                                     | */
/* +-----------------------------------------------------------------------+ */
    if (oops > 0) {
	*ierror = -5;
	return;
    }
    direct_dirdivide_(&new__, &c__0, &length[length_offset], &point[1], &arrayi[1], &
	    c__1, &list2[list2_offset], &w[1], maxi, &f[3], maxfunc, 
	    maxdeep, n);
    direct_dirinsertlist_(&new__, &anchor[-1], &point[1], &f[3], maxi, &
	    length[length_offset], maxfunc, maxdeep, n, &c__1, jones);
} /* dirinit_ */

/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE DIRInitList                                             | */
/* |    Initialise the list.                                               | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirinitlist_(integer *anchor, integer *free, integer *
	point, doublereal *f, integer *maxfunc, const integer *maxdeep)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer i__;

/*   f -- values of functions. */
/*   anchor -- anchors of lists with deep i */
/*   point -- lists */
/*   free  -- first free position */
    /* Parameter adjustments */
    f -= 3;
    --point;
    ++anchor;

    /* Function Body */
    i__1 = *maxdeep;
    for (i__ = -1; i__ <= i__1; ++i__) {
	anchor[i__] = 0;
/* L10: */
    }
    i__1 = *maxfunc;
    for (i__ = 1; i__ <= i__1; ++i__) {
	f[(i__ << 1) + 1] = 0.;
	f[(i__ << 1) + 2] = 0.;
	point[i__] = i__ + 1;
/*       point(i) = 0 */
/* L20: */
    }
    point[*maxfunc] = 0;
    *free = 1;
} /* dirinitlist_ */

/* +-----------------------------------------------------------------------+ */
/* |                                                                       | */
/* |                       SUBROUTINE DIRPREPRC                            | */
/* |                                                                       | */
/* | Subroutine DIRpreprc uses an afine mapping to map the hyper-box given | */
/* | by the constraints on the variable x onto the n-dimensional unit cube.| */
/* | This mapping is done using the following equation:                    | */
/* |                                                                       | */
/* |               x(i)=x(i)/(u(i)-l(i))-l(i)/(u(i)-l(i)).                 | */
/* |                                                                       | */
/* | DIRpreprc checks if the bounds l and u are well-defined. That is, if  | */
/* |                                                                       | */
/* |               l(i) < u(i) forevery i.                                 | */
/* |                                                                       | */
/* | On entry                                                              | */
/* |                                                                       | */
/* |          u -- A double-precision vector of length n. The vector       | */
/* |               containing the upper bounds for the n independent       | */
/* |               variables.                                              | */
/* |                                                                       | */
/* |          l -- A double-precision vector of length n. The vector       | */
/* |               containing the lower bounds for the n independent       | */
/* |               variables.                                              | */
/* |                                                                       | */
/* |          n -- An integer. The dimension of the problem.               | */
/* |                                                                       | */
/* | On return                                                             | */
/* |                                                                       | */
/* |        xs1 -- A double-precision vector of length n, used for scaling | */
/* |               and unscaling the vector x.                             | */
/* |                                                                       | */
/* |        xs2 -- A double-precision vector of length n, used for scaling | */
/* |               and unscaling the vector x.                             | */
/* |                                                                       | */
/* |                                                                       | */
/* |       oops -- An integer. If an upper bound is less than a lower      | */
/* |               bound or if the initial point is not in the             | */
/* |               hyper-box oops is set to 1 and iffco terminates.        | */
/* |                                                                       | */
/* +-----------------------------------------------------------------------+ */
/* Subroutine */ void direct_dirpreprc_(doublereal *u, doublereal *l, integer *n, 
	doublereal *xs1, doublereal *xs2, integer *oops)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer i__;
    doublereal help;

    /* Parameter adjustments */
    --xs2;
    --xs1;
    --l;
    --u;

    /* Function Body */
    *oops = 0;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* +-----------------------------------------------------------------------+ */
/* | Check if the hyper-box is well-defined.                               | */
/* +-----------------------------------------------------------------------+ */
	if (u[i__] <= l[i__]) {
	    *oops = 1;
	    return;
	}
/* L20: */
    }
/* +-----------------------------------------------------------------------+ */
/* | Scale the initial iterate so that it is in the unit cube.             | */
/* +-----------------------------------------------------------------------+ */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	help = u[i__] - l[i__];
	xs2[i__] = l[i__] / help;
	xs1[i__] = help;
/* L50: */
    }
} /* dirpreprc_ */

/* Subroutine */ void direct_dirheader_(FILE *logfile, integer *version, 
	doublereal *x, integer *n, doublereal *eps, integer *maxf, integer *
	maxt, doublereal *l, doublereal *u, integer *algmethod, integer *
	maxfunc, const integer *maxdeep, doublereal *fglobal, doublereal *fglper, 
	integer *ierror, doublereal *epsfix, integer *iepschange, doublereal *
	volper, doublereal *sigmaper)
{
(void) maxdeep;
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer imainver, i__, numerrors, isubsubver, ihelp, isubver;


/* +-----------------------------------------------------------------------+ */
/* | Variables to pass user defined data to the function to be optimized.  | */
/* +-----------------------------------------------------------------------+ */
    /* Parameter adjustments */
    --u;
    --l;
    --x;

    /* Function Body */
    if (logfile)
         directstream() << "------------------- Log file ------------------\n";

    numerrors = 0;
    *ierror = 0;
    imainver = *version / 100;
    ihelp = *version - imainver * 100;
    isubver = ihelp / 10;
    ihelp -= isubver * 10;
    isubsubver = ihelp;
/* +-----------------------------------------------------------------------+ */
/* | JG 01/13/01 Added check for epsilon. If epsilon is smaller 0, we use  | */
/* |             the update formula from Jones. We then set the flag       | */
/* |             iepschange to 1, and store the absolute value of eps in   | */
/* |             epsfix. epsilon is then changed after each iteration.     | */
/* +-----------------------------------------------------------------------+ */
    if (*eps < 0.) {
	*iepschange = 1;
	*epsfix = -(*eps);
	*eps = -(*eps);
    } else {
	*iepschange = 0;
	*epsfix = 1e100;
    }

/* +-----------------------------------------------------------------------+ */
/* | JG 07/16/01 Removed printout of contents in cdata(1).                 | */
/* +-----------------------------------------------------------------------+ */
/*      write(logfile,*) cdata(1) */

    if (logfile) {
         directstream() << "DIRECT Version " << imainver << "." << isubver << "." << isubsubver << "\n";
         directstream() << " Problem dimension n: " << *n << "\n";
         directstream() << " Eps value: " << *eps << "\n";
         directstream() << " Maximum number of f-evaluations (maxf): " << *maxf << "\n";
         directstream() << " Maximum number of iterations (MaxT): " << *maxt << "\n";
         directstream() << " Value of f_global: " << *fglobal << "\n";
         directstream() << " Global percentage wanted: " << *fglper << "\n";
         directstream() << " Volume percentage wanted: " << *volper << "\n";
         directstream() << " Measure percentage wanted: " << *sigmaper << "\n";
         directstream() << ( ( *iepschange == 1 ) ? "Epsilon is changed using the Jones formula.\n" : "Epsilon is constant.\n" );
         directstream() << ( ( *algmethod  == 0 ) ? "Jones original DIRECT algorithm is used.\n" : "Our modification of the DIRECT algorithm is used.\n" );
    }

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (u[i__] <= l[i__]) {
	    *ierror = -1;
	    if (logfile)
                 directstream() << "WARNING: bounds on variable x" << i__ << ": " << l[i__] << " <= xi <= " << u[i__] << "\n";
	    ++numerrors;
	} else {
	    if (logfile)
                 directstream() << "Bounds on variable x" << i__ << ": " << l[i__] << " <= xi <= " << u[i__] << "\n";
	}
/* L1010: */
    }
/* +-----------------------------------------------------------------------+ */
/* | If there are to many function evaluations or to many iteration, note  | */
/* | this and set the error flag accordingly. Note: If more than one error | */
/* | occurred, we give out an extra message.                               | */
/* +-----------------------------------------------------------------------+ */
    if (*maxf + 20 > *maxfunc) {
	if (logfile)
        {
directstream() << "WARNING: The maximum number of function evaluations (" << *maxf << ") is higher than\n";
directstream() << "         the constant maxfunc (" << *maxfunc << ").  Increase maxfunc in subroutine DIRECT\n";
directstream() << "         or decrease the maximum number of function evaluations.\n";
        }
        ++numerrors;
	*ierror = -2;
    }
    if (*ierror < 0) {
        if (logfile) directstream() << "----------------------------------\n";
	if (numerrors == 1) {
	     if (logfile) 
                  directstream() << "WARNING: One error in the input!\n";
	} else {
	     if (logfile) 
                  directstream() << "WARNING: " << numerrors << " errors in the input!\n";
	}
    }
    if (logfile) directstream() << "----------------------------------\n";
    if (*ierror >= 0) {
	 if (logfile)
              directstream() << "Iteration # of f-eval. minf\n";
    }
/* L10005: */
} /* dirheader_ */

/* Subroutine */ void direct_dirsummary_(FILE *logfile, doublereal *x, doublereal *
	l, doublereal *u, integer *n, doublereal *minf, doublereal *fglobal, 
	integer *numfunc, integer *ierror)
{
(void) ierror;

    /* Local variables */
    integer i__;

    /* Parameter adjustments */
    --u;
    --l;
    --x;

    /* Function Body */
    if (logfile) {
         directstream() << "-----------------------Summary------------------\n";
         directstream() << "Final function value: " << *minf << "\n";
         directstream() << "Number of function evaluations: " << *numfunc << "\n";
	 if (*fglobal > -1e99)
              directstream() << "Final function value is within " << 100*(*minf - *fglobal) / MAX(1.0, fabs(*fglobal)) << "%% of global optimum\n";
         directstream() << "Index, final solution, x(i)-l(i), u(i)-x(i)\n";
	 for (i__ = 1; i__ <= *n; ++i__)
              directstream() << i__ << ", " << x[i__] << ", " << x[i__]-l[i__] << ", " << u[i__] - x[i__] << "\n";
         directstream() << "-----------------------------------------------\n";
	      
    }
} /* dirsummary_ */
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================

//
// DIRect optimiser borrowed from nl_opt
// =====================================
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


/* C-style API for DIRECT functions.  SGJ (August 2007). */

//#include "direct_internal.h"

/* Perform global minimization using (Gablonsky implementation of) DIRECT
   algorithm.   Arguments:

   f, f_data: the objective function and any user data
       -- the objective function f(n, x, undefined_flag, data) takes 4 args:
              int n: the dimension, same as dimension arg. to direct_optimize
              const double *x: array x[n] of point to evaluate
	      int *undefined_flag: set to 1 on return if x violates constraints
	                           or don't touch otherwise
              void *data: same as f_data passed to direct_optimize
          return value = value of f(x)

   dimension: the number of minimization variable dimensions
   lower_bounds, upper_bounds: arrays of length dimension of variable bounds

   x: an array of length dimension, set to optimum variables upon return
   minf: on return, set to minimum f value

   magic_eps, magic_eps_abs: Jones' "magic" epsilon parameter, and
                             also an absolute version of the same
			     (not multipled by minf).  Jones suggests
			     setting this to 1e-4, but 0 also works...

   max_feval, max_iter: maximum number of function evaluations & DIRECT iters
   volume_reltol: relative tolerance on hypercube volume (0 if none)
   sigma_reltol: relative tolerance on hypercube "measure" (??) (0 if none)

   fglobal: the global minimum of f, if known ahead of time
       -- this is mainly for benchmarking, in most cases it
          is not known and you should pass DIRECT_UNKNOWN_FGLOBAL
   fglobal_reltol: relative tolerance on how close we should find fglobal
       -- ignored if fglobal is DIRECT_UNKNOWN_FGLOBAL

   logfile: an output file to write diagnostic info to (NULL for no I/O)

   algorithm: whether to use the original DIRECT algorithm (DIRECT_ORIGINAL)
              or Gablonsky's "improved" version (DIRECT_GABLONSKY)
*/
direct_return_code direct_optimize(
     direct_objective_func f, void *f_data,
     int dimension,
     const double *lower_bounds, const double *upper_bounds,

     double *x, double *minf, 

     int max_feval, int max_iter,
     double maxtime,
     double magic_eps, double magic_eps_abs,
     double volume_reltol, double sigma_reltol,
     volatile int &force_stop,

     double fglobal,
     double fglobal_reltol,

     direct_algorithm algorithm)
{
     TIMESTAMPTYPE starttime = getstarttime();
     FILE *logfile = stdout;

     integer algmethod = algorithm == DIRECT_GABLONSKY;
     integer ierror;
     doublereal *l, *u;
     int i;

     /* convert to percentages: */
     volume_reltol *= 100;
     sigma_reltol *= 100;
     fglobal_reltol *= 100;

     /* make sure these are ignored if <= 0 */
     if (volume_reltol <= 0) volume_reltol = -1;
     if (sigma_reltol <= 0) sigma_reltol = -1;

     if (fglobal == DIRECT_UNKNOWN_FGLOBAL)
	  fglobal_reltol = DIRECT_UNKNOWN_FGLOBAL_RELTOL;

     if (dimension < 1) return DIRECT_INVALID_ARGS;

     l = (doublereal *) malloc(sizeof(doublereal) * dimension * 2);
     if (!l) return DIRECT_OUT_OF_MEMORY;
     u = l + dimension;
     for (i = 0; i < dimension; ++i) {
	  l[i] = lower_bounds[i];
	  u[i] = upper_bounds[i];
     }
     
     direct_direct_(f, x, &dimension, &magic_eps, magic_eps_abs,
		    &max_feval, &max_iter, 
                    starttime, maxtime, force_stop,
		    minf,
		    l, u,
		    &algmethod,
		    &ierror,
		    logfile,
		    &fglobal, &fglobal_reltol,
		    &volume_reltol, &sigma_reltol,
		    f_data);

     free(l);

     return (direct_return_code) ierror;
}
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
// ===============================================================
