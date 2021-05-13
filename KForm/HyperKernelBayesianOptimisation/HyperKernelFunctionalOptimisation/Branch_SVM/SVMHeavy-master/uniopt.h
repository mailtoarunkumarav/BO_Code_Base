
//
// Global optimisation setup/run helper functions
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "globalopt.h"
#include "gridopt.h"
#include "directopt.h"
#include "bayesopt.h"

#ifndef _uniopt_h

//
// Standard procedure:
//
// 1. Construct an optimiser using one of the optimiser setup funcitons
// 2. Construct a new problem definition (use the OptProb constructor).
// 3. Optimise using the uniOpt function.
// 4. Delete the problem definition (use delete).
// 5. Delete the optimiser using deleteOptim.
//
// For example (grid optimiser skeleton)
//
// GlobalOptions *optDef = setupGridOpt(...);
// OptProb *probDef = new OptProb(...);
// uniOpt(optDef,probDef,...);
// delete probDef;
// deleteOptim(optDef);
//


//
// Optimiser Setup functions
//
// Return value: pointer to global optimiser
//
// Common arguments:
//
//  - maxtraintime: max training time (0 for unlimited)
//
//
// Grid arguments:
//
//  - numZooms: number of zooms to find optima
//  - zoomFact: zoom magnification factor
//
//
// DIRect arguments:
//
//  - maxits: maximum cube divisions (suggest 100-200)
//  - maxevals: maximum number of function evaluations (suggest 500-1000)
//  - eps: epsilon argument for DIRect.  Suggest 0, 1e-4, 1e-3 etc
//  - algorithm: method used (either DIRECT_ORIGINAL or DIRECT_GABLONSKY).
//
//
// Bayesian Optimisation arguments I:
//
// method: (ignored if impmeasu != NULL)
//         0 = raw output (ignoring variance, just minimise f(x))
//         1 = EI (expected improvement)
//         2 = PI (probability of improvement)
//         3 = gpUCB
//
// fnapprox: surrogate function approximator.  May point to ML_Base with
//           appropriate output type.  If NULL then the default scalar GPR
//           will be used.
// impmeasu: improvement measure function.  If set will be used instead
//           of EI/PI/whatever.  NULL to use inbuilt method instead.
//
// startpoints: number of random (uniformly distributed) seeds used to
//         initialise the problem.  Note that you can also put points 
//         into fnapprox before calling this funciton if you have 
//         existing results or want to follow a particular pattern.
// totiters: total number of iterations in Bayesian optimisation
//         set 0 for unlimited.
//
// ztol:  zero tolerance (used when assessing sigma > 0, sigma = 0)
// delta: used by GP-UCB algorithm (0.1 by default)
// nu:    used by GP-UCB algorithm (apparently always 1)
//
//
// Bayesian Optimisation arguments II (fnapprox GPR_Scalar, impmeasu NULL):
//
// method: (ignored if impmeasu != NULL)
//         0 = raw output (ignoring variance, just minimise f(x))
//         1 = EI (expected improvement)
//         2 = PI (probability of improvement)
//         3 = gpUCB
//
// fnsigma: sigma value for scalar GPR regressor
// fnkernr: GPR kernel parameter r (kernel is exp(-||x-y||^2/2r)
//
// startpoints: number of random (uniformly distributed) seeds used to
//         initialise the problem.  Note that you can also put points 
//         into fnapprox before calling this funciton if you have 
//         existing results or want to follow a particular pattern.
// totiters: total number of iterations in Bayesian optimisation
//         set 0 for unlimited.
//
// ztol:  zero tolerance (used when assessing sigma > 0, sigma = 0)
// delta: used by GP-UCB algorithm (0.1 by default)
// nu:    used by GP-UCB algorithm (apparently always 1)
//
//
// Bayesian Optimisation arguments III (fnapprox GPR_Vector, impmeasu EHI):
//
// odim: dimension of objective space
//
// fnsigma: sigma value for scalar GPR regressor
// fnkernr: GPR kernel parameter r (kernel is exp(-||x-y||^2/2r)
//
// startpoints: number of random (uniformly distributed) seeds used to
//         initialise the problem.  Note that you can also put points 
//         into fnapprox before calling this funciton if you have 
//         existing results or want to follow a particular pattern.
// totiters: total number of iterations in Bayesian optimisation
//         set 0 for unlimited.
//
// ztol:  zero tolerance (used when assessing sigma > 0, sigma = 0)
// delta: used by GP-UCB algorithm (0.1 by default)
// nu:    used by GP-UCB algorithm (apparently always 1)
//
//
// Bayesian Optimisation arguments IV (fnapprox GPR_Vector, impmeasu IMP_ParSVM):
//
// odim: dimension of objective space
//
// fnsigma: sigma value for scalar GPR regressor
// fnkernr: GPR kernel parameter r (kernel is exp(-||x-y||^2/2r)
//
// impkernr: IMP kernel parameter r (kernel is 1/(1+exp(-r.min(x-y)) )
//
// startpoints: number of random (uniformly distributed) seeds used to
//         initialise the problem.  Note that you can also put points 
//         into fnapprox before calling this funciton if you have 
//         existing results or want to follow a particular pattern.
// totiters: total number of iterations in Bayesian optimisation
//         set 0 for unlimited.
//
// ztol:  zero tolerance (used when assessing sigma > 0, sigma = 0)
// delta: used by GP-UCB algorithm (0.1 by default)
// nu:    used by GP-UCB algorithm (apparently always 1)
//

GlobalOptions *setupGridOpt(int numZooms = 3,
                            double zoomFact = 0.33333,
                            double maxtraintime = 0);

GlobalOptions *setupDirectOpt(long int maxits = 200,
                              long int maxevals = 1000,
                              double eps = 1e-4,
                              direct_algorithm algorithm = DIRECT_ORIGINAL,
                              double maxtraintime = 0);

GlobalOptions *setupBayesOptI(int method = 0,
                              ML_Base *fnapprox = NULL,
                              IMP_Generic *impmeasu = NULL,
                              int startpoints = 5,
                              unsigned int totiters = 200,
                              double ztol = 1e-8,
                              double delta = 0.1,
                              double nu = 1,
                              double maxtraintime = 0);

GlobalOptions *setupBayesOptII(int method = 0,
                               double fnsigma = 1,
                               double fnkernr = 1,
                               int startpoints = 5,
                               unsigned int totiters = 200,
                               double ztol = 1e-8,
                               double delta = 0.1,
                               double nu = 1,
                               long int maxits = 200,
                               long int maxevals = 1000,
                               double eps = 1e-4,
                               direct_algorithm algorithm = DIRECT_ORIGINAL,
                               double maxtraintime = 0);

GlobalOptions *setupBayesOptIII(int odim,
                                double fnsigma = 1,
                                double fnkernr = 1,
                                int ehimethod = 0,
                                int startpoints = 5,
                                unsigned int totiters = 200,
                                double ztol = 1e-8,
                                double delta = 0.1,
                                double nu = 1,
                                long int maxits = 200,
                                long int maxevals = 1000,
                                double eps = 1e-4,
                                direct_algorithm algorithm = DIRECT_ORIGINAL,
                                double maxtraintime = 0,
                                double traintimeoverride = 0);

GlobalOptions *setupBayesOptIV(int odim,
                               double fnsigma = 1,
                               double fnkernr = 1,
                               double impkernr = 1,
                               int svmmethod = 3,
                               int startpoints = 5,
                               unsigned int totiters = 200,
                               double ztol = 1e-8,
                               double delta = 0.1,
                               double nu = 1,
                               long int maxits = 200,
                               long int maxevals = 1000,
                               double eps = 1e-4,
                               direct_algorithm algorithm = DIRECT_ORIGINAL,
                               double maxtraintime = 0);

void deleteOptim(GlobalOptions *opt);

//
// Optimisation Problem Setup
//

class OptProb
{
public:

    // General problem specifications
    //
    // idim: dimensionality of problem (input space)
    // odim: dimensionality of objective space
    //       (0 in this context means single-objective optimisation)
    //
    // xmin: lower bound on x
    // xmax: upper bound on x

    int idim;
    int odim;

    Vector<gentype> xmin;
    Vector<gentype> xmax;

    // Function to be optimised
    //
    // res: idim dimensional result array, should be filled out by fn
    // x:   argument.
    // arg: set equal to fnarg

    void (*fn)(double *res, const double *x, void *arg);
    void *fnarg;

    // Grid-only specifications
    //
    // numpts:   number of points in grid on each axis (default 10)
    // distMode: distribution of points on axis (default 0)
    //           0 - linear distribution
    //           1 - log distribution
    //           2 - antilog distribution
    //           3 - random distribution
    // varsType: variable types on each axis (default 1)
    //           0 - integer
    //           1 - real

    Vector<int> numpts;
    Vector<int> distMode;
    Vector<int> varsType;

    // Constructors

    OptProb(int xidim,
            int xodim,
            double *xxmin,
            double *xxmax,
            void (*xfn)(double *, const double *, void *),
            void *xfnarg,
            int *xnumpts = NULL,
            int *xdistMode = NULL,
            int *xvarsType = NULL)
    {
        NiceAssert( xidim >= 1 );
        NiceAssert( xodim >= 0 );
        NiceAssert( xxmin );
        NiceAssert( xxmax );

        idim = xidim;
        odim = xodim;

        fn    = xfn;
        fnarg = xfnarg;

        xmin.resize(idim);
        xmax.resize(idim);

        numpts.resize(idim)   = 10;
        distMode.resize(idim) = 0;
        varsType.resize(idim) = 1;

        int i;

        for ( i = 0 ; i < idim ; i++ )
        {
            xmin("&",i) = xxmin[i];
            xmax("&",i) = xxmax[i];

            if ( xnumpts )
            {
                numpts("&",i) = xnumpts[i];
            }

            if ( xdistMode )
            {
                distMode("&",i) = xdistMode[i];
            }

            if ( xvarsType )
            {
                varsType("&",i) = xvarsType[i];
            }
        }

        return;
    }
};


//
// Actual optimisation call
//
// optDef:  global optimiser definition
// probDef: optimisation problem definition
//
// xres: x result is stored here (must be preallocated to size idim)
// fres: f(x) result is stored here (must be preallocated to size odim)
//
// allxres: pointer to array where all x evaluations are to be stored
// allfres: pointer to array where all f(x) evaluations are to be stored
//          (the pointer should be to an as-yet unallocated array)
//          (set NULL to not keep these results)
//                                                  
// Nres: number of results in allxres/allfres
//
// killSwitch: set this 1 at any time to stop optimisation early
//

int uniOpt(GlobalOptions *optDef,
           OptProb *probDef,
           double *xres,
           double *frxes,
           double ***allxres,
           double ***allfres,
           int &Nres,
           svmvolatile int &killSwitch);


#endif
