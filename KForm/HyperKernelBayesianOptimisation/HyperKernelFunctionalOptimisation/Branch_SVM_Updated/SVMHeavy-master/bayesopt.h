
//
// Bayesian Optimiser
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "directopt.h"
#include "smboopt.h"
#include "imp_generic.h"

//
// Attempts to minimise target function using Bayesian optimisation.
//

#ifndef _bayesopt_h
#define _bayesopt_h


class BayesOptions : public SMBOOptions
{
public:

    // WARNING: some documentation may be out of date.  Check code.
    //
    // method: 0  = raw output (ignoring variance, just minimise f(x))
    //         1  = EI (expected improvement) (default)
    //         2  = PI (probability of improvement)
    //         3  = gpUCB basic (Brochu et al - recommended)
    //         4  = gpUCB finite (Srinivas |D| finite)
    //         5  = gpUCB infinite (Srinivas |D| infinite)
    //         6  = gpUCB p basic (Brochu et al)
    //         7  = gpUCB p finite (Srinivas |D| finite, pi_t = zeta(p) t^p)
    //         8  = gpUCB p infinite (Srinivas |D| inf, pi_t = zeta(p) t^p)
    //         9  = VO (variance only, pure exploration)
    //         10 = MO (mean only, pure exploitation)
    //         11 = user defined GP-UCB method:
    //              beta_t = nu.betafn(t,d,delta,modD,a)
    // sigmuseparate: for multi-recommendation by default both sigma and by
    //         are approximated by the same ML.  Alternatively you can do
    //         them separately: mu is updated for each batch, and sigma
    //         independently for each point selected.
    //         0 = use the same ML.
    //         1 = use separate MLs.
    // startpoints: number of random (uniformly distributed) seeds used to
    //         initialise the problem.  Note that you can also put points 
    //         into fnapprox before calling this funciton if you have 
    //         existing results or want to follow a particular pattern.
    // startseed: seed for RNG immediately prior to generating startpoints
    //         -1 if not used, -2 to seed with time (if >= 0 incremented whenever seeding happens so that
    //         multiple repeats have different (but predictable) sets of random numbers)
    //         Default 42.
    // algseed: seed for RNG immediately prior to running algorithm
    //         -1 if not used, -2 to seed with time (if >= 0 incremented whenever seeding happens so that
    //         multiple repeats have different (but predictable) sets of random numbers)
    //         Default -2.
    // totiters: total number of iterations in Bayesian optimisation
    //         set 0 for unlimited, -1 for 10d, -2 for err method (see err
    //         parameter).
    // stepweight: typically 0.  If you want to add a penalty that is
    //         proportional to the distance covered by the step then
    //         set non-zero.  This can be handy for example when selecting
    //         hyper-parameters, in which case incremental update time may
    //         be proportional to the change in hyperparameters.
    // intrinbatch: intrinsic batch size.  Default 1.  If > 0 then:
    //         mu(x) -> max(mu(x_0),mu(x_1),...,mu(x_{d-1}))
    //         sigma(x) -> det(covar(x_i,x_j))^(1/2intrinbatch), i,j = 1,2,...,d-1
    // intrinbatchmethod: 0 is standard, 1 means use mu(x) = min(...) instead
    // direcdim: if directpre != NULL then this is the dimension of the
    //         input of the pre-processing function, which is the dimension
    //         of that DIRect sees.  Otherwise ignored.
    // itcntmethod: this controls how the iteration counter (t) used when
    //         calculating beta (for GP-UCB) is updated in batch mode.  If 
    //         0 (default) then for each batch t -> t+1.  If 1 then for
    //         each batch t -> t+B, where B is the size of the batch (number
    //         of recommendations).
    // err: if maxitcnt == -2 then stopping uses this - see Kirschner et al, 
    //         Adaptive and Safe Bayesian Optimization in High Dimensions via One-Dimensional Subspaces
    //
    // ztol:   zero tolerance (used when assessing sigma > 0, sigma = 0)
    // delta:  used by GP-UCB algorithm (0.1 by default)
    // nu:     used by GP-UCB algorithm (almost always 1)
    // modD:   used by GP-UCB {p} finite, size of search space set (-1 to infer from gridopt, if available - default).
    // a,b,r:  used by GP-UCB {p} infinite, see Srinivas theorem 2.
    // p:      used by GP-UCB p {in}finite, see Srinivas appendix.  Basically
    //         rather than set pi_t = (pi^2.t^2)/6 we3 set
    //         pi_t = zeta(p) t^p, which satisfies all the relevant
    //         requirements.  Srinivas considers the special case p = 2,
    //         where we note that zeta(2) = (pi^2)/6.
    //
    // impmeasu: improvement measure function.  If set will be used instead
    //           of EI/PI/whatever.
    // direcpre: rather than directly optimise the acquisition function 
    //           if direcpre is set non-NULL DIRect optimises a(p(x)), where
    //           a is the acquisition function and p is the function 
    //           defined here.  The input of this must have dimension direcdim.
    // direcsubseqpre: rather than directly optimise the acquisition function
    //           if direcsubseqpre is set non-NULL DIRect optimisation
    //           a(p(x)) on all but the first recommendation in a block, where
    //           a is the acquisition function and p is the function 
    //           defined here.  The input of this must have dimension direcdim.
    //
    // gridsource: usually optimisation is continuous.  If you want to optimise
    //           on a finite set/grid then set this to point to the ML containing
    //           the valid x data.  The y value from this will be put in 
    //           x[12] and the index in x[13].  NULL by default.
    // gridcache:0 = nothing
    //           1 = pre-emptivelly add all points in gridsource to model with
    //               d = 0, then kernel is cached and to "add" a point you 
    //               just set d = 2.
    //
    // penalty: this is a vector of (positive valued) penalty functions.  When
    //           evaluating the acquisition function each of these will be
    //           evaluated and subtracted from the acquisition function.  These
    //           are used to enforce additional constraints on the ML.  Set them
    //           very positive in forbidden areas, near zero in the feasible
    //           region.
    //
    // Parameters controlling stable Bayesian optimisation
    //
    // stabpmax:    0 for no stability constraints, >= 1 for stability constraints 1:p, where p < pmax
    // stabpmin:    minimum value for p, if pmax >= 1
    // stabA:       A factor
    // stabB:       B factor
    // stabF:       F factor
    // stabbal:     [0,1]: 0 means use mu- (conservative), 1 means mu+ (optimistic), linear between
    // stabZeroPt:  zero point (lowbnd, chi)
    // stabDelrRep: repeats to calculate Delta_r
    // stabDelRep:  repeats to calculate Delta
    // stabUseSig:  set 1 to put stability scores through sigmoid function (default 1)
    // stabThresh:  threshold for stability (default 0.8)
    //
    // Unscented-Bayesian Optimisation
    //
    // @inproceedings{Nog1,
    //    author      = "Nogueira, Jos{\'e} and Martinez-Cantin, Ruben and Bernardina, Alexandre and Jamone, Lorenzo",
    //    title       = "Unscented Bayesian Optimization for Safe Robot Grasping",
    //    booktitle   = "Proceedings of the {IEEE/RSJ} International Conference on Intelligent Robots and Systems {IROS}",
    //    year        = "2016"}
    //
    // unscentUse:       0 normal, 1 use unscented transform
    // unscentK:         k value used in unscented optimisation (typically either 0 or -3)
    // unscentSqrtSigma: square root of sigma matrix used by unscented transform (noise on input x)
    //
    //
    //
    // See:
    //
    // @techreport{Bro2,
    //    author      = "Brochu, Eric and Cora, Vlad~M. and {de~Freitas}, Nando",
    //    title       = "A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Applications to Active User Modeling and Heirarchical Reinforcement Learning",
    //    institution = "{arXiv.org}",
    //    month       = "December",
    //    number      = "{arXiv:1012.2599}",
    //    type        = "eprint",
    //    year        = "2010"}
    //
    // gpUCB: in all cases sqrt(beta) = sqrt(nu.tau_t).
    //        tau_t depends on particular variant.
    //        d = dim(x).
    //
    // gpUCB basic:      tau_t = 2.log(t^{2+d/2}.pi^2/(3.delta))
    // gpUCB finite:     tau_t = 2.log(modD.t^2.pi^2/(6.delta))
    // gpUCB infinite:   tau_t = 2.log(t^2.2.pi^2/(3.delta)) + 4d.log(t^2.d.b.r.sqrt(log(4.d.a/delta)))
    // gpUCB p basic:    tau_t = 2.log(2.t^{d/2}.pi_t/delta)
    //                   pi_t = zeta(p).t^p
    // gpUCB p finite:   tau_t = 2.log(modD.pi_t/delta)
    //                   pi_t = zeta(p).t^p
    // gpUCB p infinite: tau_t = 2.log(4.pi_t/delta) + 4d.log(t^2.d.b.r.sqrt(log(4.d.a/delta)))
    //                   pi_t = zeta(p).t^p
    //
    // gpUCB p basic is inferred as follows.  Brochu states (without giving
    // working) that beta_t from Srinivas has the form of gpUCB basic.
    // Comparing this with gpUCB finite we see that they are equivalent if
    // we make the (unverified) assumption that |D_t| = 2.t^{d/2}.  This is
    // somewhat reminiscent of the proofs in Srinivas but does not match
    // exactly.  In any case, we note that gpUCB finite is just gpUCB p finite
    // with p = 2, so we generalise to get the expression above.
    //
    // Of course this is very speculative, but unfortunately Brochu's paper
    // entirely fails to report *where* their "bold" claim about tau_t (page
    // 16) is drawn from.  *Assuming* that Brochu has pulled this result
    // from somewhere sensible it seems reasonable to assume that the result
    // will follow.
    //
    //
    //
    // Multi-recommendation: choose method == 11 (gpUCB, user defined) and
    // betafn a vector of functions to select multi-recommendation.  Note
    // that in this case the number of start points added is s*n, where
    // s = startpoints and n = multi-recommendation batch size.
    //
    // To reference internal methods in multi-recommendation (that is, other
    // than method 11) replace the equation with a vector (where {} indicates
    // an optional element):
    //
    // [ method   ]
    // [ {p}      ]
    // [ {betafn} ]
    // [ {modD}   ]
    // [ {nu}     ]
    // [ {delta}  ]
    // [ {a}      ]
    // [ {b}      ]
    // [ {r}      ]
    //
    // (if you want to change an element but not some before it use [] in
    // place of the elements you don't want to change).
    //
    // For multi-objective based multi-recommendation use betafn = null
    // (or [ null null ... ] for multiple rounds of multi-objective multi-rec)

    int method;            // default 1 (EI)
    int intrinbatch;       // 
    int intrinbatchmethod; // 
    int sigmuseparate;     // 
    int startpoints;       // 
    int startseed;         //
    int algseed;           //
    int totiters;          // 
    double stepweight;     // 
    int itcntmethod;       // 
    int gridcache;         // 
    double err;            // 

    double ztol;    //
    double delta;   //
    double nu;      //
    double modD;    //
    double a;       //
    double b;       //
    double r;       //
    double p;       //
    gentype betafn; //

    IMP_Generic *impmeasu;   //
    ML_Base *direcpre;       //
    ML_Base *direcsubseqpre; //
    int direcdim;            //
    Vector<double> direcmin; //
    Vector<double> direcmax; //
    ML_Base *gridsource;     //

    Vector<ML_Base *> penalty; //

    // Stable optimisation

    int stabpmax;
    int stabpmin;
    int stabUseSig;
    double stabA;
    double stabB;
    double stabF;
    double stabbal;
    double stabZeroPt;
    double stabDelrRep;
    double stabDelRep;
    double stabThresh;

    // Unscented optimisation

    int unscentUse;
    int unscentK;
    Matrix<double> unscentSqrtSigma;

    // DIRect options (note that global options in this are over-ridden by *this, so only DIRect parts matter)

    DIRectOptions goptssingleobj;

    // Multi-objective, multi-recommendation part

    DIRectOptions goptsmultiobj; // full over-ride
    int startpointsmultiobj;
    int totitersmultiobj;
    int ehimethodmultiobj;

    BayesOptions(IMP_Generic *impmeasux = NULL, ML_Base *xdirecpre = NULL, int xdirecdim = 0, ML_Base *xdirecsubseqpre = NULL, ML_Base *xgridsource = NULL) : SMBOOptions()
    {
        method            = 1;
        intrinbatch       = 1;
        intrinbatchmethod = 0;
        sigmuseparate     = 0;
        startpoints       = -1; //5; //500; //10;
        startseed         = 42;
        algseed           = 69;
        totiters          = -1; //100; //200; //500;
        stepweight        = 0;
        itcntmethod       = 0;
        gridcache         = 1;
        err               = 1e-1;

        ztol   = DEFAULT_BAYES_ZTOL;
        delta  = DEFAULT_BAYES_DELTA;
        nu     = DEFAULT_BAYES_NU;
        modD   = -1; // this is entirely arbitrary and must be set by the user
        a      = DEFAULT_BAYES_A; // a value
        b      = DEFAULT_BAYES_B; // another value
        r      = DEFAULT_BAYES_R; // This is basically the width of our search region in
                                  // any given dimension.  Usually you would want to
                                  // normalise to 0->1, so 1 is correct.
        p      = DEFAULT_BAYES_P;
        betafn = 0;

        impmeasu       = impmeasux;
        direcpre       = xdirecpre;
        direcsubseqpre = xdirecsubseqpre;
        direcdim       = xdirecdim;
        direcmin.resize(direcdim);
        direcmax.resize(direcdim);

        gridsource = xgridsource;

        startpointsmultiobj = startpoints;
        totitersmultiobj    = totiters;
        ehimethodmultiobj   = 0;

        stabpmax    = DEFAULT_BAYES_STABPMAX;
        stabpmin    = DEFAULT_BAYES_STABPMIN;
        stabUseSig  = DEFAULT_BAYES_STABUSESIG;
        stabA       = DEFAULT_BAYES_STABA;
        stabB       = DEFAULT_BAYES_STABB;
        stabF       = DEFAULT_BAYES_STABF;
        stabbal     = DEFAULT_BAYES_STABBAL;
        stabZeroPt  = DEFAULT_BAYES_STABZEROPT;
        stabDelrRep = DEFAULT_BAYES_STABDELRREP;
        stabDelRep  = DEFAULT_BAYES_STABDELREP;
        stabThresh  = DEFAULT_BAYES_STABTHRESH;

        unscentUse = 0;
        unscentK   = 0;

        return;
    }

    BayesOptions(const BayesOptions &src) : SMBOOptions(src)
    {
        *this = src;

        return;
    }

    BayesOptions &operator=(const BayesOptions &src)
    {
        SMBOOptions::operator=(src);

        method            = src.method;
        intrinbatch       = src.intrinbatch;
        intrinbatchmethod = src.intrinbatchmethod;
        sigmuseparate     = src.sigmuseparate;
        startpoints       = src.startpoints;
        startseed         = src.startseed;
        algseed           = src.algseed;
        totiters          = src.totiters;
        stepweight        = src.stepweight;
        itcntmethod       = src.itcntmethod;
        gridcache         = src.gridcache;
        err               = src.err;

        ztol   = src.ztol;
        delta  = src.delta;
        nu     = src.nu;
        modD   = src.modD;
        a      = src.a;
        b      = src.b;
        r      = src.r;
        p      = src.p;
        betafn = src.betafn;

        impmeasu       = src.impmeasu;
        direcpre       = src.direcpre;
        direcsubseqpre = src.direcsubseqpre;
        direcdim       = src.direcdim;
        direcmin       = src.direcmin;
        direcmax       = src.direcmax;
        gridsource     = src.gridsource;

        penalty           = src.penalty;

        stabpmax    = src.stabpmax;
        stabpmin    = src.stabpmin;
        stabUseSig  = src.stabUseSig;
        stabA       = src.stabA;
        stabB       = src.stabB;
        stabF       = src.stabF;
        stabbal     = src.stabbal;
        stabZeroPt  = src.stabZeroPt;
        stabDelrRep = src.stabDelrRep;
        stabDelRep  = src.stabDelRep;
        stabThresh  = src.stabThresh;

        unscentUse       = src.unscentUse;
        unscentK         = src.unscentK;
        unscentSqrtSigma = src.unscentSqrtSigma;

        goptssingleobj = src.goptssingleobj;

        goptsmultiobj       = src.goptsmultiobj;
        startpointsmultiobj = src.startpointsmultiobj;
        totitersmultiobj    = src.totitersmultiobj;
        ehimethodmultiobj   = src.ehimethodmultiobj;

        return *this;
    }

    // Generate a copy of the relevant optimisation class.

    virtual GlobalOptions *makeDup(void) const
    {
        BayesOptions *newver;

        MEMNEW(newver,BayesOptions(*this));

        return newver;
    }

    // supres: [ see .cc file ] for each evaluation of (*fn),
    //         where beta is the value used to find the point being
    //         evaluated (zero for initial startpoint block)

    virtual int optim(int dim,
                      Vector<gentype> &xres,
                      gentype &fres,
                      int &ires,
                      Vector<Vector<gentype> > &allxres,
                      Vector<gentype> &allfres,
                      Vector<gentype> &allfresmod,
                      Vector<gentype> &supres,
                      Vector<double> &sscore,
                      const Vector<gentype> &xmin,
                      const Vector<gentype> &xmax,
                      void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                      void *fnarg,
                      svmvolatile int &killSwitch);

    virtual int optim(int dim,
                      Vector<gentype> &xres,
                      Vector<gentype> &Xres,
                      gentype &fres,
                      int &ires,
                      int &mInd,
                      int &muInd,
                      int &sigInd,
                      int &srcmodInd,
                      int &diffmodInd,
                      Vector<Vector<gentype> > &allxres,
                      Vector<Vector<gentype> > &allXres,
                      Vector<gentype> &allfres,
                      Vector<gentype> &allmres,
                      Vector<gentype> &allsres,
                      Vector<double>  &s_score,
                      const Vector<gentype> &xmin,
                      const Vector<gentype> &xmax,
                      const Vector<int> &distMode,
                      const Vector<int> &varsType,
                      void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                      void *fnarg,
                      svmvolatile int &killSwitch,
                      unsigned int numReps, 
                      gentype &meanfres, gentype &varfres,
                      gentype &meanires, gentype &varires,
                      gentype &meantres, gentype &vartres,
                      gentype &meanTres, gentype &varTres,
                      Vector<gentype> &meanallfres, Vector<gentype> &varallfres,
                      Vector<gentype> &meanallmres, Vector<gentype> &varallmres)
    {
        int res = SMBOOptions::optim(dim,xres,Xres,fres,ires,mInd,muInd,sigInd,srcmodInd,diffmodInd,allxres,allXres,allfres,allmres,allsres,s_score,xmin,xmax,distMode,varsType,fn,fnarg,killSwitch,numReps,meanfres,varfres,meanires,varires,meantres,vartres,meanTres,varTres,meanallfres,varallfres,meanallmres,varallmres);

        return res;
    }

    // IMP use

    int isimphere(void) const { return impmeasu ? 1 : 0; }

    int modelimp_imp(gentype &resi, const SparseVector<gentype> &xxmean, const gentype &xxvar) const 
    { 
        NiceAssert( impmeasu ); 

        return (*impmeasu).imp(resi,xxmean,xxvar); 
    }

    int modelimp_N(void) const 
    { 
        NiceAssert( impmeasu ); 

        return (*impmeasu).N(); 
    }

    int modelimp_addTrainingVector(const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1)
    {
        NiceAssert( impmeasu );

        int i = modelimp_N();

        return (*impmeasu).addTrainingVector(i,y,x,Cweigh,epsweigh);
    }

    int modelimp_train(int &res, svmvolatile int &killSwitch) 
    { 
        int ires = 0;

        if ( isimphere() )
        {
            ires = (*impmeasu).train(res,killSwitch); 
        }

        return ires;
    }



    virtual int optdefed(void)
    {
        return 3;
    }

    int impmeasuNonLocal(void) const
    {
        return impmeasu != NULL;
    }

    int direcpreDef(void) const
    {
        return direcpre != NULL;
    }

    int direcsubseqpreDef(void) const
    {
        return direcsubseqpre != NULL;
    }
};

#endif
