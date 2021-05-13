
//
// Bayesian Optimiser
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "bayesopt.h"
#include "ml_mutable.h"

#define FEEDBACK_CYCLE 50
#define MAJOR_FEEDBACK_CYCLE 1000

#define DISCOUNTRATE 1e-6

//
// Notes:
//
// - Max/min decisions:
//   o Code was originally written to maximise function fn.
//   o All expressions for EI, PI etc were written with this in mind.
//   o To bring it in line with DIRect I decided to change this to min.
//   o Rather than re-write various negations have been introduced.
// - Returning beta:
//   o We need to return beta in supres.
//   o To do this we need (a) a variant of the direct call (a function
//     passed to the direct optimiser for evaluation) that just returns beta,
//     and (b) a means of passing this up to the next level.
//   o We do (a) using the variable "justreturnbeta"
//   o We do (b) by hiding beta at the end of the x vector which is passed
//     to fninner for evaluation of the actual function being minimised.
//   o We use a slightly ugly trick of morphing a sparse vector that is
//     actually not sparse (ie sparse in name only) into a double * by
//     dereferencing the first element in that vector.  It works because
//     reasons.



int bayesOpt(int dim,
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
             BayesOptions &bopts,
             svmvolatile int &killSwitch);

// The term addvar here is additional variance (on top of whatever is assumed 
// by the GPR (or similar) model).  This is included as a sigma scaling factor
// when updating the model.  Set to zero for no additional variance.

int bayesOpt(int dim,
             Vector<double> &xres,
             gentype &fres,
             const Vector<double> &xmin,
             const Vector<double> &xmax,
             void (*fn)(int n, gentype &res, const double *x, void *arg, double &addvar),
             void *fnarg,
             BayesOptions &bopts,
             svmvolatile int &killSwitch,
             Vector<double> &sscore);



int BayesOptions::optim(int dim,
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
                      svmvolatile int &killSwitch)
{
    return bayesOpt(dim,xres,fres,ires,allxres,allfres,allfresmod,supres,sscore,
                        xmin,xmax,fn,fnarg,*this,killSwitch);
}














//inline double Phifn(double z);
//inline double phifn(double z);
//
//inline double Phifn(double z)
//{
//    double res;
//
//    return 0.5 + (0.5*erf(z*NUMBASE_SQRT1ON2));
//}
//
//inline double phifn(double z)
//{
//    return exp(-z*z/2)/2.506628;
//}









void calcsscore(Vector<double> &sscore, const BayesOptions &bopts, const Vector<int> &xdatind, int stabp, double stabpnrm, int stabrot, double stabmu, double stabB);
void calcsscore(Vector<double> &sscore, const BayesOptions &bopts, const Vector<int> &xdatind, int stabp, double stabpnrm, int stabrot, double stabmu, double stabB)
{
    NiceAssert( sscore.size() == xdatind.size() );

    int j;

    for ( j = 0 ; j < sscore.size() ; j++ )
    {
        bopts.model_stabProbTrainingVector(sscore("&",j),xdatind(j),stabp,stabpnrm,stabrot,stabmu,stabB);
    }

    return;
}









class fninnerinnerArg
{
    public:

    fninnerinnerArg(BayesOptions &_bbopts,
                    SparseVector<gentype> &_x,
                    gentype &_muy,
                    gentype &_sigmay,
                    gentype &_ymax,
                    const unsigned int &_iters,
                    const int &_dim,
                    const int &_effdim,
                    const double &_ztol,
                    const double &_delta,
                    const double &_nu,
                    gentype &_ires,
                    SparseVector<double> &_prevx,
                    SparseVector<double> &_diffx,
                    const double &_a,
                    const double &_b,
                    const double &_r,
                    const double &_p,
                    const double &_modD,
                    gentype &_betafn,
                    const int &_locmethod,
                    const double &_stepweight,
                    const int &_justreturnbeta,
                    Matrix<gentype> &_covarmatrix,
                    Vector<gentype> &_meanvector,
                    const int &_thisbatchsize,
                    Vector<SparseVector<gentype> > &_multivecin,
                    const ML_Base *_direcpre,
                    gentype &_xytemp,
                    const int &_thisbatchmethod,
                    SparseVector<gentype> &_xappend,
                    const int &_anyindirect,
                    const double &_softmax,
                    const int &_itcntmethod,
                    const int &_itinbatch,
                    const Vector<ML_Base *> &_penalty,
                    gentype &_locpen,
                    const vecInfo **_xinf,
                    const int &_Nbasemu,
                    const int &_Nbasesigma,
                    const int &_gridi,
                    const int &_isgridopt,
                    const int &_isgridcache,
                    const int &_isstable,
                    Vector<int> &_ysort,
                    const int &_stabp,
                    const double &_stabpnrm,
                    const int &_stabrot,
                    const double &_stabmu,
                    const double &_stabB,
                    Vector<double> &_sscore,
                    int &_firstevalinseq,
                    const double &_stabZeroPt,
                    const int &_unscentUse,
                    const int &_unscentK,
                    const Matrix<double> &_unscentSqrtSigma,
                    const int &_stabUseSig,
                    const double &_stabThresh) : bbopts(_bbopts),
                                                __x(&_x),
                                                muy(_muy),
                                                sigmay(_sigmay),
                                                ymax(_ymax),
                                                iters(_iters),
                                                dim(_dim),
                                                effdim(_effdim),
                                                ztol(_ztol),
                                                __delta(_delta),
                                                __nu(_nu),
                                                ires(_ires),
                                                __prevx(&_prevx),
                                                diffx(_diffx),
                                                __a(_a),
                                                __b(_b),
                                                __r(_r),
                                                __p(_p),
                                                __modD(_modD),
                                                __betafn(&_betafn),
                                                __locmethod(_locmethod),
                                                stepweight(_stepweight),
                                                justreturnbeta(_justreturnbeta),
                                                covarmatrix(_covarmatrix),
                                                meanvector(_meanvector),
                                                __thisbatchsize(_thisbatchsize),
                                                multivecin(_multivecin),
                                                direcpre(_direcpre),
                                                xytemp(_xytemp),
                                                __thisbatchmethod(_thisbatchmethod),
                                                xappend(_xappend),
                                                anyindirect(_anyindirect),
                                                softmax(_softmax),
                                                itcntmethod(_itcntmethod),
                                                itinbatch(_itinbatch),
                                                penalty(_penalty),
                                                locpen(_locpen),
                                                xinf(_xinf),
                                                Nbasemu(_Nbasemu),
                                                Nbasesigma(_Nbasesigma),
                                                __gridi(_gridi),
                                                isgridopt(_isgridopt),
                                                isgridcache(_isgridcache),
                                                isstable(_isstable),
                                                ysort(_ysort),
                                                stabp(_stabp),
                                                stabpnrm(_stabpnrm),
                                                stabrot(_stabrot),
                                                stabmu(_stabmu),
                                                stabB(_stabB),
                                                sscore(_sscore),
                                                firstevalinseq(_firstevalinseq),
                                                stabZeroPt(_stabZeroPt),
                                                unscentUse(_unscentUse),
                                                unscentK(_unscentK),
                                                unscentSqrtSigma(_unscentSqrtSigma),
                                                stabUseSig(_stabUseSig),
                                                stabThresh(_stabThresh)
    {
        return;
    }

    fninnerinnerArg(const fninnerinnerArg &src) : bbopts(src.bbopts),
                                                __x(src.__x),
                                                muy(src.muy),
                                                sigmay(src.sigmay),
                                                ymax(src.ymax),
                                                iters(src.iters),
                                                dim(src.dim),
                                                effdim(src.effdim),
                                                ztol(src.ztol),
                                                __delta(src.__delta),
                                                __nu(src.__nu),
                                                ires(src.ires),
                                                __prevx(src.__prevx),
                                                diffx(src.diffx),
                                                __a(src.__a),
                                                __b(src.__b),
                                                __r(src.__r),
                                                __p(src.__p),
                                                __modD(src.__modD),
                                                __betafn(src.__betafn),
                                                __locmethod(src.__locmethod),
                                                stepweight(src.stepweight),
                                                justreturnbeta(src.justreturnbeta),
                                                covarmatrix(src.covarmatrix),
                                                meanvector(src.meanvector),
                                                __thisbatchsize(src.__thisbatchsize),
                                                multivecin(src.multivecin),
                                                direcpre(src.direcpre),
                                                xytemp(src.xytemp),
                                                __thisbatchmethod(src.__thisbatchmethod),
                                                xappend(src.xappend),
                                                anyindirect(src.anyindirect),
                                                softmax(src.softmax),
                                                itcntmethod(src.itcntmethod),
                                                itinbatch(src.itinbatch),
                                                penalty(src.penalty),
                                                locpen(src.locpen),
                                                xinf(src.xinf),
                                                Nbasemu(src.Nbasemu),
                                                Nbasesigma(src.Nbasesigma),
                                                __gridi(src.__gridi),
                                                isgridopt(src.isgridopt),
                                                isgridcache(src.isgridcache),
                                                isstable(src.isstable),
                                                ysort(src.ysort),
                                                stabp(src.stabp),
                                                stabpnrm(src.stabpnrm),
                                                stabrot(src.stabrot),
                                                stabmu(src.stabmu),
                                                stabB(src.stabB),
                                                sscore(src.sscore),
                                                firstevalinseq(src.firstevalinseq),
                                                stabZeroPt(src.stabZeroPt),
                                                unscentUse(src.unscentUse),
                                                unscentK(src.unscentK),
                                                unscentSqrtSigma(src.unscentSqrtSigma),
                                                stabUseSig(src.stabUseSig),
                                                stabThresh(src.stabThresh)
    {
        throw("Can't use copy constructer on fninnerinnerArg");
        return;
    }

    fninnerinnerArg &operator=(const fninnerinnerArg &src)
    {
        (void) src;
        throw("Can't copy fninnerinnerArg");
        return *this;
    }

    BayesOptions &bbopts;
    SparseVector<gentype> *__x;
    gentype &muy;
    gentype &sigmay;
    gentype &ymax;
    const unsigned int &iters;
    const int &dim;
    const int &effdim;
    const double &ztol;
    double __delta;
    double __nu;
    gentype &ires;
    SparseVector<double> *__prevx;
    SparseVector<double> &diffx;
    double __a;
    double __b;
    double __r;
    double __p;
    double __modD;
    gentype *__betafn;
    int __locmethod;
    const double &stepweight;
    const int &justreturnbeta;
    Matrix<gentype> &covarmatrix;
    Vector<gentype> &meanvector;
    int __thisbatchsize;
    Vector<SparseVector<gentype> > &multivecin;
    const ML_Base *direcpre;
    gentype &xytemp;
    int __thisbatchmethod;
    SparseVector<gentype> &xappend;
    const int &anyindirect;
    const double &softmax;
    const int &itcntmethod;
    const int &itinbatch;
    const Vector<ML_Base *> &penalty;
    gentype &locpen;
    const vecInfo **xinf;
    const int &Nbasemu;
    const int &Nbasesigma;
    int __gridi;
    const int &isgridopt;
    const int &isgridcache;
    const int &isstable;
    Vector<int> &ysort;
    const int &stabp;
    const double &stabpnrm;
    const int &stabrot;
    const double &stabmu;
    const double &stabB;
    Vector<double> &sscore;
    int &firstevalinseq;
    const double &stabZeroPt;
    const int &unscentUse;
    const int &unscentK;
    const Matrix<double> &unscentSqrtSigma;
    const int &stabUseSig;
    const double &stabThresh;

    double fnfnapprox(int n, const double *xx)
    {
        // Outer loop of unscented optimisation

        double res = 0;

        res = fnfnapproxNoUnscent(n,xx);

        if ( unscentUse )
        {
            NiceAssert( n == unscentSqrtSigma.numRows() );
            NiceAssert( n == unscentSqrtSigma.numCols() );

            res *= ((double) unscentK)/((double) (n+unscentK));

            double *xxx;

            MEMNEWARRAY(xxx,double,n);

            int i,j;
            double temp;

            for ( i = 0 ; i < n ; i++ )
            {
                for ( j = 0 ; j < n ; j++ )
                {
                    xxx[j] = xx[j] + sqrt(n+unscentK)*unscentSqrtSigma(i,j);
                }

                temp = fnfnapproxNoUnscent(n,xxx);
                res += temp/((double) (2*(n+unscentK)));

                for ( j = 0 ; j < n ; j++ )
                {
                    xxx[j] = xx[j] - sqrt(n+unscentK)*unscentSqrtSigma(i,j);
                }

                temp = fnfnapproxNoUnscent(n,xxx);
                res += temp/((double) (2*(n+unscentK)));
            }

            MEMDELARRAY(xxx);
        }

        return res;
    }

  double fnfnapproxNoUnscent(int n, const double *xx)
  {
errstream() << ".";
    double delta = __delta;
    double nu = __nu;
    double a = __a;
    double b = __b;
    double r = __r;
    double p = __p;
    double modD = __modD;

    int locmethod = __locmethod;
    int thisbatchsize = __thisbatchsize;
    int thisbatchmethod = __thisbatchmethod;
    int gridi = __gridi;

    SparseVector<gentype> &x = *__x;
    SparseVector<double> &prevx = *__prevx;
    gentype &betafn = *__betafn;


    // NB: n != dim in general (in fact n = dim*multivecin)

    (void) softmax;
    (void) unscentUse;
    (void) unscentK;
    (void) unscentSqrtSigma;

    int i,j;

    // =======================================================================
    // First work out "beta"
    // =======================================================================

    double beta = 0;
    int epspinf = 0;
    int betasgn = ( locmethod >= 0 ) ? 1 : -1;
    int method = betasgn*locmethod;

    //if ( !(bopts.isimphere()) ) - work out beta anyhow
    {
        double altiters = ( itcntmethod == 2 ) ? iters-itinbatch : iters; //+1;
        double d = (double) dim;
        double eps = 0;
        double locnu = 1; // nu (beta scale) only applied to GP-UCB.

        switch ( method )
        {
            case 0:
            {
                eps = 0;

                break;
            }

            case 1:
            {
//errstream() << "EIEIEEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIE\n";
                // EI

                eps = 0; // beta ill-defined for this case

                break;
            }

            case 2:
            {
                // PI

                eps = 0; // beta ill-defined for this case

                break;
            }

            case 3:
            {
//errstream() << "GPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGP\n";
                // gpUCB basic

                eps = 2*log(pow(altiters,(2+(d/2)))*(NUMBASE_PI*NUMBASE_PI/(3*delta)));

                locnu = nu;

                break;
            }

            case 4:
            {
                // gpUCB finite

                eps = 2*log(modD*pow(altiters,2)*(NUMBASE_PI*NUMBASE_PI/(6*delta)));

                locnu = nu;

                break;
            }

            case 5:
            {
                // gpUCB infinite

                eps = (2*log(pow(altiters,2)*(2*NUMBASE_PI*NUMBASE_PI/(3*delta))))
                    + (2*thisbatchsize*d*log(((thisbatchsize==1)?1.0:2.0)*pow(altiters,2)*d*b*r*sqrt(log(4*thisbatchsize*d*a/delta))));

                locnu = nu;

                break;
            }

            case 6:
            {
                // gpUCB p basic

                double pi_t = 0;//  = numbase_zeta(p)*pow(altiters,p);
                double modDt = 2*pow(sqrt(altiters),d);

                numbase_zeta(pi_t,p);
                pi_t *= pow(altiters,p);

                eps = 2*log(modDt*pi_t/delta);

                locnu = nu;

                break;
            }

            case 7:
            {
                // gpUCB p finite

                double pi_t = 0;//  = numbase_zeta(p)*pow(altiters,p);

                numbase_zeta(pi_t,p);
                pi_t *= pow(altiters,p);

                eps = 2*log(modD*pi_t/delta);

                locnu = nu;

                break;
            }

            case 8:
            {
                // gpUCB p infinite

                double pi_t = 0;//  = numbase_zeta(p)*pow(altiters,p);

                numbase_zeta(pi_t,p);
                pi_t *= pow(altiters,p);

                eps = (2*log(4*pi_t/delta))
                    + (2*thisbatchsize*d*log(((thisbatchsize==1)?1.0:2.0)*pow(altiters,2)*d*b*r*sqrt(log(4*thisbatchsize*d*a/delta))));

                locnu = nu;

                break;
            }

            case 9:
            {
                // VO

                eps = valpinf();
                epspinf = 1;

                break;
            }

            case 10:
            {
                // MO

                eps = 0;

                break;
            }

            case 11:
            {
                // gpUCB user defined

                gentype teval(altiters);
                gentype deval((double) dim);
                gentype deltaeval(delta);
                gentype modDeval(modD);
                gentype aeval(a);

                eps = (double) betafn(teval,deval,deltaeval,modDeval,aeval);

                break;
            }

            default:
            {
                eps = 0;

                break;
            }
        }

        beta = locnu*eps;
    }

    if ( justreturnbeta == 1 )
    {
        return beta;
    }

    // =======================================================================
    // Re-express function input as sparse vector of gentype (size n)
    // =======================================================================

    x.zero();

    for ( i = 0 ; i < n ; i++ )
    {
        x("&",i) = xx[i]; // gentype sparsevector
    }

    // =======================================================================
    // Add vector to end of vector if required
    // =======================================================================
    
    if ( xappend.size() )
    {
        for ( i = n ; i < n+xappend.size() ; i++ )
        {
            x("&",i) = xappend(i-n);
        }
    }

    // =======================================================================
    // Pre-process DIRect input if required
    // =======================================================================

    if ( anyindirect )
    {
        (*direcpre).gg(xytemp,x,*xinf);
        x = (const Vector<gentype> &) xytemp;

        n = xytemp.size(); // This is important.  dim cannot be used as it means something else.
    }

    // =======================================================================
    // Calculate step cost if required
    //
    // stepcost = -stepweight.||prevx-x||_2
    // (negative as we want to minimise the step cost, which is
    //  equivalent to maximising the negative of the step cost)
    // =======================================================================

    double stepcost = 0;

    if ( stepweight > 0 )
    {
        diffx.zero();

        for ( i = 0 ; i < n ; i++ )
        {
            diffx("&",i) = (double) x(i);
        }

        diffx -= prevx;     
        stepcost = -stepweight*abs2(diffx);

        for ( i = 0 ; i < n ; i++ )
        {
            prevx("&",i) = x(i);
        }
    }

    // =======================================================================
    // Calculate output mean and variance
    // (Note that muy can be a vector (multi-objective optim) or 
    //  a scalar (regular optimiser))
    // =======================================================================

    NiceAssert( thisbatchsize >= 1 );

    if ( thisbatchsize == 1 )
    {
        if ( isgridopt && isgridcache && ( gridi >= 0 ) )
        {
            if ( ( method != 0 ) && ( method != 10 ) )
            {
                bbopts.model_muvarTrainingVector(sigmay,muy,Nbasesigma+gridi,Nbasemu+gridi);
//errstream() << "phantomxyq 4a: muy(b) = " << muy << "\n";
//errstream() << "phantomxyq 4a: sigmay(b)^2 = " << sigmay << "\n";

                OP_sqrt(sigmay);
            }

            else
            {
//errstream() << "phantomxyq 2\n";
                bbopts.model_ggTrainingVector(muy,Nbasemu+gridi);

                sigmay = 0.0;
            }
        }

        else
        {
            if ( ( method != 0 ) && ( method != 10 ) )
            {
                bbopts.model_muvar(sigmay,muy,x,*xinf);
//errstream() << "phantomxyq 4b: muy(b) = " << muy << "\n";

                OP_sqrt(sigmay);
            }

            else
            {
//errstream() << "phantomxyq 4d\n";
                bbopts.model_gg(muy,x,*xinf);

                sigmay = 0.0;
            }
        }
//errstream() << "phantomxyq 4z\n";
    }

    else
    {
        // x(given above) = [ v0, v1, ..., vthisbatchsize-1 ], where xi has dimension
        // dim.  Need to split these up and calculate mean vector and covariance matrix,
        // then reform them to get the min mean vector and modified determinat of the
        // covariance matrix (the geometric mean of the determinants).

        meanvector.resize(thisbatchsize);
        covarmatrix.resize(thisbatchsize,thisbatchsize);
        multivecin.resize(thisbatchsize);

        for ( i = 0 ; i < thisbatchsize ; i++ )
        {
            multivecin("&",i).zero();

            for ( j = 0 ; j < dim ; j++ )
            {
                multivecin("&",i)("&",j) = x((i*dim)+j);
            }

//errstream() << "phantomxyq 4e\n";
            bbopts.model_gg(meanvector("&",i),multivecin(i),NULL);
        }

        bbopts.model_covar(covarmatrix,multivecin);

        switch ( thisbatchmethod )
        {
            case 1:
            {
                muy = mean(meanvector);
                sigmay = pow((double) covarmatrix.det(),1/(2.0*thisbatchsize));
                
                break;
            }

            case 2:
            {
                muy = min(meanvector,i);
                sigmay = pow((double) covarmatrix.det(),1/(2.0*thisbatchsize));
                
                break;
            }

            case 3:
            {
                muy = max(meanvector,i);
                sigmay = sqrt(thisbatchsize*((double) covarmatrix.invtrace()));
                
                break;
            }

            case 4:
            {
                muy = mean(meanvector);
                sigmay = sqrt(thisbatchsize*((double) covarmatrix.invtrace()));
                
                break;
            }

            case 5:
            {
                muy = min(meanvector,i);
                sigmay = sqrt(thisbatchsize*((double) covarmatrix.invtrace()));
                
                break;
            }

            default:
            {
                muy = max(meanvector,i);
                sigmay = pow((double) covarmatrix.det(),1/(2.0*thisbatchsize));
                
                break;
            }
        }
    }

//errstream() << "phantomxyq 5: mu = " << muy << "\n";
//errstream() << "phantomxyq 5: sigma = " << sigmay << "\n";
    if ( testisvnan((double) sigmay) || testisinf((double) sigmay) )
    {
        sigmay = 0.0;
    }

    double stabscore = 1.0;

    if ( isstable )
    {
        // =======================================================================
        // Calculate stability scores
        // =======================================================================

        // Calculate stability score on x

        if ( !( isgridopt && isgridcache && ( gridi >= 0 ) ) )
        {
            bbopts.model_stabProb(stabscore,x,stabp,stabpnrm,stabrot,stabmu,stabB);
        }

        else
        {
            bbopts.model_stabProbTrainingVector(stabscore,Nbasemu+gridi,stabp,stabpnrm,stabrot,stabmu,stabB);
        }

        if ( stabUseSig )
        {
//stabscore = ( stabscore >= stabThresh ) ? 1.0 : DISCOUNTRATE;
stabscore = 1/(1+exp(-1000*(stabscore-stabThresh)));
//            stabscore = 1/(1+exp(-(stabscore-stabThresh)/(stabscore*(1-stabscore))));
        }

        if ( firstevalinseq && ( method == 1 ) )
        {
            firstevalinseq = 0;

            // ...then the stability score on x(j)...

            sscore.resize(ysort.size()+1); // This will eventually become the products
            sscore("&",ysort.size()) = 0.0;

            retVector<double> tmpva;

            calcsscore(sscore("&",zeroint(),1,ysort.size()-1,tmpva),bbopts,ysort,stabp,stabpnrm,stabrot,stabmu,stabB);

            // ...and finally convert stability scores to combined stability scores.

            sscore *= -1.0;
            sscore += 1.0;

            for ( j = sscore.size()-1 ; j >= 1 ; j-- )
            {
                sscore("&",0,1,j-1,tmpva) *= sscore(j);
            }

            if ( stabUseSig )
            {
                for ( j = 0 ; j < sscore.size() ; j++ )
                {
//sscore("&",j) = ( sscore(j) >= stabThresh ) ? 1.0 : DISCOUNTRATE;
sscore("&",j) = 1/(1+exp(-1000*(sscore(j)-stabThresh)));
//                    sscore("&",j) = 1/(1+exp(-(sscore(j)-stabThresh)/(sscore(j)*(1-sscore(j)))));
                }
            }
//errstream() << "phantomx 0: " << sscore << "\n";
        }
//errstream() << "phantomx 1: " << stabscore << "\t-\t";
    }

    // =======================================================================
    // Work out improvement measure
    // =======================================================================

    double res = 0;

    if ( bbopts.isimphere() )
    {
        // muy may be a vector in this case!

        muy.negate();

        SparseVector<gentype> xmean;

        if ( muy.isValVector() )
        {
            int j;

            const Vector<gentype> &ghgh = (const Vector<gentype> &) muy;

            for ( j = 0 ; j < muy.size() ; j++ )
            {
                xmean("&",j) = ghgh(j);
            }
        }

        else
        {
            xmean("&",zeroint()) = muy;
        }

        bbopts.modelimp_imp(ires,xmean,sigmay); // IMP may or may not update sigmay
        res = -((double) ires);

        muy.force_double() = res; // This is done so that passthrough will apply to IMP
    }

    //else - do this anyhow to allow for standard GP-UCB.  Method 0 does passthrough of IMP, other methods will change things around
    {
        NiceAssert( !(muy.isValVector()) );

        switch ( method )
        {
            case 1:
            {
//errstream() << "EIEIEEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIEIE\n";
                // EI

                if ( isstable )
                {
                    // BIG ASSUMPTION: sigmay > ztol

                    res = 0.0;

                    double parta = 0;
                    double partb = 0;
                    double partadec = 0;
                    double partbdec = 0;
                    double scalea,scaleb;

                    int k;

                    // Range 0 to N+1
                    for ( k = 0 ; k <= ysort.size() ; k++ )
                    {
                        double dmuy    = (double) muy;
                        double dsigy   = (double) sigmay;

                        double ykdec = ( k == 0 ) ? stabZeroPt : ((double) (bbopts.modelmu_y())(ysort(k-1)));
                        double yk    = ( k == ysort.size() ) ? 1e12 : ((double) (bbopts.modelmu_y())(ysort(k))); // 1e12 is a placeholder for bugnum.  It is never actually used

                        //parta = phifn((dmuy-ykdec)/dsigy) - ( ( k == ysort.size() ) ? 0.0 : phifn((dmuy-yk)/dsigy) );
                        //partb = Phifn((dmuy-ykdec)/dsigy) - ( ( k == ysort.size() ) ? 0.0 : Phifn((dmuy-yk)/dsigy) );

                        numbase_phi(parta,(dmuy-yk)/dsigy);
                        numbase_Phi(partb,(dmuy-yk)/dsigy);

                        numbase_phi(partadec,(dmuy-ykdec)/dsigy);
                        numbase_Phi(partbdec,(dmuy-ykdec)/dsigy);

                        parta = partadec - ( ( k == ysort.size() ) ? 0.0 : parta );
                        partb = partbdec - ( ( k == ysort.size() ) ? 0.0 : partb );

                        scalea = sscore(k);
                        scaleb = sscore(k)*(dmuy-ykdec)/dsigy;

                        for ( i = 0 ; i < k-1 ; i++ )
                        {
                            double yidec = ( i == 0 ) ? stabZeroPt : ((double) (bbopts.modelmu_y())(ysort(i-1)));
                            double yi    = ( i == ysort.size() ) ? 1e12 : ((double) (bbopts.modelmu_y())(ysort(i))); // i == ysort.size() never attained.

                            scaleb += sscore(i)*(yi-yidec)/dsigy;
                        }

                        res += (parta*scalea)+(partb*scaleb);
                    }

                    res *= ((double) sigmay)*stabscore;
//errstream() << "phantomx 2: " << res << "\n";
                }

                else
                {
                    if ( (double) sigmay > ztol )
                    {
                        double z = ( ( (double) muy ) - ( (double) ymax ) ) / ( (double) sigmay );

                        //double Phiz = Phifn(z);
                        //double phiz = phifn(z);

                        double Phiz = 0;
                        double phiz = 0;

                        numbase_Phi(Phiz,z);
                        numbase_phi(phiz,z);

                        res = ( ( ( (double) muy ) - ( (double) ymax ) ) * Phiz )
                            + ( ( (double) sigmay ) * phiz );
//errstream() << "phantomx 2a: " << res << " (" << muy << "," << ymax << ";" << sigmay << " - " << z << "," << Phiz << "," << phiz << ")\n";
                    }

                    else
                    {
                        // if muy > ymax then z = +infty, so Phiz = +1, phiz = 0
                        // if muy < ymax then z = -infty, so Phiz = 0,  phiz = infty
                        // assume lim_{z->-infty} sigmay.phiz = 0

                        if ( (double) muy > (double) ymax )
                        {
                            res = ( (double) muy ) - ( (double) ymax );
                        }

                        else
                        {
                            res = 0;
                        }
//errstream() << "phantomx 2b: " << res << "\n";
                    }
                }

                break;
            }

            case 2:
            {
                // PI
        
                NiceAssert( !isstable );

                if ( (double) sigmay > ztol )
                {
                    double z = ( ( (double) muy ) - ( (double) ymax ) ) / ( (double) sigmay );

                    //double Phiz = Phifn(z);

                    double Phiz = 0;

                    numbase_Phi(Phiz,z);

                    res = Phiz;
                }

                else
                {
                    if ( (double) muy > (double) ymax )
                    {
                        res = 1;
                    }

                    else
                    {
                        res = 0;
                    }
                }

                break;
            }

            default:
            {
//errstream() << "GPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGPGP\n";
                // gpUCB variants, VO and MO, default method

                //NiceAssert( !isstable );

                double locsigmay = (double) sigmay;
                double locmuy = (double) muy;

                //FIXME: assuming binomial distribution here, may not be valid

//errstream() << "Stab sum: ";
                if ( isstable && !stabUseSig )
                {
                    // For independent vars: var(x.y) = var(x).var(y) + var(x).mu(y).mu(y) + var(y).mu(x).mu(x)

//                    double stabvar = stabscore*(1-stabscore);
//errstream() << stabvar << "\t" << stabscore << "\t" << locmuy << "\t" << locsigmay << "\t";

//THIS REALLY DOESNT WORK AT ALL!  MAKES VARIANCE LARGE EVERYWHERE, WHICH HINDERS EXPLORATION                    locsigmay = (locsigmay*stabvar) + (locsigmay*stabscore*stabscore) + (locmuy*locmuy*stabvar);
                    locmuy    = stabscore*locmuy;
//phantomx - new variance calc
locsigmay = stabscore*stabscore*locsigmay;
//errstream() << "(\t" << locmuy << "\t" << locsigmay << "\t)\t";
                }

                //if ( beta >= 1/ztol )
                if ( epspinf )
                {
                    res = betasgn*locsigmay;
                }

                else
                {
                    res = locmuy + ( betasgn*sqrt(beta) * locsigmay );
                }

                if ( isstable && stabUseSig )
                {
                    res *= stabscore;
                }
//errstream() << beta << "\t" << res << "\n";

                break;
            }
        }
    }
//errstream() << "mu + beta.sigma = " << muy << " + " << beta << "*" << sigmay << " = " << res << "\n"; // phantomx

    // =======================================================================
    // Add penalties
    // =======================================================================

    if ( penalty.size() )
    {
        for ( i = 0 ; i < penalty.size() ; i++ )
        {
            (*(penalty(i))).gg(locpen,x,*xinf);
            res -= (double) locpen;
        }
    }

    // =======================================================================
    // Negate on return (DIRect minimises, Bayesian Optimisation maximises -
    // but keep in mind note about max/min changes).
    // =======================================================================

    res = -((1+stepcost)*res); // Set up for minimiser
//errstream() << beta << "\t" << res << "\t" << muy << "\t" << sigmay << "\t" << x << "\n";

    return res;
  }
};



// fnfnapprox: callback for DIRect optimiser.
// fnfnfnapprox: has mean (to be maxed) in res[0], variance in res[1]

double fnfnapprox(int n, const double *xx, void *arg)
{
    return (*((fninnerinnerArg *) arg)).fnfnapprox(n,xx);
}

double fnfnapproxNoUnscent(int n, const double *xx, void *arg)
{
    return (*((fninnerinnerArg *) arg)).fnfnapproxNoUnscent(n,xx);
}





















// Alternative "grid" minimiser: takes all x indexed by vector in some
// model, tests them, then returns the index (and evaluation) of the 
// minimum.  This is a drop-in replacement for directOpt when optimising
// on a grid, except:
//
// - has additional ires argument to return index of gridi vector result
// - has additional gridires argument to return index of x vector result
// - has no xmin/xmax arguments as they are meaningless here
// - takes gridsource (source of grid data_ and gridind (which grid 
//   elements are as yet untested) arguments.

int dogridOpt(int dim,
              Vector<double> &xres,
              gentype &fres,
              int &ires,
              int &gridires,
              double (*fn)(int n, const double *x, void *arg),
              void *fnarg,
              const BayesOptions &dopts,
              ML_Base &gridsource,
              Vector<int> &gridind,
              svmvolatile int &force_stop,
              double xmtrtime);


int dogridOpt(int dim,
              Vector<double> &xres,
              gentype &fres,
              int &ires,
              int &gridires,
              double (*fn)(int n, const double *x, void *arg),
              void *fnarg,
              const BayesOptions &dopts,
              ML_Base &gridsource,
              Vector<int> &gridind,
              svmvolatile int &force_stop,
              double xmtrtime)
{
    NiceAssert( dim > 0 );
    NiceAssert( gridind.size() > 0 );

    const vecInfo **xinf = (*((fninnerinnerArg *) fnarg)).xinf;
    int &gridi = (*((fninnerinnerArg *) fnarg)).__gridi;

    double hardmin = dopts.hardmin;
    double hardmax = dopts.hardmax;
    double tempfres;

    xres.resize(dim);

    double *x = &xres("&",zeroint());
    double *xx;

    const vecInfo *xinfopt = NULL;

    MEMNEWARRAY(xx,double,dim);

    errstream() << "MLGrid Optimisation Initiated:\n";

    int i,j;
    int oldgridi = gridi;

    ires     = -1;
    gridires = -1;

    int timeout = 0;
    double *uservars[] = { &xmtrtime, NULL };
    const char *varnames[] = { "traintime", NULL };
    const char *vardescr[] = { "Maximum training time (seconds, 0 for unlimited)", NULL };
    time_used start_time = TIMECALL;
    time_used curr_time = start_time;

    for ( i = 0 ; ( i < gridind.size() ) && !force_stop && !timeout ; i++ )
    {
errstream() << "~";
        // This will propagate through fnarg[41], which will then be passed into function
        gridi = gridind(i);

        // This will propagate through fnarg[38], which will then be passed into function
        *xinf = &((gridsource.xinfo())(gridi));

        for ( j = 0 ; j < dim ; j++ )
        {
            xx[j] = (double) (gridsource.x(gridi))(j);
        }

        tempfres = (*fn)(dim,xx,fnarg);

        if ( ( ires == -1 ) || ( tempfres < (double) fres ) )
        {
            for ( j = 0 ; j < dim ; j++ )
            {
                x[j] = xx[j];
            }

            fres     = tempfres;
            ires     = i;
            gridires = gridi;

            xinfopt = *xinf;

            if ( tempfres <= hardmin )
            {
                break;
            }

            else if ( tempfres >= hardmax )
            {
                break;
            }
        }

        if ( xmtrtime > 1 )
        {
            curr_time = TIMECALL;

            if ( TIMEDIFFSEC(curr_time,start_time) > xmtrtime )
            {
                timeout = 1;
            }
        }

        if ( !timeout )
        {
            timeout = kbquitdet("Bayesian optimisation",uservars,varnames,vardescr);
        }
    }

    gridi = oldgridi;
    *xinf = xinfopt;

    errstream() << "MLGrid Optimisation Ended\n";

    MEMDELARRAY(xx);

    return 0;
}






//FIXME: working backwards, up (back) to here!











// ===========================================================================
// Evaluates [ mu sigma ] for multi-recommendation via multi-objective
// inner loop.  In multi-recommendation via multi-obj systems there is an 
// inner loop that multi-objectively maximises [ mu sigma ], so this function 
// evaluates and returns that.  Everything needs to be in the negative
// quadrant, so we need to negative sigma.
// ===========================================================================

void multiObjectiveCombine(gentype &res, Vector<gentype> &x, void *arg);
void multiObjectiveCombine(gentype &res, Vector<gentype> &x, void *arg)
{
    BayesOptions &bopts = *((BayesOptions *) ((void **) arg)[0]);

    res.force_vector();
    res.resize(2);

    SparseVector<gentype> xx(x);

    // mu approx is designed for maximisation, so we must negate this as it is used in a minimisation context
    bopts.model_muvar(res("&",1),res("&",0),xx,NULL); // ditto sigma by definition

    res("&",1).negate();

    return;
}












// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// Bayesian optimiser.
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================

int bayesOpt(int dim,
             Vector<double> &xres,
             gentype &fres,
             const Vector<double> &qmin,
             const Vector<double> &qmax,
             void (*fn)(int n, gentype &res, const double *x, void *arg, double &addvar),
             void *fnarg,
             BayesOptions &bopts,
             svmvolatile int &killSwitch,
             Vector<double> &sscore)
{
//    BayesOptions bopts(bbopts);

    int i,j,k;
    double addvar = 0;

    // =======================================================================
    // Work out levels of indirection in acquisition function.
    //
    // - isindirect: 1 if the acquisition function is a(p(x)), where x is the
    //   value given by DIRect, a is the usual acquisition function, and p is
    //   some pre-processing function given by direcpre.
    // - partindirect: 1 if the first recommendation in a batch is obtained 
    //   by optimising a(x) and subsequent recommendations in the batch by 
    //   optimising a(p(x))
    // - anyindirect: this is set on a per-iteration basis.  If it is 1 for a
    //   given iteration then that particular recommendation is generated by
    //   optimising a(p(x)), otherwise it is found by optimising a(x).
    // - direcdim: if isindirect or partindirect this is the dimension seen by
    //   the optimiser (DIRect) when optimising a(p(x)), otherwise it is the
    //   dimension seen when optimising a(x).  This is used on a per-iteration
    //   basis in conjunction with anyindirect to work out the dimension seen 
    //   for that particular iteration
    // - isstable: zero if no stability constraints, otherwise the number
    //   of stability constraints.
    //
    // - direcpre: the p function being used (if any).
    //
    //
    //
    //
    //
    // =======================================================================

    errstream() << "Entering Bayesian Optimisation Module.\n";

    int isindirect   = bopts.direcpre ? 1 : 0;
    int partindirect = bopts.direcsubseqpre ? 1 : 0;
    int anyindirect  = 0;
    int direcdim     = ( isindirect || partindirect ) ? bopts.direcdim : dim;
    int isstable     = bopts.stabpmax;
    int muapproxsize = 0;

    ML_Base *direcpre = (bopts.direcpre) ? (bopts.direcpre) : (bopts.direcsubseqpre);

    NiceAssert( dim > 0 );
    NiceAssert( direcdim > 0 );
    NiceAssert( qmin.size() == dim );
    NiceAssert( qmax.size() == dim );
    NiceAssert( qmax >= qmin );

    // =======================================================================
    // Put min/max bounds in gentype vector
    // =======================================================================
    //
    // effdim: dimension ignoring points at-end that are fixed (xmin(i) == xmax(i))

    errstream() << "Re-expressing bounds.\n";

    Vector<double> xmin(qmin);
    Vector<double> xmax(qmax);

    Vector<gentype> xminalt(dim);
    Vector<gentype> xmaxalt(dim);

    int effdim = 0;

    for ( i = 0 ; i < dim ; i++ )
    {
        xminalt("&",i) = xmin(i);
        xmaxalt("&",i) = xmax(i);

        if ( xmax(i) > xmin(i) )
        {
            effdim = i+1;
        }
    }

    NiceAssert( effdim > 0 );

    // =======================================================================
    // recBatchSize: this is the number of recommendations in each "batch".  
    // Note that this only gets set if we have n explicit strategies, it is 
    // set to one otherwise.  This is not the same thing as determinant method.
    //
    // numRecs: counter that keeps track of the number of recommendations so far.
    // newRecs: the number of recommendations in a given iteration of the algorithm.
    //
    // =======================================================================

    int recBatchSize = ( ( bopts.method == 11 ) && (bopts.betafn).isValVector() ) ? (bopts.betafn).size() : 1;
    int numRecs      = 0;
    int newRecs      = 0;

    NiceAssert( recBatchSize > 0 );
    NiceAssert( ( recBatchSize == 1 ) || ( effdim == dim ) );

    // =======================================================================
    // Local references for function approximation and sigma approximation.
    // sigmaml separate is useful when we have multi-rec and need 
    // "hallucinated" samples that need to update the sigma and not the mu.
    // =======================================================================

//    int sigmuseparate = bopts.sigmuseparate;

    // =======================================================================
    //
    // gridsource: non-null if we are restricted to a grid of x values that are
    // mapped by the given ML.
    //
    // =======================================================================

    errstream() << "Grid optimiser setup (if relevant).\n";

    retVector<int> tmpva;

    ML_Base *gridsource = bopts.gridsource;
    int isgridopt       = gridsource ? 1 : 0;
    int isgridcache     = bopts.gridcache;
    int gridi           = -1;
    double gridy        = 0;
    int Nbasemu         = bopts.modelmu_N();
    int Nbasesigma      = Nbasemu; // bopts.modelsigma_N(); - these are the same at this point (no hallucinations yet!)
    int Ngrid           = isgridopt ? (*gridsource).N() : 0;
    const vecInfo *xinf = NULL;

    Vector<int> gridind(cntintvec(Ngrid,tmpva));

    NiceAssert( !isgridopt || !(bopts.isXconvertNonTrivial()) );

    if ( isgridopt && isgridcache && Ngrid )
    {
        // Pre-add vectors to mu and sigma approximators

        for ( i = 0 ; i < Ngrid ; i++ )
        {
            bopts.modelmu_addTrainingVector(((*gridsource).y())(i),((*gridsource).y())(i),(*gridsource).x(i));

            if ( bopts.sigmuseparate )
            {
                bopts.modelsigma_addTrainingVector(((*gridsource).y())(i),(*gridsource).x(i));
            }

            muapproxsize++;
            bopts.model_setd(Nbasemu+i,Nbasesigma+i,0);
        }
    }

    // =======================================================================
    // Actual optimiser starts here
    // =======================================================================

    int dummy = 0;

    Vector<double> xa(dim);
    Vector<SparseVector<double> > xb(recBatchSize);
    Vector<gentype> xxa(dim);
    Vector<SparseVector<gentype> > xxb(recBatchSize);
    gentype fnapproxout;
    gentype nothingmuch('N');
    SparseVector<gentype> xinb;
    gentype xytemp;

    // =======================================================================
    // 
    // - softmax: we convert the minimisation problem to maximisation by
    //   negating, so this is the "soft" maximum.
    // - betaval: the beta for GP-UCB and related acquisition functions.  It
    //   is 1 for the initial random seeds to ensure that variance approxs
    //   make sense.
    // - mupred: stores mu predictions
    // - sigmapred: sigma predictions
    // 
    // =======================================================================

    double softmax = -(bopts.softmin);

    double betaval = 1.0; // for initial batch
    Vector<gentype> mupred(recBatchSize);
    Vector<gentype> sigmapred(recBatchSize);

    // =======================================================================
    // Various timers.
    // =======================================================================

    timediffunits bayesruntime;
    timediffunits mugptraintime;
    timediffunits sigmagptraintime;
    
    ZEROTIMEDIFF(bayesruntime);
    ZEROTIMEDIFF(mugptraintime);
    ZEROTIMEDIFF(sigmagptraintime);

    // =======================================================================
    // Ensure the model has at least startpoints seed points to begin.
    // 
    // NB: we DO NOT want to add points to the model if it already has enough
    //     points in it!  This matters for nested methods like cBO where the
    //     "inner" model should provide one recommendation per "outer" 
    //     iteration, and adding additional seeds will simply slow it down.
    //
    // Default: if startpoints == -1 then add dim+1 points.  Apparently this
    //          is standard practice.
    // =======================================================================

    int startpoints = ( bopts.startpoints == -1 ) ? effdim+1 : bopts.startpoints;; //( bopts.startpoints == -1 ) ? dim+1 : bopts.startpoints; - note use of effdim here!
    int &startseed  = bopts.startseed; // reference because we need to *persistently* update it!
    int &algseed    = bopts.algseed; // reference because we need to *persistently* update it!
    double modD     = ( bopts.modD == -1 ) ? ( isgridopt ? Ngrid : 10 ) : bopts.modD; // 10 is arbitrary here!
    int Nmodel      = Nbasemu; //bopts.modelmu_N();
    int Nsigma      = Nbasesigma; //bopts.modelsigma_N();
    int ires        = -1;

    // Do no start points if scheduled Bernstein, not first iteration)

    if ( bopts.spOverride )
    {
        startpoints = 0;
    }

    fres = 0;

    // =======================================================================
    // Initial batch generation
    // =======================================================================

    errstream() << "Testing initial batch.\n";

    int Nstart = Nmodel;

    int oldstartseed = startseed;

//errstream() << "phantomx 0\n";
    if ( startpoints )
    {
        k = 0;
        {
            for ( i = Nstart ; ( i < Nstart+startpoints ) && !killSwitch && ( !isgridopt || gridind.size() ) ; i++ )
            {
                // ===========================================================
                // Generate random point.  Each random point is generating by
                // sampling from U(0,1) and scaling to lie in U(xmin,xmax).
                // ===========================================================

                if ( isgridopt )
                {
                    j     = svm_rand()%(gridind.size());
                    gridi = gridind(j);
                    gridy = ((*gridsource).y())(gridi);
                    xinf  = &(((*gridsource).xinfo())(gridi));

                    gridind.remove(j);
//errstream() << "phantomxgrid 0: i = " << gridi << "\n";
//errstream() << "phantomxgrid 1: yi = " << gridy << "\n";
                }

//errstream() << "phantomx 1\n";
                if ( startseed >= 0 )
                {
                    svm_srand(startseed); // We seed the RNG to create predictable random numbers
                                          // We dpo it HERE so that random factors OUTSIDE of this
                                          // code that may get called don't get in the way and mess
                                          // up our predictable random sequence

                    startseed += 12; // this is a predictable increment so that multiple repeat
                                     // experiments each have a different but predictable startpoint

                    startseed = ( startseed < 0 ) ? 42 : startseed; // Just in case (very unlikely that this will roll over unless we add a *lot* of points)
                }

                else if ( startseed == -2 )
                {
                    svm_srand(-2); // seed with time
                }

                for ( j = 0 ; j < dim ; j++ )
                {
                    if ( !isgridopt && ( j < effdim ) )
                    {
                        // Continuous approximation: values are chosen randomly from
                        // uniform distribution on (continuous) search space.

                        randfill(xxb("&",k)("&",j).force_double());

                        xxb("&",k)("&",j).dir_double() *= (xmax(j)-xmin(j));
                        xxb("&",k)("&",j).dir_double() += xmin(j); 
                    }

                    else if ( !isgridopt && ( j >= effdim ) )
                    {
                        xxb("&",k)("&",j).dir_double() = xmin(j); 
                    }

                    else
                    {
                        // Discrete approximation: values are chosen randomly from
                        // a finite grid.

                        xxb("&",k)("&",j) = ((*gridsource).x(gridi))(j);
                    }
                }

                // ===========================================================
                // Generate preliminary predictions
                // ===========================================================

//errstream() << "phantomx 2\n";
                for ( j = 0 ; j < dim ; j++ )
                {
                    xb("&",k)("&",j) = xxb(k)(j);
                }

//errstream() << "phantomx 2a\n";
                if ( isgridopt && isgridcache )
                {
//errstream() << "phantomx 2b\n";
                    bopts.model_muvarTrainingVector(sigmapred("&",k),mupred("&",k),Nbasesigma+gridi,Nbasemu+gridi);
                }

                else
                {
//errstream() << "phantomx 2c: " << k << "\n";
//errstream() << "phantomx 2d: " << xxb(k) << "\n";
//errstream() << "phantomx 2e: " << sigmapred(k) << "\n";
//errstream() << "phantomx 2f: " << mupred(k) << "\n";
                    bopts.model_muvar(sigmapred("&",k),mupred("&",k),xxb(k),xinf);
                }

                // ===========================================================
                // Calculate supplementary data
                // ===========================================================

                double rmupred    = mupred(k).isCastableToReal()    ? ( (double) mupred(k)    ) : 0.0;
                double rsigmapred = sigmapred(k).isCastableToReal() ? ( (double) sigmapred(k) ) : 0.0;
                double standev    = sqrt(betaval)*rsigmapred;

//errstream() << "phantomx 3\n";
                xb("&",k)("&",dim  )  = 0; //newRecs-1;
                xb("&",k)("&",dim+1)  = betaval;
                xb("&",k)("&",dim+2)  = rmupred;
                xb("&",k)("&",dim+3)  = rsigmapred;
                xb("&",k)("&",dim+4)  = rmupred+standev;
                xb("&",k)("&",dim+5)  = rmupred-standev;
                xb("&",k)("&",dim+6)  = 2*standev;
                xb("&",k)("&",dim+7)  = softmax;
                xb("&",k)("&",dim+8)  = 0; // You need this to ensure vector is not sparse!
                xb("&",k)("&",dim+9)  = bayesruntime;
                xb("&",k)("&",dim+10) = mugptraintime;
                xb("&",k)("&",dim+11) = sigmagptraintime;
                xb("&",k)("&",dim+12) = gridi;
                xb("&",k)("&",dim+13) = gridy;

                // ===========================================================
                // Run experiment:
                // ===========================================================

//errstream() << "phantomx 4\n";
                fnapproxout.force_int() = -1;
                (*fn)(dim,fnapproxout,&xb(k)(zeroint()),fnarg,addvar);
                fnapproxout.negate();

                // ===========================================================
                // Add new point to machine learning block, correct variances.
                // ===========================================================

                int Ninmu = ( isgridopt && isgridcache ) ? Nbasemu+gridi : Nmodel;

//errstream() << "phantomx 5\n";
                if ( isgridopt && isgridcache )
                {
                    bopts.model_setd(Nbasemu+gridi,Nbasesigma+gridi,2);
                }

                else
                {
//errstream() << "phantomx 5b\n";
//errstream() << "phantomx 5b: " << fnapproxout << "\n";
//errstream() << "phantomx 5b: " << mupred(k) << "\n";
//errstream() << "phantomx 5b: " << xxb(k) << "\n";
                    bopts.modelmu_addTrainingVector(fnapproxout,mupred(k),xxb(k));

                    if ( bopts.sigmuseparate )
                    {
                        bopts.modelsigma_addTrainingVector(fnapproxout,xxb(k));
                    }

                    muapproxsize++;
                }

//errstream() << "phantomx 6\n";
                if ( addvar != 0 )
                {
                    bopts.model_setsigmaweight(Nmodel,Nsigma,((bopts.model_sigma())+addvar)/(bopts.model_sigma()));
                }

//errstream() << "phantomx 7\n";
                if ( bopts.isimphere() )
                {
                    if ( fnapproxout.isValVector() )
                    {
                        const Vector<gentype> &ghgh = (const Vector<gentype> &) fnapproxout;

                        for ( j = 0 ; j < fnapproxout.size() ; j++ )
                        {
                            xinb("&",j) = ghgh(j);
                        }
                    }

                    else
                    {
                        xinb("&",zeroint()) = fnapproxout;
                    }

                    xinb.negate();

                    bopts.modelimp_addTrainingVector(nothingmuch,xinb);
                }

                // ===========================================================
                // Feedback
                // ===========================================================

//errstream() << "phantomx 8\n";
                if ( bopts.isimphere() )
                {
                    errstream() << "(" << Nmodel << "," << bopts.modelimp_N() << ").";
                }

                else
                {
                    errstream() << "(" << Nmodel << ").";
                }

                // ===========================================================
                // Fill in fres and ires based on result
                // ===========================================================

                if ( ( ires == -1 ) || ( (bopts.modelmu_y())(Ninmu) > fres ) )
                {
                    fres = (bopts.modelmu_y())(Ninmu);
                    ires = Ninmu;
                }

                // ===========================================================
                // Counters
                // ===========================================================

//errstream() << "phantomx 9\n";
                Nmodel++;
                Nsigma++;
            }
        }
    }

    if ( startseed >= 0 )
    {
        startseed = oldstartseed + 19;
    }

//errstream() << "phantomx 10\n";
    // ===========================================================
    // ysort is used to index bopts.modelmu_y() from smallest to largest,
    // which is required for calculating stability scores.
    // ===========================================================

    Vector<int> ysort;

    int stabpmax = bopts.stabpmax;
    int stabpmin = bopts.stabpmin;
    int stabrot  = 0; //bopts.stabrot;

    double stabpnrm   = 2; //bopts.stabpnrm;
    double stabA      = bopts.stabA;
    double stabB      = bopts.stabB;
    double stabF      = bopts.stabF;
    double stabbal    = bopts.stabbal;
    double stabZeroPt = bopts.stabZeroPt;

    int stabp = 0;

    double stabmumin = 0;
    double stabmumax = 0;
    double stabmu    = 0;
    double stablogD  = 0;
    double stabM     = 0;
    double stabLs    = 0;
    double stabDelta = 0;

    if ( isstable )
    {
        errstream() << "Setting stability parameters.\n";

        // FIXME: currently we are simply *assuming* an RBF kernel.  In general
        //        though we need to test what sort of kernel is actually being
        //        used and then set these constants accordingly, testing where
        //        required.

        stabLs    = 1.0/(((bopts.model_getKernel().cRealConstants())(zeroint()))*((bopts.model_getKernel().cRealConstants())(zeroint())));
        stabDelta = 0;

        double Lsca = 2*stabLs*stabB*stabB;

        NiceAssert( Lsca < 1 );

        for ( j = 0 ; j < dim ; j++ )
        {
            stabM += ((xmax(j)-xmin(j))*(xmax(j)-xmin(j)));
        }

        stabM = sqrt(stabM);
        //stabD = 0.816*NUMBASE_SQRTSQRTPI*exp(0.5*stabM*stabM*stabLs); - this overflows
        stablogD = log(0.816*NUMBASE_SQRTSQRTPI)+(0.5*stabM*stabM*stabLs);

        double lambres = 0.0;

        //numbase_lambertW(lambres, (2.0/NUMBASE_E)*(1.0/Lsca)*(log(NUMBASE_1ONSQRTSQRT2PI*((stabD*stabF)/(stabA-(stabDelta*stabF)))*(1.0/(1.0-sqrt(Lsca))))) ); - this overflows
        numbase_lambertW(lambres, (2.0/NUMBASE_E)*(1.0/Lsca)*(stablogD+log(NUMBASE_1ONSQRTSQRT2PI*(stabF/(stabA-(stabDelta*stabF)))*(1.0/(1.0-sqrt(Lsca))))) );

        stabp = (int) ceil((Lsca*exp(1+lambres))-1);

        //double Rbnd = (((stabD/sqrt(xnfact(stabp+1)))*(pow(Lsca,(stabp+1.0)/2.0)/(1.0-Lsca)))+stabDelta)*stabF; // Need to use un-slipped pmin here!
        // Need to do this manually to prevent an overflow

        /* aaand it still overflows
        double Rbnd = stabD;

        for ( j = 1 ; j <= stabp+1 ; j++ )
        {
            Rbnd /= sqrt(j);
        }
        */

        double Rbnd = stablogD;

        for ( j = 1 ; j <= stabp+1 ; j++ )
        {
            Rbnd -= (sqrt(j)/2.0);
        }

        Rbnd = exp(Rbnd);

        Rbnd *= pow(sqrt(Lsca),stabp+1)/(1.0-sqrt(Lsca));
        Rbnd += stabDelta;
        Rbnd *= stabF;

        stabmumin = stabA - Rbnd;
        stabmumax = stabA + Rbnd;

        stabmu = stabmumin + (stabbal*(stabmumax-stabmumin));

        errstream() << "Stable optimisation A     = " << stabA     << "\n";
        errstream() << "Stable optimisation B     = " << stabB     << "\n";
        errstream() << "Stable optimisation F     = " << stabF     << "\n";
        errstream() << "Stable optimisation M     = " << stabM     << "\n";
        errstream() << "Stable optimisation log(D)= " << stablogD  << "\n";
        errstream() << "Stable optimisation Ls    = " << stabLs    << "\n";
        errstream() << "Stable optimisation Delta = " << stabDelta << "\n";
        errstream() << "Stable optimisation Up    = " << Rbnd      << "\n";
        errstream() << "Stable optimisation pmin  = " << stabp     << "\n";
        errstream() << "Stable optimisation mu-   = " << stabmumin << "\n";
        errstream() << "Stable optimisation mu+   = " << stabmumax << "\n";
        errstream() << "Stable optimisation mu    = " << stabmu    << "\n";
        errstream() << "Stable optimisation p     = " << stabp     << "\n";

        stabp = ( stabp >= stabpmin ) ? stabp : stabpmin;
        stabp = ( stabp <= stabpmax ) ? stabp : stabpmax;

        errstream() << "Stable optimisation p (adjusted) = " << stabp << "\n";

        retVector<int> tmpva;

        Vector<int> rort(cntintvec(bopts.modelmu_N(),tmpva));

        int jj = 0;
        int firstone;
        int NC = bopts.modelmu_N()-bopts.modelmu_NNCz();

        ysort.prealloc(NC+(( bopts.totiters == -1 ) ? 10*dim : bopts.totiters)+10);
        sscore.prealloc(NC+(( bopts.totiters == -1 ) ? 10*dim : bopts.totiters)+10);

        while ( NC )
        {
            firstone = 1;

            for ( j = 0 ; j < rort.size() ; j++ )
            {
                if ( (bopts.modelmu_d())(j) && ( firstone || ( (bopts.modelmu_y())(rort(j)) < (bopts.modelmu_y())(rort(jj)) ) ) )
                {
                    jj = j;
                    firstone = 0;
                }
            }

            ysort.add(ysort.size());
            ysort("&",ysort.size()-1) = rort(jj);
            rort.remove(jj);

            NC--;
        }
    }

    xinf = NULL;

    // ===================================================================
    // Train the machine learning block(s)
    // ===================================================================

    errstream() << "Model tuning.\n";

    bopts.model_train(dummy,killSwitch);
    bopts.modelimp_train(dummy,killSwitch);

    // =======================================================================
    // =======================================================================
    // =======================================================================
    // =======================================================================
    // =======================================================================
    // =======================================================================
    // =======================================================================
    // =======================================================================
    // =======================================================================





    // =======================================================================
    // Proceed with Bayesian optimisation
    // 
    // - itcnt: iteration count not including random seed.
    // 
    // - altitcnt is used as t when calculating beta.  In plain vanilla bayesian
    //   this starts as N+1 (to make sure everything is well defined and noting 
    //   the possible presence of previous iterations on the batch that affect the
    //   confidence interval, which is what beta_t measures) and iterates by 1 for 
    //   each recommendation.  In batch things are a bit different:
    //
    //   - if itcntmethod = 0 then it starts as (N/B)+1 (where B is the batch size)
    //     and iterates by 1 for every batch.  This is in line with GP-UCB-PE, where
    //     t is the number of batches, and is default behaviour.
    //
    //   - if itcntmethod = 1 then it starts as N+1 and iterates by B for every
    //     batch.  This is in line with GP-BUCB, where t is the number of 
    //     recommendations.
    // 
    // - betavalmin: minimum beta value in the current batch.  This is necessary
    //   when calculating the variance adjustment factors eg for cBO.
    // 
    // - ismultitargrec: 1 set 1 if using multi-objective mu/sigma optimisation
    //   to do multi-recommendation for this iteration.
    // 
    // - currintrinbatchsize: see later.
    // 
    // =======================================================================

    errstream() << "Training setup.\n";

    unsigned int itcnt    = 0;
    int itcntmethod       = bopts.itcntmethod;
    unsigned int altitcnt = ( bopts.modelmu_N() / ( itcntmethod ? 1 : recBatchSize ) ) + 1;
    double betavalmin     = 0.0;

    int ismultitargrec      = 0;
    int currintrinbatchsize = 1;

    gentype dummyres(0.0);
    gentype sigmax('R');
    gentype tempval;

    SparseVector<double> diffx;
    SparseVector<gentype> xappend;
    int justreturnbeta = 0;

    int thisbatchsize   = bopts.intrinbatch;
    int thisbatchmethod = bopts.intrinbatchmethod;

    Matrix<gentype> covarmatrix(thisbatchsize,thisbatchsize);
    Vector<gentype> meanvector(thisbatchsize);
    Vector<SparseVector<gentype> > multivecin(thisbatchsize);
    Vector<SparseVector<double> > yb(recBatchSize);
    Vector<SparseVector<gentype> > yyb(recBatchSize);
    gentype locpen;

    double ztol  = bopts.ztol;
    double delta = bopts.delta;
    double nu    = bopts.nu;

    // Variables used in stable bayesian

    int firstevalinseq = 1;

    // =======================================================================
    // The following variable is used to pass variables through to fnfnapprox.
    // =======================================================================

    fninnerinnerArg fnarginner(bopts,
                               yyb("&",zeroint()),
                               fnapproxout,
                               sigmax,
                               fres,
                               ( itcntmethod != 3 ) ? altitcnt : itcnt,
                               dim,
                               effdim,
                               ztol,
                               delta,
                               nu,
                               tempval,
                               yb("&",zeroint()),
                               diffx,
                               (bopts.a),
                               (bopts.b),
                               (bopts.r),
                               (bopts.p),
                               modD,
                               (bopts.betafn),
                               (bopts.method),
                               (bopts.stepweight),
                               justreturnbeta,
                               covarmatrix,
                               meanvector,
                               thisbatchsize,
                               multivecin,
                               direcpre,
                               xytemp,
                               thisbatchmethod,
                               xappend,
                               anyindirect,
                               softmax,
                               itcntmethod,
                               k,
                               (bopts.penalty),
                               locpen,
                               &xinf,
                               Nbasemu,
                               Nbasesigma,
                               gridi,
                               isgridopt,
                               isgridcache,
                               isstable,
                               ysort,
                               stabp,
                               stabpnrm,
                               stabrot,
                               stabmu,
                               stabB,
                               sscore,
                               firstevalinseq,
                               stabZeroPt,
                               (bopts.unscentUse),
                               (bopts.unscentK),
                               (bopts.unscentSqrtSigma),
                               (bopts.stabUseSig),
                               (bopts.stabThresh));

    void *fnarginnerdr = (void *) &fnarginner;

    // =======================================================================
    // Variables used to calculate termination conditions, timers etc
    // 
    // maxitcnt: if totiters == -1 then this is set to 10dim, which is apparently
    //           the "standard" number of iterations.
    // 
    // =======================================================================

    int timeout     = 0;
    int isopt       = 0;
    double xmtrtime = bopts.maxtraintime;
    double maxitcnt = ( bopts.totiters == -2 ) ? 0 : ( ( bopts.totiters == -1 ) ? 10*effdim : bopts.totiters ); // note use of effdim, not dim ( bopts.totiters == -2 ) ? 0 : ( ( bopts.totiters == -1 ) ? 10*dim : bopts.totiters );
    int dofreqstop  = ( bopts.totiters == -2 ) ? 1 : 0;

    double *uservars[]     = { &maxitcnt, &xmtrtime, &ztol, &delta, &nu, NULL };
    const char *varnames[] = { "itercount", "traintime", "ztol", "delta", "nu", NULL };
    const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", "Zero tolerance", "delta", "nu", NULL };

    time_used start_time = TIMECALL;
    time_used curr_time  = start_time;









    // =======================================================================
    // Main optimisation loop
    // =======================================================================

    errstream() << "Entering main optimisation loop.\n";

    if ( algseed >= 0 )
    {
        // See comments in startseed

        svm_srand(algseed);

        algseed += 12;
        algseed = ( algseed == -1 ) ? 42 : algseed;
    }

    else if ( algseed == -2 )
    {
        svm_srand(-2); // seed with time
    }


    while ( !killSwitch && !isopt && ( ( itcnt < (unsigned int) maxitcnt ) || !maxitcnt ) && !timeout && ( !isgridopt || gridind.size() ) )
    {
        // ===================================================================
        // Run intermediate string, if there is one.
        // ===================================================================

        errstream() << "Intermediate string.\n";

        gentype dummyyres;

        dummyyres.force_int() = 0;
        (*fn)(-1,dummyyres,NULL,fnarg,addvar);

        errstream() << "-------------------------------------------\n";

        // ===================================================================
        // Optimisation continues.
        // ===================================================================

        xappend.zero();

        numRecs = 0;
        betavalmin = 0;

        for ( k = 0 ; k < recBatchSize ; k++ )
        {
            anyindirect = isindirect || ( partindirect && k );

            // ===============================================================
            // Multi-recommendation: these change depending on which
            // recommendation we are processing
            // ===============================================================

            fnarginner.__x     = &(yyb("&",k));
            fnarginner.__prevx = &yb("&",k);

            ismultitargrec      = 0;
            currintrinbatchsize = bopts.intrinbatch;

            if ( ( bopts.method == 11 ) && (bopts.betafn).isValVector() )
            {
                if ( ((bopts.betafn)(k)).isValVector() )
                {
                    // =======================================================
                    // Explicit multi-recommendation via multiple strategies
                    // =======================================================

                    NiceAssert( ((bopts.betafn)(k)).size() >= 1  );
                    NiceAssert( ((bopts.betafn)(k)).size() <= 10 );

                    gentype locbetafn((bopts.betafn)(k)(2));

                    // Note use of cast_ here (without finalisation) just in case any of these are functions.

                    fnarginner.__locmethod       = ((bopts.betafn)(k)(zeroint())).cast_int(0);
                    fnarginner.__p               = ( ( ((bopts.betafn)(k)).size() >= 2 ) && !(((bopts.betafn)(k)(1 )).isValVector()) ) ?  (((bopts.betafn)(k)(1 )).cast_double(0)) :  (bopts.p);
                    fnarginner.__betafn          = ( ( ((bopts.betafn)(k)).size() >= 3 ) && !(((bopts.betafn)(k)(2 )).isValVector()) ) ? &locbetafn                                : &(bopts.betafn);
                    fnarginner.__modD            = ( ( ((bopts.betafn)(k)).size() >= 4 ) && !(((bopts.betafn)(k)(3 )).isValVector()) ) ?  (((bopts.betafn)(k)(3 )).cast_double(0)) :  bopts.modD;
                    fnarginner.__delta           = ( ( ((bopts.betafn)(k)).size() >= 5 ) && !(((bopts.betafn)(k)(4 )).isValVector()) ) ?  (((bopts.betafn)(k)(4 )).cast_double(0)) :  delta;
                    fnarginner.__nu              = ( ( ((bopts.betafn)(k)).size() >= 6 ) && !(((bopts.betafn)(k)(5 )).isValVector()) ) ?  (((bopts.betafn)(k)(5 )).cast_double(0)) :  nu;
                    fnarginner.__a               = ( ( ((bopts.betafn)(k)).size() >= 7 ) && !(((bopts.betafn)(k)(6 )).isValVector()) ) ?  (((bopts.betafn)(k)(6 )).cast_double(0)) :  (bopts.a);
                    fnarginner.__b               = ( ( ((bopts.betafn)(k)).size() >= 8 ) && !(((bopts.betafn)(k)(7 )).isValVector()) ) ?  (((bopts.betafn)(k)(7 )).cast_double(0)) :  (bopts.b);
                    fnarginner.__r               = ( ( ((bopts.betafn)(k)).size() >= 9 ) && !(((bopts.betafn)(k)(8 )).isValVector()) ) ?  (((bopts.betafn)(k)(8 )).cast_double(0)) :  (bopts.r);
                    fnarginner.__thisbatchsize   = ( ( ((bopts.betafn)(k)).size() >= 10) && !(((bopts.betafn)(k)(9 )).isValVector()) ) ?  (((bopts.betafn)(k)(9 )).cast_int   (0)) :  (bopts.intrinbatch);
                    fnarginner.__thisbatchmethod = ( ( ((bopts.betafn)(k)).size() >= 11) && !(((bopts.betafn)(k)(10)).isValVector()) ) ?  (((bopts.betafn)(k)(10)).cast_int   (0)) :  (bopts.intrinbatchmethod);

                    if ( ( (bopts.betafn(k)).size() >= 10 ) && !(((bopts.betafn)(k)(9)).isValVector()) )
                    {
                        currintrinbatchsize = (int) ((bopts.betafn)(k)(9));
                    }
                }

                else
                {
                    // =======================================================
                    // Multi-recommendation via multi-objective optimisation 
                    // (beta function null)
                    // ...or...
                    // Single explicit strategy (beta function non-null)
                    // =======================================================

                    fnarginner.__locmethod       = (bopts.method);
                    fnarginner.__p               = (bopts.p);
                    fnarginner.__betafn          = &(bopts.betafn);
                    fnarginner.__modD            = bopts.modD;
                    fnarginner.__delta           = delta;
                    fnarginner.__nu              = nu;
                    fnarginner.__a               = (bopts.a);
                    fnarginner.__b               = (bopts.b);
                    fnarginner.__r               = (bopts.r);
                    fnarginner.__thisbatchsize   = (bopts.intrinbatch);
                    fnarginner.__thisbatchmethod = (bopts.intrinbatchmethod);

                    // If this gets set then we are using multi-objective optimisation on (mu,sigma) to construct relevant number of recommendations
                    ismultitargrec = ((bopts.betafn)(k)).isValNull() ? 1 : 0;
                }
            }

            // ===============================================================
            // Find next experiment parameters using DIRect global optimiser
            // ===============================================================

            errstream() << "DIRect@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@(" << itcnt << ")...";

            time_used bayesbegintime = TIMECALL;

            if ( !ismultitargrec )
            {
                int jji,jjj;

                if ( dim*currintrinbatchsize > xa.size() )
                {
                    int oldibs = (xa.size())/dim;

                    xa.resize(dim*currintrinbatchsize);

                    xmax.resize(dim*currintrinbatchsize);
                    xmin.resize(dim*currintrinbatchsize);

                    for ( jji = oldibs ; jji < currintrinbatchsize ; jji++ )
                    {
                        for ( jjj = 0 ; jjj < dim ; jjj++ )
                        {
                            xmin("&",(jji*dim)+jjj) = xmin(jjj);
                            xmax("&",(jji*dim)+jjj) = xmax(jjj);
                        }
                    }
                }

                if ( direcdim > xa.size() )
                {
                    xa.resize(direcdim);
                }

                int n = anyindirect ? direcdim : dim*currintrinbatchsize;

                // Bounds are different if we are using direcpre

                const Vector<double> &direcmin = anyindirect ? bopts.direcmin : xmin;
                const Vector<double> &direcmax = anyindirect ? bopts.direcmax : xmax;

                double temphardmin = bopts.hardmin;
                double temphardmax = bopts.hardmax;

                bopts.hardmin = valninf();
                bopts.hardmax = valpinf();

                int dres = 0;

                firstevalinseq = 1;

                if ( !isgridopt )
                {
                    // Continuous search space, use DIRect to find minimum 

                    // but first over-ride goptssingleobj with relevant components from *this (assuming non-virtual assignment operators)
                    static_cast<GlobalOptions &>(bopts.goptssingleobj) = static_cast<GlobalOptions &>(bopts);

                    retVector<double> tmpva;
                    retVector<double> tmpvb;
                    retVector<double> tmpvc;

                    dres = directOpt(n,xa("&",zeroint(),1,n-1,tmpva),dummyres,direcmin(zeroint(),1,n-1,tmpvb),direcmax(zeroint(),1,n-1,tmpvc),
                                     fnfnapprox,fnarginnerdr,bopts.goptssingleobj,killSwitch);
errstream() << "phantomx Unfiltered result x: " << xa << "\n";
errstream() << "phantomx Unfiltered result y: " << dummyres << "\n";
                }

                else
                {
                    // Discrete (grid) search space, use grid search.
                    //
                    // NB: - intrinsic batch can't work out x, so can't use it here.
                    //     - itorem: index in gridi of minimum, so we can remove it from grid.

                    NiceAssert( currintrinbatchsize == 1 );

                    int itorem   = -1;
                    int gridires = -1;

                    retVector<double> tmpva;

                    dres = dogridOpt(n,xa("&",zeroint(),1,n-1,tmpva),dummyres,itorem,gridires,
                                     fnfnapprox,fnarginnerdr,bopts,*gridsource,gridind,killSwitch,xmtrtime);

                    gridi = gridires;

                    gridind.remove(itorem);
                    gridy = ((*gridsource).y())(gridi);
                }

                bopts.hardmin = temphardmin;
                bopts.hardmax = temphardmax;

                errstream() << "Return code = " << dres << "\n\n";

                if ( anyindirect )
                {
                    // Direct result needs to be processed to get recommendation batch

                    SparseVector<gentype> tempx;

                    for ( i = 0 ; i < n ; i++ )
                    {
                        tempx("&",i) = xa(i); // gentype sparsevector
                    }

                    if ( xappend.size() && k )
                    {
                        for ( i = n ; i < n+xappend.size() ; i++ )
                        {
                            tempx("&",i) = xappend(i-n);
                        }
                    }

                    (*direcpre).gg(xytemp,tempx);

                    const Vector<gentype> &ghgh = (const Vector<gentype> &) xytemp;

                    for ( i = 0 ; i < dim*currintrinbatchsize ; i++ )
                    {
                        xa("&",i) = (double) ghgh(i);
                    }
                }

                if ( partindirect && !k )
                {
                    assert( !anyindirect );

                    // update xappend here

                    for ( i = 0 ; i < n ; i++ )
                    {
                        xappend("&",i) = xa(i);
                    }
                }

                errstream() << dres << "...";

                newRecs = currintrinbatchsize;

                if ( numRecs+newRecs >= xb.size() )
                {
                    xb.resize(numRecs+newRecs);
                    xxb.resize(numRecs+newRecs);
                }

                retVector<double> tmpva;

                for ( jji = 0 ; jji < newRecs ; jji++ )
                {
                    xb("&",numRecs+jji).zero();
                    xb("&",numRecs+jji) = xa((jji*dim),1,(jji*dim)+dim-1,tmpva);
                }
            }

            else
            {
                // ===========================================================
                // grid-search is incompatible with this method!
                // ===========================================================

                NiceAssert( currintrinbatchsize == 1 );
                NiceAssert( !anyindirect );
                NiceAssert( !isgridopt );

                // ===========================================================
                // Rather than maximising a simple acquisition function we are
                // multi-objectively maximising (mu,sigma) to give many 
                // solutions (recommendations) in a single batch.
                //
                // To do this we basically need to recurse to an inner-loop
                // Bayesian optimiser, so we need to set up all the relevant
                // variables.
                // ===========================================================

                IMP_Expect locimpmeasu;
                BayesOptions locbopts(bopts);

                locimpmeasu.setehimethod(bopts.ehimethodmultiobj);

                locbopts.ismoo             = 1;
                locbopts.impmeasu          = &(static_cast<IMP_Generic &>(locimpmeasu));
                locbopts.method            = 1;
                locbopts.intrinbatch       = 1;
                locbopts.intrinbatchmethod = 0;
                locbopts.startpoints       = bopts.startpointsmultiobj;
                locbopts.totiters          = bopts.totitersmultiobj;

                locbopts.goptssingleobj = bopts.goptsmultiobj;

                Vector<gentype> dummyxres;
                gentype dummyfres;
                int dummyires;
                Vector<double> dummyhypervol;

                Vector<Vector<gentype> > locallxres;
                Vector<gentype> locallfres;
                Vector<gentype> locallfresmod;
                Vector<gentype> locsupres;
                Vector<double> locsscore;
                Vector<int> locparind;

                // ===========================================================
                // Call multi-objective bayesian optimiser
                // ===========================================================

                void *locfnarg[1];

                locfnarg[0] = (void *) &bopts;

                bayesOpt(dim,
                         dummyxres,
                         dummyfres,
                         dummyires,
                         locallxres,
                         locallfres,
                         locallfresmod,
                         locsupres,
                         locsscore,
                         xminalt,
                         xmaxalt,
                         &multiObjectiveCombine,
                         (void *) locfnarg,
                         locbopts,
                         killSwitch);

                // ===========================================================
                // Find pareto (recommendation) set and set numRecs
                // Pareto set will be indexed by locparind
                // ===========================================================

                newRecs = bopts.analyse(locallxres,locallfresmod,dummyhypervol,locparind,1);

                NiceAssert( newRecs );

                // ===========================================================
                // Grow xb and xxb.  We grow enough to fit what has been + 
                // this + one recommendation for each future batch
                // ===========================================================

                if ( numRecs+newRecs >= xb.size() )
                {
                    xb.resize(numRecs+newRecs);
                    xxb.resize(numRecs+newRecs);
                }

                // ===========================================================
                // Transfer recommendations to xb
                // NB: xb is of type Vector<SparseVector<double> >
                //     locallxres(locparind) is of type Vector<SparseVector<gentype> >
                // ===========================================================

                int jji,jki;
                
                for ( jji = 0 ; jji < newRecs ; jji++ )
                {
                    xb("&",numRecs+jji).resize(locallxres(locparind(jji)).size());

                    for ( jki = 0 ; jki < locallxres(locparind(jji)).size() ; jki++ )
                    {
                        xb("&",numRecs+jji)("&",jki) = (double) locallxres(locparind(jji))(jki);
                    }
                }
            }

            time_used bayesendtime = TIMECALL;
            bayesruntime = TIMEDIFFSEC(bayesendtime,bayesbegintime);

            errstream() << " " << bayesruntime << " sec ";

            // ===============================================================
            // Update models, record results etc
            // ===============================================================

            justreturnbeta = 1; // This makes fnfnapprox return beta
            betaval = fnfnapprox(dim,&(xa("&",0)),fnarginnerdr);
            betavalmin = ( !k || ( betaval < betavalmin ) ) ? betaval : betavalmin;
            justreturnbeta = 0; // This puts things back to standard operation

            // If the iteration count is per batch size then do this

            altitcnt += itcntmethod ? 1 : 0;

            if ( newRecs )
            {
                int ij;

                // Find minimum beta value

                for ( ij = 0 ; ij < newRecs ; ij++ )
                {
//da hell? Why is there nothing here?  What is this code meant to do?
                }
            }

            while ( newRecs )
            {
                // ===========================================================
                // We do this now for convenience.
                // ===========================================================

                for ( j = 0 ; j < dim ; j++ )
                {
                    xxb("&",numRecs)("&",j) = xb(numRecs)(j);
                }

                // ===========================================================
                // Update sigma model if separate ("hallucinated" samples)
                // ===========================================================

                if ( bopts.sigmuseparate )
                {
                    if ( isgridopt && isgridcache )
                    {
                        bopts.modelsigma_setd(Nbasesigma+gridi,2);
                    }

                    else
                    {
                        bopts.modelsigma_addTrainingVector(fnapproxout,xxb(numRecs));
                    }

                    bopts.modelsigma_train(dummy,killSwitch);
                }

                // ===========================================================
                // Record beta, Nrec etc
                // ===========================================================

                if ( isgridopt && isgridcache )
                {
                    bopts.model_muvarTrainingVector(sigmapred("&",numRecs),mupred("&",numRecs),Nbasesigma+gridi,Nbasemu+gridi);
                }

                else
                {
                    bopts.model_muvar(sigmapred("&",numRecs),mupred("&",numRecs),xxb(numRecs),xinf);
                }

                double rmupred    = mupred(numRecs).isCastableToReal()    ? ( (double) mupred(numRecs)    ) : 0.0;
                double rsigmapred = sigmapred(numRecs).isCastableToReal() ? ( (double) sigmapred(numRecs) ) : 0.0;
                double standev    = sqrt(betavalmin)*rsigmapred;

                xb("&",numRecs)("&",dim  )  = numRecs; //newRecs-1;
                xb("&",numRecs)("&",dim+1)  = betaval;
                xb("&",numRecs)("&",dim+2)  = rmupred;
                xb("&",numRecs)("&",dim+3)  = rsigmapred;
                xb("&",numRecs)("&",dim+4)  = rmupred+standev;
                xb("&",numRecs)("&",dim+5)  = rmupred-standev;
                xb("&",numRecs)("&",dim+6)  = 2*standev;
                xb("&",numRecs)("&",dim+7)  = softmax;
                xb("&",numRecs)("&",dim+8)  = 0; // You need this to ensure vector is not sparse!
                xb("&",numRecs)("&",dim+9)  = (double) bayesruntime;
                xb("&",numRecs)("&",dim+10) = (double) mugptraintime;
                xb("&",numRecs)("&",dim+11) = (double) sigmagptraintime;
                xb("&",numRecs)("&",dim+12) = gridi;
                xb("&",numRecs)("&",dim+13) = gridy;

                numRecs++;
                newRecs--;
            }
        }

        //int sigmamod = 0;

        for ( k = 0 ; k < numRecs ; k++ )
        {
            errstream() << "e";

            // ===============================================================
            // Run experiment
            // ===============================================================

            fnapproxout.force_int() = itcnt+1;
            (*fn)(dim,fnapproxout,&(xb(k)(zeroint())),fnarg,addvar);
            fnapproxout.negate();

            // ===============================================================
            // Add new point to machine learning block
            // ===============================================================

            errstream() << "a(" << fnapproxout << "," << mupred(k) << ";" << sigmapred(k) << ")";

            int addpointpos = 0;

            if ( isgridopt && isgridcache )
            {
                addpointpos = Nbasemu+gridi;

                bopts.modelmu_setd(Nbasemu+gridi,2);

                if ( addvar != 0 )
                {
                    bopts.modelmu_setsigmaweight(Nbasemu+gridi,((bopts.model_sigma())+addvar)/(bopts.model_sigma()));
                }

                if ( bopts.sigmuseparate && ( addvar != 0 ) )
                {
                    bopts.modelsigma_setsigmaweight(Nbasesigma+gridi,((bopts.model_sigma())+addvar)/(bopts.model_sigma()));

                    //sigmamod = 1;
                }
            }

            else
            {
                addpointpos = bopts.modelmu_N();

                bopts.modelmu_addTrainingVector(fnapproxout,mupred(k),xxb(k));
                muapproxsize++;

                if ( addvar != 0 )
                {
//FIXME: suspect this might mess with variance estimations in env-GP
                    bopts.modelmu_setsigmaweight(bopts.modelmu_N()-1,((bopts.model_sigma())+addvar)/(bopts.model_sigma()));
                }

                if ( bopts.sigmuseparate && ( addvar != 0 ) )
                {
                    bopts.modelsigma_setsigmaweight(bopts.modelsigma_N()-(numRecs-k),((bopts.model_sigma())+addvar)/(bopts.model_sigma()));

                    //sigmamod = 1;
                }
            }

            if ( bopts.isimphere() )
            {
                if ( fnapproxout.isValVector() )
                {
                    const Vector<gentype> &ghgh = (const Vector<gentype> &) fnapproxout;

                    for ( j = 0 ; j < fnapproxout.size() ; j++ )
                    {
                        xinb("&",j) = ghgh(j);
                    }
                }

                else
                {
                    xinb("&",zeroint()) = fnapproxout;
                }

                xinb.negate();

                bopts.modelimp_addTrainingVector(nothingmuch,xinb);
            }

            // ===============================================================
            // Sort into y list if stability constraints active
            // ===============================================================

            if ( isstable )
            {
                int jj;

                if ( !( isgridopt && isgridcache ) )
                {
                    for ( jj = 0 ; jj < ysort.size() ; jj++ )
                    {
                        if ( ysort(jj) >= addpointpos )
                        {
                            ysort("&",jj)++;
                        }
                    }
                }

                for ( jj = 0 ; jj < ysort.size() ; jj++ )
                {
                    if ( (bopts.modelmu_y())(addpointpos) < (bopts.modelmu_y())(ysort(jj)) )
                    {
                        break;
                    }
                }

                ysort.add(jj);
                ysort("&",jj) = addpointpos;
            }
        }

        // ===================================================================
        // Train model
        // ===================================================================

        time_used mugpbegintime = TIMECALL;

        errstream() << "t";
        bopts.model_train(dummy,killSwitch);
        bopts.modelimp_train(dummy,killSwitch);
        errstream() << "...";

        time_used mugpendtime = TIMECALL;
        mugptraintime = TIMEDIFFSEC(mugpendtime,mugpbegintime);

        // ===================================================================
        // Update fres and ires
        // ===================================================================

        if ( ( ires == -1 ) || ( fnapproxout > fres ) )
        {
            fres = fnapproxout;
            ires = ( isgridopt && isgridcache ) ? Nbasemu+gridi : (bopts.modelmu_N())-1;
        }

        // ===================================================================
        // Termination condition using model_err
        // ===================================================================

        if ( dofreqstop )
        {
            // See "Adaptive and Safe Bayesian Optimization in High Dimensions via One-Dimensional Subspaces", Kirschner et al

            double model_errpred = bopts.model_err(dim,qmin,qmax,killSwitch);

            //errstream() << "min_x err(x) = " << model_errpred << "\n";
            outstream() << "min_x err(x) = " << model_errpred << "\n";

            if ( model_errpred < bopts.err )
            {
                isopt = 1;
            }
        }

        // ===================================================================
        // Iterate and go again
        // ===================================================================

        if ( !(++itcnt%FEEDBACK_CYCLE) )
        {
            if ( (itcnt/FEEDBACK_CYCLE)%4 == 0 )
            {
                errstream() << "|\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 1 )
            {
                errstream() << "/\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 2 )
            {
                errstream() << "-\b";
            }

            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 3 )
            {
                errstream() << "\\\b";
            }
        }

        if ( !(itcnt%MAJOR_FEEDBACK_CYCLE) )
        {
            errstream() << "=" << itcnt << "=  ";
        }

        if ( xmtrtime > 1 )
        {
            curr_time = TIMECALL;

            if ( TIMEDIFFSEC(curr_time,start_time) > xmtrtime )
            {
                timeout = 1;
            }
        }

        if ( !timeout )
        {
            timeout = kbquitdet("Bayesian optimisation",uservars,varnames,vardescr);
        }

        altitcnt += itcntmethod ? 0 : 1;

        xinf = NULL;
    }

    xinf = NULL;

    NiceAssert( ires >= 0 );

    // =======================================================================
    // Strip out unused pre-cached vectors
    // =======================================================================

//    if ( isgridopt && isgridcache && Ngrid )
//    {
//        while ( gridind.size() )
//        {
//            i = gridind(gridind.size()-1);
//            gridind.remove(gridind.size());
//
//            bopts.model_removeTrainingVector(Nbasemu+i,Nbasesigma+i);
//        }
//    }

    // =======================================================================
    // Final calculation of stability scores
    // =======================================================================

    sscore.resize(muapproxsize);

    if ( isstable )
    {
        sscore = 0.0;

//NB: we *absolutely do not* want to use ysort ordering here!
        retVector<int> tmpva;

        calcsscore(sscore,bopts,cntintvec(sscore.size(),tmpva),stabp,stabpnrm,stabrot,stabmu,stabB);
    }

    else
    {
        sscore = 1.0;
    }

    // =======================================================================
    // Record minimum
    // =======================================================================

    bopts.modelmu_xcopy(xres,ires); 
    // This is OK.  The only case where xres is not locally stored is gridopt, 
    // and in this case we have asserted !isXconvertNonTrivial(), so x and
    // x convert are the same and backconvert will succeed.

    // =======================================================================
    // See note re max/min changes
    // =======================================================================

    setnegate(fres);

    // =======================================================================
    // Done
    // =======================================================================

    return 0;
}


















class fninnerArg
{
    public:

    fninnerArg(int dim,
               int nres,
               void (*_fn)(gentype &, Vector<gentype> &, void *arg),
               void *_arginner,
               int &_ires,
               Vector<Vector<gentype> > &_allxres,
               Vector<gentype> &_allfres,
               Vector<gentype> &_allfresmod,
               gentype &_fres,
               Vector<gentype> &_xres,
               Vector<gentype> &_supres,
               volatile int &_force_stop,
               double &_hardmin,
               double &_hardmax) : fn(_fn),
                                   arginner(_arginner),
                                   ires(_ires),
                                   allxres(_allxres),
                                   allfres(_allfres),
                                   allfresmod(_allfresmod),
                                   fres(_fres),
                                   xres(_xres),
                                   supres(_supres),
                                   force_stop(_force_stop),
                                   hardmin(_hardmin),
                                   hardmax(_hardmax)
    {
        xx.prealloc(dim+1);
        allxres.prealloc(nres);
        allfres.prealloc(nres);
        allfresmod.prealloc(nres);
        supres.prealloc(nres);

        xx.resize(dim);

        return;
    }

    fninnerArg(const fninnerArg &src) : fn(src.fn),
                                   arginner(src.arginner),
                                   ires(src.ires),
                                   allxres(src.allxres),
                                   allfres(src.allfres),
                                   allfresmod(src.allfresmod),
                                   fres(src.fres),
                                   xres(src.xres),
                                   supres(src.supres),
                                   force_stop(src.force_stop),
                                   hardmin(src.hardmin),
                                   hardmax(src.hardmax)
    {
        (void) src;
        throw("Can't duplicate fninnerArg");
        return;
    }

    fninnerArg &operator=(const fninnerArg &src)
    {
        (void) src;
        throw("Can't copy fninnerArg");
        return *this;
    }

    Vector<gentype> xx;
    Vector<gentype> dummyxarg;
    void (*fn)(gentype &, Vector<gentype> &, void *arg);
    void *arginner;
    int &ires;
    Vector<Vector<gentype> > &allxres;
    Vector<gentype> &allfres;
    Vector<gentype> &allfresmod;
    gentype &fres;
    Vector<gentype> &xres;
    Vector<gentype> &supres;
    volatile int &force_stop;
    double &hardmin;
    double &hardmax;

    void operator()(int dim, gentype &res, const double *x, double &addvar)
    {
        // ===========================================================================
        // Inner loop evaluation function.  This is used as a buffer between the
        // actual Bayesian optimiser (above) and the outside-visible optimiser (below)
        // and saves things like timing, beta etc.
        // ===========================================================================

        addvar = 0;

        if ( dim == -1 )
        {
            // This is just to trigger the intermediate command

            (*fn)(res,dummyxarg,arginner);

            return;
        }

        if ( dim )
        {
            int i;

            for ( i = 0 ; i < dim ; i++ )
            {
                xx("&",i).force_double() = x[i];
            }
        }

        int gridi = (int) x[dim+12];

        // gridi will be -1 if this is not grid optimisation.  Otherwise we need
        // to extend the size of xx to argnum+1 and load gridy into it.  The +1 
        // will roll around to zero in mlinter, so the result will automatically 
        // be loaded into result var which will be evaluated and passed back as 
        // result!

        if ( gridi >= 0 )
        {
            xx.resize(dim+1);

            xx("[]",dim) = x[dim+13];
        }

        // =======================================================================
        // Call function and record times
        // =======================================================================

        time_used starttime = TIMECALL;
        (*fn)(res,xx,arginner);
        time_used endtime = TIMECALL;

        if ( gridi >= 0 )
        {
            xx.resize(dim);
        }

        if ( res.isValSet() )
        {
            // Modified variance is returned as the second element of a set

            res    = (res.all())(zeroint());
            addvar = (double) (res.all())(1);
        }

        if ( !(res.isValVector()) )
        {
            if ( ( allfres.size() == 0 ) || ( res < fres ) )
            {
                ires = allfres.size();
                fres = res;
                xres = xx;
            }

            if ( (double) res <= hardmin )
            {
                // Trigger early termination if hardmin reached

                force_stop = 1;
            }

            else if ( (double) res >= hardmax )
            {
                // Trigger early termination if hardmax reached

                force_stop = 1;
            }
        }

        // =======================================================================
        // Store results if required
        // =======================================================================

        if ( 1 )
        {
            allxres.append(allxres.size(),xx);
            allfres.append(allfres.size(),res);
            allfresmod.append(allfresmod.size(),res);
            supres.add(supres.size());

            double dstandev = x[dim+6];
            double softmax  = x[dim+7];

            double ucbdist = softmax - ( fres.isCastableToReal() ? ( (double) fres ) : 0.0 );
            double sigbnd  = ( ucbdist < dstandev ) ? ucbdist : dstandev;

            supres("&",supres.size()-1).force_vector().resize(17);

            supres("&",supres.size()-1)("&",0)  = TIMEABSSEC(starttime);
            supres("&",supres.size()-1)("&",1)  = TIMEABSSEC(endtime);
            supres("&",supres.size()-1)("&",2)  = x[dim];    // numRecs
            supres("&",supres.size()-1)("&",3)  = x[dim+1];  // beta
            supres("&",supres.size()-1)("&",4)  = x[dim+2];  // mu
            supres("&",supres.size()-1)("&",5)  = x[dim+3];  // sigma
            supres("&",supres.size()-1)("&",6)  = x[dim+4];  // UCB
            supres("&",supres.size()-1)("&",7)  = x[dim+5];  // LCB
            supres("&",supres.size()-1)("&",8)  = dstandev;  // DVAR
            supres("&",supres.size()-1)("&",9)  = ucbdist;   // UCBDIST
            supres("&",supres.size()-1)("&",10) = sigbnd;    // SIGBND
            supres("&",supres.size()-1)("&",11) = x[dim+9];  // DIRect runtime
            supres("&",supres.size()-1)("&",12) = x[dim+10]; // mu model training time
            supres("&",supres.size()-1)("&",13) = x[dim+11]; // sigma model training time
            supres("&",supres.size()-1)("&",14) = TIMEDIFFSEC(endtime,starttime);
            supres("&",supres.size()-1)("&",15) = x[dim+12]; // grid index (-1 if none)
            supres("&",supres.size()-1)("&",16) = x[dim+13]; // known grid evaluation (0.0 if none)
       }

        // Resize x (it is able to be changed, so this is important)

        xx.resize(dim);

        return;
    }
};



// ===========================================================================
// Inner loop evaluation function.  This is used as a buffer between the
// actual Bayesian optimiser (above) and the outside-visible optimiser (below)
// and saves things like timing, beta etc.
// ===========================================================================

void fninner(int dim, gentype &res, const double *x, void *arg, double &addvar);
void fninner(int dim, gentype &res, const double *x, void *arg, double &addvar)
{
    (*((fninnerArg *) arg))(dim,res,x,addvar);
    return;
}




// ===========================================================================
// ===========================================================================
// ===========================================================================
// Outer loop bayesian optimiser.
// ===========================================================================
// ===========================================================================
// ===========================================================================

int bayesOpt(int dim,
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
              BayesOptions &bopts,
              svmvolatile int &force_stop)
{
    NiceAssert( dim > 0 );
    NiceAssert( xmin.size() == dim );
    NiceAssert( xmax.size() == dim );

    double hardmin = bopts.hardmin;
    double hardmax = bopts.hardmax;

    allxres.resize(0);
    allfres.resize(0);
    allfresmod.resize(0);
    supres.resize(0);
    sscore.resize(0);

    Vector<double> locxres;

    Vector<double> locxmin(dim);
    Vector<double> locxmax(dim);

    int i,j,k;
    gentype locfres(0.0);

    for ( i = 0 ; i < dim ; i++ )
    {
        locxmin("&",i) = (double) xmin(i);
        locxmax("&",i) = (double) xmax(i);
    }

    fninnerArg optargs(dim,
                       ( ( bopts.startpoints == -1 ) ? dim+1 : bopts.startpoints ) + ( ( bopts.totiters == -1 ) ? 10*dim : bopts.totiters ),
                       fn,
                       fnarg,
                       ires,
                       allxres,
                       allfres,
                       allfresmod,
                       fres,
                       xres,
                       supres,
                       force_stop,
                       hardmin,
                       hardmax);

    int res = bayesOpt(dim,locxres,locfres,locxmin,locxmax,fninner,(void *) &optargs,bopts,force_stop,sscore);
    int isstable = bopts.stabpmax;

    force_stop = 0; // Need to reset this trigger so that subsequent runs don't get hit (it is tripped by hardmin/hardmax)

    if ( bopts.unscentUse )
    {
        // Post-calculation for unscented optimisation

        int unscentK                           = bopts.unscentK;
        const Matrix<double> &unscentSqrtSigma = bopts.unscentSqrtSigma;

        NiceAssert( unscentSqrtSigma.numCols() == unscentSqrtSigma.numRows() );

        int N = allfres.size();
        int d = unscentSqrtSigma.numRows();

        if ( N )
        {
            ires = -1;

            gentype tempres;
            SparseVector<gentype> xxx;
            double modres = 0.0;

            for ( k = 0 ; k < N ; k++ )
            {
                for ( j = 0 ; j < d ; j++ )
                {
                    xxx("&",j) = allxres(k)(j);
                }

                bopts.model_gg(tempres,xxx);
                modres = (((double) unscentK)/((double) (d+unscentK)))*((double) tempres);

                for ( i = 0 ; i < d ; i++ )
                {
                    for ( j = 0 ; j < d ; j++ )
                    {
                        xxx("&",j) = (allxres(k))(j) + sqrt(d+unscentK)*unscentSqrtSigma(i,j);
                    }

                    bopts.model_gg(tempres,xxx);
                    modres += ((double) tempres)/((double) (2*(d+unscentK)));

                    for ( j = 0 ; j < d ; j++ )
                    {
                        xxx("&",j) = (allxres(k))(j) - sqrt(d+unscentK)*unscentSqrtSigma(i,j);
                    }

                    bopts.model_gg(tempres,xxx);
                    modres += ((double) tempres)/((double) (2*(d+unscentK)));
                }

                allfresmod("&",k) = -modres; // Don't forget all that negation stuff

                if ( ( ires == -1 ) || ( allfresmod(k) < allfresmod(ires) ) )
                {
                    ires = k;
                }
            }

            NiceAssert( ires >= 0 );

            fres = allfres(ires);
            xres = allxres(ires);
        }
    }

    if ( isstable )
    {
//FIXME: at this point need to modify allfresmod to include sscore
        // Need to re-analyse results to find optimal result *that satisfies gradient constraints*

        int N = allfres.size();

        if ( N )
        {
            ires = -1;

            for ( k = 0 ; k < N ; k++ )
            {
                if ( bopts.stabUseSig )
                {
//allfresmod("&",k) *= ( ( sscore(k) > bopts.stabThresh ) ? 1.0 : DISCOUNTRATE );
allfresmod("&",k) *= 1/(1+exp(-1000*(sscore(k)-bopts.stabThresh)));
//                    allfresmod("&",k) *= 1/(1+exp(-(sscore(k)-bopts.stabThresh)/(sscore(k)*(1-sscore(k)))));
                }

                else
                {
                    allfresmod("&",k) *= sscore(k); // Allows us to do unscented and stable together
                }

                if ( ( ires == -1 ) || ( allfresmod(k) < allfresmod(ires) ) )
                {
                    ires = k;
                }
            }

            NiceAssert( ires >= 0 );

            fres = allfres(ires);
            xres = allxres(ires);
        }
    }

//    xres.resize(dim);
//
//    for ( i = 0 ; i < dim ; i++ )
//    {
//        xres("&",i) = locxres(i);
//    }

    return res;
}
