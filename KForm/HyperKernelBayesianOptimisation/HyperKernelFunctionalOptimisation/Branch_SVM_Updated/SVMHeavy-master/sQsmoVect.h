
//
// Sparse quadratic solver - large scale, loosely SMO base, vector target, warm start
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _sQsmoVect_h
#define _sQsmoVect_h
#include "vector.h"
#include "matrix.h"
#include "optstate.h"
#include "smatrix.h"
#include "zerocross.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctime>

// Pretty self explanatory: you give is a state and relevant matrices and it
// will solve:
//
// [ alpha ]' [ Gp   Gpn ] [ alpha ] + [ alpha ]' [ gp ] + | alpha' |' [ hpeff ]
// [ beta  ]  [ Gpn' Gn  ] [ beta  ]   [ beta  ]  [ gn ]   | beta   |  [ 0     ]
//
// such that: alpha_i, beta, gp_i and gn are vector-valued
//            Gp_ij is real, anionic, or real-valued matrix**
//            | alpha_i |_q <= ubeff
//
// to within precision optsol.  It is assumed that:
//
// - Gp is positive semi-definite hermitian
// - Gpn is a size(Gp)*1 matrix of all 1s
// - Gn is a 1*1 zero matrix
// - ubeff and hpeff are *not* included in the factorisation
// - GpnRowTwoSigned = 0
// - fixHigherOrderTerms = NULL
//
// Will return 0 on success or an error code otherwise
//
// **Gp_ij matrix must be single-valued on the diagonal and anti-symmetric
// in the off-diagonals.  Gpsigma calculated from this will be effectively
// real (identity matrix with single-valued diagonal), so is stored as
// a real matrix.


template <class S>
class fullOptStateSMOVect : public fullOptState<Vector<double>,S>
{
public:

    fullOptStateSMOVect(optState<Vector<double>,S> &_x,
        const Matrix<S> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<Vector<double> > &_gp, const Vector<Vector<double> > &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub,
        double (*_fixHigherOrderTerms)(fullOptState<Vector<double>,S> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *_htArg = NULL,
        double _stepscalefactor = 1,
        int _zcmaxitcnt = DEFZCMAXITCNT, int _inmaxitcnt = DEFINMAXITCNT, double _kappa = DEFAULT_OPTTOL, double _stol = DEFAULT_ZTOL, double _iota = DEFAULT_ZTOL, int _vdim = -1)
        : fullOptState<Vector<double>,S>(_x,_Gp,_Gpsigma,_Gn,_Gpn,_gp,_gn,_hp,_lb,_ub,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        NiceAssert( !_fixHigherOrderTerms );

        zcmaxitcnt = _zcmaxitcnt;
        inmaxitcnt = _inmaxitcnt;
        kappa      = _kappa;
        stol       = _stol;
        iota       = _iota;
        vdim       = _vdim;

        return;
    }

    fullOptStateSMOVect(optState<Vector<double>,S> &_x,
        const Matrix<S> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<Vector<double> > &_gp, const Vector<Vector<double> > &_gn,
        const Vector<double> &_lb, const Vector<double> &_ub,
        double (*_fixHigherOrderTerms)(fullOptState<Vector<double>,S> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *_htArg = NULL,
        double _stepscalefactor = 1,
        int _zcmaxitcnt = DEFZCMAXITCNT, int _inmaxitcnt = DEFINMAXITCNT, double _kappa = DEFAULT_OPTTOL, double _stol = DEFAULT_ZTOL, double _iota = DEFAULT_ZTOL, int _vdim = -1)
        : fullOptState<Vector<double>,S>(_x,_Gp,_Gpsigma,_Gn,_Gpn,_gp,_gn,_lb,_ub,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        NiceAssert( !_fixHigherOrderTerms );

        zcmaxitcnt = _zcmaxitcnt;
        inmaxitcnt = _inmaxitcnt;
        kappa      = _kappa;
        stol       = _stol;
        iota       = _iota;
        vdim       = _vdim;

        return;
    }

    fullOptStateSMOVect(optState<Vector<double>,S> &_x,
        const Matrix<S> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<Vector<double> > &_gp, const Vector<Vector<double> > &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub, const Vector<double> &_GpnRowTwoMag,
        double (*_fixHigherOrderTerms)(fullOptState<Vector<double>,S> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *_htArg = NULL,
        double _stepscalefactor = 1,
        int _zcmaxitcnt = DEFZCMAXITCNT, int _inmaxitcnt = DEFINMAXITCNT, double _kappa = DEFAULT_OPTTOL, double _stol = DEFAULT_ZTOL, double _iota = DEFAULT_ZTOL, int _vdim = -1)
        : fullOptState<Vector<double>,S>(_x,_Gp,_Gpsigma,_Gn,_Gpn,_gp,_gn,_hp,_lb,_ub,_GpnRowTwoMag,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        NiceAssert( !_fixHigherOrderTerms );

        zcmaxitcnt = _zcmaxitcnt;
        inmaxitcnt = _inmaxitcnt;
        kappa      = _kappa;
        stol       = _stol;
        iota       = _iota;
        vdim       = _vdim;

        return;
    }

    virtual ~fullOptStateSMOVect() { return; }

    // Overwrite just the matrices/vectors, copy the rest

    virtual fullOptState<Vector<double>,S> *gencopy(int _chistart,
        const Matrix<S> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<Vector<double> > &_gp, const Vector<Vector<double> > &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub, const Vector<double> &_GpnRowTwoMag)
    {
        fullOptStateSMOVect<S> *res;

        MEMNEW(res,fullOptStateSMOVect<S>(fullOptState<Vector<double>,S>::x,_Gp,_Gpsigma,_Gn,_Gpn,_gp,_gn,_hp,_lb,_ub,_GpnRowTwoMag));

        copyvars(res,_chistart);

        return static_cast<fullOptState<Vector<double>,S> *>(res);
    }

// private:

    int zcmaxitcnt;
    int inmaxitcnt;
    double kappa;
    double stol;
    double iota;
    int vdim;

    // Actual quadratic optimiser

    virtual int solve(svmvolatile int &killSwitch);

    virtual void copyvars(fullOptState<Vector<double>,S> *dest, int _chistart)
    {
        fullOptState<Vector<double>,S>::copyvars(dest,_chistart);

        fullOptStateSMOVect *ddest = static_cast<fullOptStateSMOVect *>(dest);

        ddest->zcmaxitcnt = zcmaxitcnt;
        ddest->inmaxitcnt = inmaxitcnt;
        ddest->kappa      = kappa;
        ddest->stol       = stol;
        ddest->iota       = iota;
        ddest->vdim       = vdim;

        return;
    }
};























































template<class S>
int solve_quadratic_program_smoVect(svmvolatile int &killSwitch, optState<Vector<double>,S> &op, 
    const Matrix<S> &GpGrad, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, 
    const Vector<Vector<double> > &gp, const Vector<Vector<double> > &gn, const Vector<double> &hpeff, 
    const Vector<double> &ubeff, 
    int maxitcnt, int maxtraintime, 
    int zcmaxitcnt, int inmaxitcnt, double kappa, double stol, double iota, int vdim);


template<class S>
int fullOptStateSMOVect<S>::solve(svmvolatile int &killSwitch)
{
    if ( vdim == -1 )
    {
        vdim = (fullOptState<Vector<double>,S>::gn)(zeroint()).size();
    }

    NiceAssert( !(fullOptState<Vector<double>,S>::GpnRowTwoSigned) );
    NiceAssert( vdim == (fullOptState<Vector<double>,S>::gn)(zeroint()).size() );
    NiceAssert( (fullOptState<Vector<double>,S>::x).bN() == 1 );
    NiceAssert( (fullOptState<Vector<double>,S>::maxitcnt) >= 0 );
//    NiceAssert( !(fullOptState<Vector<double>,S>::fixHigherOrderTerms) );

    return solve_quadratic_program_smoVect(killSwitch,fullOptState<Vector<double>,S>::x,
        fullOptState<Vector<double>,S>::Gp,fullOptState<Vector<double>,S>::Gpsigma,fullOptState<Vector<double>,S>::Gn,fullOptState<Vector<double>,S>::Gpn,
        fullOptState<Vector<double>,S>::gp,fullOptState<Vector<double>,S>::gn,fullOptState<Vector<double>,S>::hp,
        fullOptState<Vector<double>,S>::ub,
        fullOptState<Vector<double>,S>::maxitcnt,fullOptState<Vector<double>,S>::maxruntime,
        zcmaxitcnt,inmaxitcnt,kappa,stol,iota,vdim);
}







#define FEEDBACK_CYCLE 50
#define MAJOR_FEEDBACK_CYCLE 1000


// Outer loop parameters for barrier method

#define TMIN 10
#define TMAX 10000
#define MUFACTOR 30
#define STEPSPERBLOCK 2




template <class T>
class smoVectScratch;

//int solve_quadratic_program_smoVect(optState<Vector<double>,T> &op, const Matrix<T> &GpGrad, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<Vector<double> > &gp, const Vector<Vector<double> > &gn, const Vector<double> &hpeff, const Vector<double> &ubeff, int maxitcnt, int maxtraintime, int zcmaxitcnt, int inmaxitcnt, double kappa, double ztol, double stol, double mu, double iota, int vdim);
template <class T> int optSMOVect(svmvolatile int &killSwitch, optState<Vector<double>,T> &op, const Matrix<double> &Gp, const Matrix<T> &GpGrad, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<Vector<double> > &gp, const Vector<Vector<double> > &gn, const Vector<double> &hpeff, const Vector<double> &ubeff, int maxitcnt, int maxtraintime, smoVectScratch<T> &pad, int vdim);
template <class T> int innerLoop(smoVectScratch<T> &pad, const Vector<double> &startAlphai, const Vector<double> &startAlphaj, const Vector<double> &startei, const Vector<double> &startej);
template <class T> int calcBiasStep(smoVectScratch<T> &pad, const Vector<double> &startei, const Vector<double> &startej);
template <class T> int stepFind(smoVectScratch<T> &pad, int testdiscontinuities);
template <class T> Vector<double> &dirFind(smoVectScratch<T> &pad);
template <class T> double lineSearch(smoVectScratch<T> &pad);






template <class T>
class smoVectScratch
{
public:

    smoVectScratch(int dim = 0)
    {
        DeltaBias.resize(dim);
        dbi.resize(dim);
        dbj.resize(dim);

        alphai.resize(dim);
        alphaj.resize(dim);
        eij.resize(dim);
        ascr.resize(dim);

        g0.resize(dim);
        DeltaAlpha.resize(dim);
    }

    // Given information

    T Kii; // used to calculate bias step
    T Kij; // used to calculate bias step
    T Kjj; // used to calculate bias step
    double etaij;
    double CNCi;
    double CNCj;
    double epsiloni;
    double epsilonj;

    int zcmaxitcnt; // zero tolerance max iteration count
    int inmaxitcnt; // inner loop max iteration count
    double t;       // see Boyd
    double iota;    // boundary tolerance
    double kappa;   // gradient optimality tolerance
    double ztol;    // distance tolerance from discontinuities
    double stol;    // step size "significance" tolerance

    // Filled by calcBiasStep

    Vector<double> DeltaBias;
    Vector<double> dbi;
    Vector<double> dbj;

    // Filled by innerLoop function

    Vector<double> alphai;
    Vector<double> alphaj;
    Vector<double> eij;
    Vector<double> ascr;

    // Filled by stepFind function

    double normalphai;
    double normalphaj;
    double absalphai;
    double absalphaj;
    double alphaijprod;
    double normalphaipalphaj;
    double absalphaipalphaj;
    int atAlphai;
    int atAlphaj;
    int alphaieqnegalphaj;
    int havegrad;
    double ascale;
    double bscale;
    Vector<double> g0;
    double absg0;
    double sM;
    int steptodisci; // set if step ends at discontinuity i
    int steptodiscj; // set if step ends at discontinuity j

    // Filled by dirFind function

    double chii;
    double chij;
    double zetai;
    double zetaj;
    double betai;
    double betaj;
    double etatilde;
    double Mii;
    double Mjj;
    double Mij;
    double Mdet;
    double Minvii;
    double Minvjj;
    double Minvij;
    double ki;
    double kj;
    double zi;
    double zj;
    Vector<double> DeltaAlpha;

    // Filled by lineSearch function

    double xabs;
    double thetai;
    double thetaj;
    double tildeeij;
    double tildesM;
    double CNCisq;
    double CNCjsq;
    double lambdaip;
    double lambdajp;
    double lambda;
    int isidiscont;
    int isjdiscont;
    double tildesmin;
    double tildesmax;
    double tgrad;
    double qi;
    double qj;
    double ri;
    double rj;
    double tti;
    double ttj;
    double qgrad;
};








template <class T> 
int solve_quadratic_program_smoVect(svmvolatile int &killSwitch, optState<Vector<double>,T> &op, const Matrix<T> &GpGrad, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<Vector<double> > &gp, const Vector<Vector<double> > &gn, const Vector<double> &hpeff, const Vector<double> &ubeff, int maxitcnt, int maxtraintime, int zcmaxitcnt, int inmaxitcnt, double kappa, double stol, double iota, int vdim)
{
    double ztol = op.zerotol();

    // NB: Gpsigma is often used as a dummy with op "requires" Gp (although
    // it doesn't actually, as this is only needed if the factorisation is
    // kept, which it is not).  Note that this is just a reference.

    const Matrix<double> &Gp = Gpsigma;

    if ( !(op.betaState(0)) )
    {
        op.modBetaCtoFhpzero(0,Gp,GpGrad,Gn,Gpn,gp,gn);
    }

    op.refreshGradhpzero(GpGrad,Gn,Gpn,gp,gn);

    smoVectScratch<T> pad;

    pad.zcmaxitcnt = zcmaxitcnt;
    pad.inmaxitcnt = inmaxitcnt;
    pad.kappa      = kappa;
    pad.ztol       = ztol;
    pad.stol       = stol;
    pad.iota       = iota;

    // See Boyd, around page 569, for an explanation here

    int isopt = 0;

    for ( pad.t = TMIN ; pad.t <= TMAX ; pad.t *= MUFACTOR )
    {
//errstream() << "\n\n\nphantomx: call outer loop t = " << pad.t << "\n";
        isopt = optSMOVect(killSwitch,op,Gp,GpGrad,Gpsigma,Gn,Gpn,gp,gn,hpeff,ubeff,maxitcnt,maxtraintime,pad,vdim);
    }

    return isopt ? 0 : 1;
}






template <class T> 
int optSMOVect(svmvolatile int &killSwitch, optState<Vector<double>,T> &op, 
const Matrix<double> &Gp, const Matrix<T> &GpGrad, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, 
const Vector<Vector<double> > &gp, const Vector<Vector<double> > &gn, const Vector<double> &hpeff, 
const Vector<double> &ubeff, 
int maxitcntint, int maxtraintime, smoVectScratch<T> &pad, int vdim)
{
    int N = op.aN();

    Vector<double> eup(N);
    Vector<double> fup(N);
    Vector<double> em(N);
    Vector<double> fm(N);
    Vector<int> epivot(N);
    Vector<int> fpivot(N);
    Vector<Vector<double> > g(N);
    double absg;
    double absa;

    int i,j,k,l;
    int Nbad,Ngood,Nposs;
    int allorF;
    int stepfound,newstep;
    int Navail = 0;

    // Count available vectors

    for ( i = 0 ; i < N ; i++ )
    {
        if ( op.alphaRestrict(i) != 3 )
        {
            Navail++;
        }
    }

    // Setup g vector

    for ( i = 0 ; i < N ; i++ )
    {
        g("&",i).resize(vdim);
    }

    // Construct ordering based on projection to axis testaxis, which
    // increments for each loop.

    allorF = 0; // 1 means everything, 0 means only free

    double maxitcnt = maxitcntint;
    double xmtrtime = maxtraintime;
    double *uservars[] = { &maxitcnt, &xmtrtime, NULL };
    const char *varnames[] = { "itercount", "traintime", NULL };
    const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", NULL };

    int res = 1;
    int isopt = 0;
    time_used start_time = TIMECALL;
    time_used curr_time = start_time;
    unsigned long long itcnt = 0;
    int timeout = 0;
    int bailout = 0;

    // Obscure note: in c++, if maxitcnt is a double then !maxitcnt is
    // true if maxitcnt == 0, false otherwise.  This is defined in the
    // standard, and the reason the following while statement will work.

    while ( !killSwitch && !isopt && ( ( itcnt < (unsigned int) maxitcnt ) || !maxitcnt ) && !timeout && !bailout )
    {
        if ( allorF )
        {
            Ngood = 0;
            Nposs = Navail; //N;
            Nbad  = Nposs;
        }

        else
        {
            Ngood = 0;
            Nposs = op.aNF(); // NB restricted variables are never free
            Nbad  = Nposs;
        }

        // Ngood will track the number of optimal alphas in the search set
        // Nbad will track the number of non-optimal in same

        // Calculate gradients

//errstream() << "phantomxa: Calculating gradients\n";
        j = 0;

        for ( i = 0 ; i < Nposs ; i++,j++ )
        {
            if ( allorF )
            {
                k = j;

                while ( op.alphaRestrict(k) == 3 )
                {
                    k = ++j;
                }
            }

            else
            {
                k = (op.pivAlphaF())(i);
            }

            g("&",k) = op.alphaGrad(k);
            absa = abs2(op.alpha(k));

            if ( op.alphaState(k) && ( absa > 0 ) )
            {
                g("&",k).scaleAdd((hpeff(k)+(1/((pad.t)*(ubeff(k)-absa))))/absa,op.alpha(k));
            }

            else
            {
                absg = abs2(g(k));

                if ( absg > hpeff(k)+(1/((pad.t)*(ubeff(k)))) )
                {
                    g("&",k) *= (absg-(hpeff(k)+(1/((pad.t)*(ubeff(k))))))/absg;
                }

                else
                {
                    g("&",k) = 0.0;
                }
            }

            if ( abs2(g(k)) <= pad.kappa )
            {
                Ngood++;
                Nbad--;
            }

            em("&",k) = norm2(g(k));
            eup("&",i) = em(k);
            epivot("&",i) = k;
        }
//errstream() << "phantomxa: eup = " << eup(zeroint(),1,Nposs-1) << "\n";
//errstream() << "phantomxa: epivot = " << epivot(zeroint(),1,Nposs-1) << "\n";
//errstream() << "phantomxa: g = " << g(epivot(zeroint(),1,Nposs-1)) << "\n";
//errstream() << "phantomxa: alpha = " << (op.alpha())(epivot(zeroint(),1,Nposs-1)) << "\n";
//errstream() << "phantomxa: alphaGrad = " << (op.alphaGrad())(epivot(zeroint(),1,Nposs-1)) << "\n";

        // If not optimal, do exhaustive search until a step is found

        if ( Nbad )
        {
            // Sort gradients based on projections

//errstream() << "phantomxa: Sorting gradients\n";
            for ( i = 1 ; i < Nposs ; i++ )
            {
                for ( j = i-1 ; j >= 0 ; j-- )
                {
                    if ( eup(j) > eup(i) )
                    {
                        break;
                    }
                }

                epivot.blockswap(i,j+1);
                eup.blockswap(i,j+1);
            }

            // Work through vectors starting with least optimal

            stepfound = 0;

            for ( i = 0 ; ( i < Nposs-1 ) && ( stepfound < STEPSPERBLOCK ) ; i++ )
            {
                // Calculate ||g(i)-g(j)||^2

                retVector<int> tmpva;
                retVector<int> tmpvb;

                fpivot("&",0,1,Nposs-1-i-1,tmpva) = epivot(i+1,1,Nposs-1,tmpvb);

                double temp;

                for ( j = 0 ; j < Nposs-1-i ; j++ )
                {
                    // ||x-y||^2 = ||x||^2 + ||y||^2 - 2.<x,y>

                    k = epivot(i);
                    l = fpivot(j);

                    fm("&",l) = em(k)+em(l)-(2*twoProductNoConj(temp,g(k),g(l)));
                    fup("&",j) = fm(l);
                }

                // Sort gradient difference

                for ( k = 1 ; k < Nposs-1-i ; k++ )
                {
                    for ( l = k-1 ; l >= 0 ; l-- )
                    {
                        if ( fup(l) > fup(k) )
                        {
                            break;
                        }
                    }

                    fpivot.blockswap(k,l+1);
                    fup.blockswap(k,l+1);
                }

                // Work through second choices to find a step

                for ( j = 0 ; ( j < Nposs-1-i ) && ( stepfound < STEPSPERBLOCK ) ; j++ )
                {
                    k = epivot(i);
                    l = fpivot(j);

                    pad.Kii      = GpGrad(k,k);
                    pad.Kjj      = GpGrad(l,l);
                    pad.Kij      = GpGrad(k,l);
                    pad.etaij    = Gpsigma(k,l);
                    pad.CNCi     = ubeff(k);
                    pad.CNCj     = ubeff(l);
                    pad.epsiloni = hpeff(k);
                    pad.epsilonj = hpeff(l);

//errstream() << "phantomx 4 - " << k << "," << l << "\n";
                    innerLoop(pad,op.alpha(k),op.alpha(l),op.alphaGrad(k),op.alphaGrad(l));

//errstream() << "phantomx 6\n";
                    newstep = 0;

                    if ( abs2(pad.DeltaAlpha) > pad.stol )
                    {
//errstream() << "phantomx 6x " << pad.DeltaAlpha << " - " << pad.stol << "\n";
                        stepfound++;
                        newstep = 1;
                    }

//errstream() << "phantomx 5a " << pad.DeltaAlpha << "\n";
//errstream() << "phantomx 5b " << pad.DeltaBias << "\n";
                    op.betaStephpzero(0,pad.DeltaBias,GpGrad,Gn,Gpn,gp,gn);

//errstream() << "phantomx 7\n";
                    if ( newstep && !(op.alphaState(k)) && !(pad.steptodisci) )
                    {
                        // From Z to NZ

                        op.modAlphaZtoUFhpzero(op.findInAlphaZ(k),Gp,GpGrad,Gn,Gpn,gp,gn);
                        op.alphaStephpzero(k,pad.DeltaAlpha,GpGrad,Gn,Gpn,gp,gn);
                    }

                    else if ( newstep && (op.alphaState(k)) && (pad.steptodisci) )
                    {
                        // From NZ to Z

                        op.alphaStephpzero(k,pad.DeltaAlpha,GpGrad,Gn,Gpn,gp,gn);
                        op.modAlphaUFtoZhpzero(op.findInAlphaF(k),Gp,GpGrad,Gn,Gpn,gp,gn);
                    }

                    else if ( newstep && (op.alphaState(k)) && !(pad.steptodisci) )
                    {
                        // From NZ to NZ

                        op.alphaStephpzero(k,pad.DeltaAlpha,GpGrad,Gn,Gpn,gp,gn);
                    }

//errstream() << "phantomx 8\n";
                    (pad.DeltaAlpha).negate();

//errstream() << "phantomx 9\n";
                    if ( newstep && !(op.alphaState(l)) && !(pad.steptodiscj) )
                    {
                        // From Z to NZ

                        op.modAlphaZtoUFhpzero(op.findInAlphaZ(l),Gp,GpGrad,Gn,Gpn,gp,gn);
                        op.alphaStephpzero(l,pad.DeltaAlpha,GpGrad,Gn,Gpn,gp,gn);
                    }

                    else if ( newstep && (op.alphaState(l)) && (pad.steptodiscj) )
                    {
                        // From NZ to Z

                        op.alphaStephpzero(l,pad.DeltaAlpha,GpGrad,Gn,Gpn,gp,gn);
                        op.modAlphaUFtoZhpzero(op.findInAlphaF(l),Gp,GpGrad,Gn,Gpn,gp,gn);
                    }

                    else if ( newstep && (op.alphaState(l)) && !(pad.steptodiscj) )
                    {
                        // From NZ to NZ

                        op.alphaStephpzero(l,pad.DeltaAlpha,GpGrad,Gn,Gpn,gp,gn);
                    }
                }
            }

//errstream() << "phantomx 11 " << stepfound << "\n";
            if ( !stepfound && allorF )
            {
                break;
            }

            else if ( !stepfound && !allorF )
            {
                allorF = 1;
            }

            else
            {
                allorF = 0;
            }
        }

        else if ( !allorF )
        {
            allorF = 1;
        }

        else
        {
            isopt = 1;
            res = 0;
        }

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
            timeout = kbquitdet("Vectorial SMO (sQsmoVect) optimisation",uservars,varnames,vardescr);
        }
    }

    return res;
}
    



template <class T> 
int innerLoop(smoVectScratch<T> &pad, const Vector<double> &startAlphai, const Vector<double> &startAlphaj, const Vector<double> &startei, const Vector<double> &startej)
{
    int isopt = 0;
    int steptaken = 0;
    int newsteptaken = 0;
    int firststep = 1;
    int itcnt = 0;

    pad.alphai = startAlphai;
    pad.alphaj = startAlphaj;
    pad.eij =  startei;
    pad.eij -= startej;
//errstream() << "\nphantomxa 0 " << pad.alphai << "\n";
//errstream() << "phantomxa 1 " << pad.alphaj << "\n";
//errstream() << "phantomxa 2 " << startei << "\n";
//errstream() << "phantomxa 2 " << startej << "\n";
//errstream() << "phantomxa 2 " << pad.eij << "\n";
//errstream() << "phantomxa 3 " << pad.etaij << "\n";

//errstream() << "Inner loop calculate step\n";
    while ( !isopt && ( itcnt < pad.inmaxitcnt ) )
    {
        // Calculate step

        isopt = stepFind(pad,firststep);
        newsteptaken = ( abs2(pad.DeltaAlpha) >= pad.stol );
//errstream() << "phantomxyz 0: " << pad.DeltaAlpha << "\n";
        steptaken |= newsteptaken;
        firststep = 0;

        // Take step (if there is one), break otherwise

        if ( newsteptaken )
        {
            pad.alphai += pad.DeltaAlpha;
            pad.alphaj -= pad.DeltaAlpha;

            (pad.eij).scaleAdd(pad.etaij,pad.DeltaAlpha);
        }

        else
        {
            break;
        }

        itcnt++;
    }

//errstream() << "phantomxyz 1: " << pad.alphai << "\n";
//errstream() << "phantomxyz 2: " << pad.alphaj << "\n";
    if ( steptaken )
    {
        // calculate bias change if required.

        pad.DeltaAlpha =  pad.alphai;
        pad.DeltaAlpha -= startAlphai;
//errstream() << "phantomxyz 3: " << pad.DeltaAlpha << "\n";

        if ( !(pad.steptodisci) )
        {
            pad.ascr =  pad.alphai;
            pad.ascr += pad.DeltaAlpha;

            pad.steptodisci = ( abs2(pad.ascr) < pad.ztol );
        }

        if ( !(pad.steptodiscj) )
        {
            pad.ascr =  pad.alphaj;
            pad.ascr -= pad.DeltaAlpha;

            pad.steptodiscj = ( abs2(pad.ascr) < pad.ztol );
        }

        calcBiasStep(pad,startei,startej);
    }

    else
    {
        pad.DeltaBias = pad.DeltaAlpha;
        (pad.DeltaBias).zero();
    }
//errstream() << "phantomxyz 4: " << pad.DeltaBias << "\n";

    return isopt;
}



// Calculate the bias step

template <class T> 
int calcBiasStep(smoVectScratch<T> &pad, const Vector<double> &startei, const Vector<double> &startej)
{
    int numcerts = 0;

    pad.DeltaBias = 0.0;

    if ( !(pad.steptodisci) ) { numcerts++; }
    if ( !(pad.steptodiscj) ) { numcerts++; }

    if ( numcerts )
    {
        if ( !(pad.steptodisci) )
        {
            pad.absalphai = abs2(pad.alphai);
            pad.ascale = ( pad.epsiloni + (1/((pad.t)*((pad.CNCi)-(pad.absalphai)))) ) / (pad.absalphai);

            pad.dbi = startei;
            (pad.dbi) += ((pad.Kii)-(pad.Kij))*(pad.DeltaAlpha);
            (pad.dbi).scaleAdd(pad.ascale,pad.alphai);

            pad.DeltaBias -= pad.dbi;
        }

        if ( !(pad.steptodiscj) )
        {
            pad.absalphaj = abs2(pad.alphaj);
            pad.ascale = ( pad.epsilonj + (1/((pad.t)*((pad.CNCj)-(pad.absalphaj)))) ) / (pad.absalphaj);

            pad.dbj = startej;
            (pad.dbj) += ((pad.Kij)-(pad.Kjj))*(pad.DeltaAlpha);
            (pad.dbj).scaleAdd(pad.ascale,pad.alphaj);

            pad.DeltaBias -= pad.dbj;
        }

        pad.DeltaBias /= (double) numcerts;
    }

    else
    {
        pad.absalphai = abs2(pad.alphai);
        //pad.ascale = ( pad.epsiloni + (1/((pad.t)*((pad.CNCi)-(pad.absalphai)))) ) / (pad.absalphai);

        pad.dbi = startei;
        (pad.dbi) += ((pad.Kii)-(pad.Kij))*(pad.DeltaAlpha);
        //(pad.dbi).scaleAdd(pad.ascale,pad.alphai);

        pad.absalphaj = abs2(pad.alphaj);
        //pad.ascale = ( pad.epsilonj + (1/((pad.t)*((pad.CNCj)-(pad.absalphaj)))) ) / (pad.absalphaj);

        pad.dbj = startej;
        (pad.dbj) += ((pad.Kij)-(pad.Kjj))*(pad.DeltaAlpha);
        //(pad.dbj).scaleAdd(pad.ascale,pad.alphaj);

        double s = 0;
        double si = 0;
        double sj = 0;
        int isvalid = 0;

        while ( !isvalid )
        {
            s = (si+sj)/2;

            pad.DeltaBias = pad.dbi;
            pad.DeltaBias *= -s;
            (pad.DeltaBias).scaleAdd(-(1-s),pad.dbj);

            pad.dbi += pad.DeltaBias;
            pad.dbj += pad.DeltaBias;

            if ( abs2(pad.dbi) > (pad.epsiloni)+(pad.kappa) )
            {
                sj = s;

                pad.dbi -= pad.DeltaBias;
                pad.dbj -= pad.DeltaBias;
            }

            else if ( abs2(pad.dbj) > (pad.epsilonj)+(pad.kappa) )
            {
                si = s;

                pad.dbi -= pad.DeltaBias;
                pad.dbj -= pad.DeltaBias;
            }

            else
            {
                isvalid = 1;
            }
        }
    }

    return numcerts;
}





// Return 0 if step found that doesn't lead to an optimal soln, nz otherwise

template <class T> 
int stepFind(smoVectScratch<T> &pad, int testdiscontinuities)
{
    double temp;

    pad.steptodisci = 0;
    pad.steptodiscj = 0;

    pad.normalphai = norm2(pad.alphai);
    pad.normalphaj = norm2(pad.alphaj);

    pad.absalphai  = sqrt(pad.normalphai);
    pad.absalphaj  = sqrt(pad.normalphaj);

    pad.alphaijprod       = twoProductNoConj(temp,pad.alphai,pad.alphaj);
    pad.normalphaipalphaj = pad.normalphai + pad.normalphaj + 2*(pad.alphaijprod);
    pad.absalphaipalphaj  = sqrt(pad.normalphaipalphaj);

    pad.alphaieqnegalphaj = ( pad.absalphaipalphaj < pad.ztol );

    pad.atAlphai = ( pad.absalphai < pad.ztol );
    pad.atAlphaj = ( pad.absalphaj < pad.ztol );

    // If alphai is feasible then check optimality

    pad.havegrad = 0;

//errstream() << "\nphantomxb -1\n";
    if ( pad.atAlphai || ( testdiscontinuities && ( pad.absalphaipalphaj <= pad.CNCi - pad.iota ) ) )
    {
        pad.g0 = pad.eij;
//errstream() << "phantomxb 0 " << pad.g0 << "\n";
        (pad.g0).scaleAdd(-pad.etaij,pad.alphai);
//errstream() << "phantomxb 1 " << pad.g0 << "\n";

        if ( !(pad.alphaieqnegalphaj) )
        {
            pad.ascale = ( pad.epsilonj + (1/((pad.t)*((pad.CNCj)-(pad.absalphaipalphaj)))) ) / (pad.absalphaipalphaj);

            (pad.g0).scaleAdd(-(pad.ascale),pad.alphai);
            (pad.g0).scaleAdd(-(pad.ascale),pad.alphaj);

            pad.ascale = abs2(pad.g0);
            pad.bscale = (pad.ascale) - (pad.epsiloni) - (1/((pad.t)*(pad.CNCi)));
            pad.bscale = ( (pad.bscale) > 0 ) ? (pad.bscale) : 0;

            pad.g0 *= ( ( pad.ascale == 0 ) ? 0 : (pad.bscale)/(pad.ascale) );
//errstream() << "phantomxb 2 " << pad.g0 << "\n";
        }

        else
        {
            pad.ascale = abs2(pad.g0);
            pad.bscale = (pad.ascale) - (pad.epsiloni) - (pad.epsilonj) - (1/((pad.t)*(pad.CNCi))) - (1/((pad.t)*(pad.CNCj)));
            pad.bscale = ( (pad.bscale) > 0 ) ? (pad.bscale) : 0;

            pad.g0 *= ( ( pad.ascale == 0 ) ? 0 : (pad.bscale)/(pad.ascale) );
//errstream() << "phantomxb 3 " << pad.g0 << "\n";
        }

        if ( pad.atAlphai )
        {
            pad.havegrad = 1;
//errstream() << "phantomxb 4 " << pad.g0 << "\n";
        }

        pad.absg0 = abs2(pad.g0);
//errstream() << "phantomxb 5 " << pad.g0 << "\n";

        if ( pad.absg0 <= pad.kappa )
        {
            pad.DeltaAlpha = pad.alphai;
            (pad.DeltaAlpha).negate();

            pad.steptodisci = 1;

//errstream() << "phantomxb 6 " << pad.g0 << "\n";
//errstream() << "phantomxb i\n";
            return 1;
        }
    }

    // If alphaj is feasible then check optimality

    if ( !(pad.alphaieqnegalphaj) && ( pad.atAlphaj || ( testdiscontinuities && ( pad.absalphaipalphaj <= pad.CNCj - pad.iota ) ) ) )
    {
        pad.g0 = pad.eij;
//errstream() << "phantomxb 10 " << pad.g0 << "\n";
        (pad.g0).scaleAdd(pad.etaij,pad.alphaj);
//errstream() << "phantomxb 11 " << pad.g0 << "\n";

        {
            pad.ascale = ( pad.epsiloni + (1/((pad.t)*((pad.CNCi)-(pad.absalphaipalphaj)))) ) / (pad.absalphaipalphaj);

            (pad.g0).scaleAdd((pad.ascale),pad.alphai);
            (pad.g0).scaleAdd((pad.ascale),pad.alphaj);

            pad.ascale = abs2(pad.g0);
            pad.bscale = (pad.ascale) - (pad.epsilonj) - (1/((pad.t)*(pad.CNCj)));
            pad.bscale = ( (pad.bscale) > 0 ) ? (pad.bscale) : 0;

            pad.g0 *= ( ( pad.ascale == 0 ) ? 0 : (pad.bscale)/(pad.ascale) );
        }
//errstream() << "phantomxb 12 " << pad.g0 << "\n";

        if ( pad.atAlphaj )
        {
            pad.havegrad = 1;
//errstream() << "phantomxb 12b " << pad.g0 << "\n";
        }

        pad.absg0 = abs2(pad.g0);

//errstream() << "phantomxb 13 " << pad.g0 << "\n";
        if ( pad.absg0 <= pad.kappa )
        {
            pad.DeltaAlpha = pad.alphaj;

            pad.steptodiscj = 1;

//errstream() << "phantomxb j " << pad.g0 << "\n";
            return 1;
        }
    }

    // Calculate gradient

    if ( !(pad.havegrad) )
    {
        pad.g0 = pad.eij;
//errstream() << "phantomxb 15 " << pad.g0 << "\n";

        pad.ascale = ( pad.epsiloni + (1/((pad.t)*((pad.CNCi)-(pad.absalphai)))) ) / (pad.absalphai);
        (pad.g0).scaleAdd((pad.ascale),pad.alphai);
//errstream() << "phantomxb 16 " << pad.g0 << "\n";

        pad.bscale = ( pad.epsilonj + (1/((pad.t)*((pad.CNCj)-(pad.absalphaj)))) ) / (pad.absalphaj);
        (pad.g0).scaleAdd(-(pad.bscale),pad.alphaj);
//errstream() << "phantomxb 17 " << pad.g0 << "\n";

        pad.absg0 = abs2(pad.g0);

//errstream() << "phantomxb 50 " << pad.g0 << "\n";
//errstream() << "phantomxb 51 " << pad.absg0 << "\n";
        if ( pad.absg0 <= pad.kappa )
        {
            pad.DeltaAlpha = 0.0;

            return 1;
        }
    }

    // Calculate step direction (linear if at discontinuity) and scale

    dirFind(pad);
//errstream() << "phantomxb 24 " << pad.DeltaAlpha << "\n";
    pad.sM = 2;

//errstream() << "phantomxb 25 " << pad.DeltaAlpha << "\n";
    pad.DeltaAlpha *= lineSearch(pad);
//errstream() << "phantomxb 26 " << pad.DeltaAlpha << "\n";

    return 0;
}



template <class T> 
Vector<double> &dirFind(smoVectScratch<T> &pad)
{
//errstream() << "\nphantomxc -1\n";
    if ( !(pad.atAlphai)  && !(pad.atAlphaj) )
    {
        double temp;

//errstream() << "phantomxc 0\n";
        pad.chii = (pad.CNCi) - (pad.absalphai);
//errstream() << "phantomxc 01 " << pad.chii << "\n";
        pad.chij = (pad.CNCj) - (pad.absalphaj);
//errstream() << "phantomxc 02 " << pad.chij << "\n";

        pad.zetai = ( (pad.epsiloni) + (1/((pad.t)*(pad.chii))) ) / (pad.absalphai);
//errstream() << "phantomxc 03 " << pad.zetai << "\n";
        pad.zetaj = ( (pad.epsilonj) + (1/((pad.t)*(pad.chij))) ) / (pad.absalphaj);
//errstream() << "phantomxc 04 " << pad.zetaj << "\n";

        pad.betai = ( (1/((pad.t)*(pad.chii)*(pad.chii))) - (pad.zetai) ) / (pad.normalphai);
        pad.betaj = ( (1/((pad.t)*(pad.chij)*(pad.chij))) - (pad.zetaj) ) / (pad.normalphaj);

        pad.etatilde = (pad.etaij) + (pad.zetai) + (pad.zetaj);

        pad.Mii = ((pad.etatilde)/(pad.betai)) + (pad.normalphai);
        pad.Mjj = ((pad.etatilde)/(pad.betaj)) + (pad.normalphaj);
        pad.Mij = pad.alphaijprod;

        pad.Mdet = ((pad.Mii)*(pad.Mjj)) - ((pad.Mij)*(pad.Mij));

        pad.Minvii = (pad.Mjj)/(pad.Mdet);
        pad.Minvjj = (pad.Mii)/(pad.Mdet);
        pad.Minvij = -(pad.Mij)/(pad.Mdet);

        pad.ki = twoProductNoConj(temp,pad.alphai,pad.g0);
        pad.kj = twoProductNoConj(temp,pad.alphaj,pad.g0);

        pad.zi = ((pad.Minvii)*(pad.ki)) + ((pad.Minvij)*(pad.kj));
        pad.zj = ((pad.Minvij)*(pad.ki)) + ((pad.Minvjj)*(pad.kj));

        pad.DeltaAlpha = pad.g0;
        (pad.DeltaAlpha).negate();
        (pad.DeltaAlpha).scaleAdd(pad.zi,pad.alphai);
        (pad.DeltaAlpha).scaleAdd(pad.zj,pad.alphaj);
        pad.DeltaAlpha /= pad.etatilde;
    }

    else if ( !(pad.atAlphai) )
    {
        double temp;

//errstream() << "phantomxc 1\n";
        pad.chii = (pad.CNCi) - (pad.absalphai);
        pad.zetai = ( (pad.epsiloni) + (1/((pad.t)*(pad.chii))) ) / (pad.absalphai);
        pad.betai = ( (1/((pad.t)*(pad.chii)*(pad.chii))) - (pad.zetai) ) / (pad.normalphai);
        pad.etatilde = (pad.etaij) + (pad.zetai);

        pad.DeltaAlpha = pad.g0;
        (pad.DeltaAlpha).negate();
        (pad.DeltaAlpha).scaleAdd(twoProductNoConj(temp,pad.alphai,pad.g0)/(((pad.etatilde)/(pad.betai))+(pad.normalphai)),pad.alphai);
        pad.DeltaAlpha /= pad.etatilde;
    }

    else if ( !(pad.atAlphaj) )
    {
        double temp;

//errstream() << "phantomxc 2\n";
        pad.chij = (pad.CNCj) - (pad.absalphaj);
        pad.zetaj = ( (pad.epsilonj) + (1/((pad.t)*(pad.chij))) ) / (pad.absalphaj);
        pad.betaj = ( (1/((pad.t)*(pad.chij)*(pad.chij))) - (pad.zetaj) ) / (pad.normalphaj);
        pad.etatilde = (pad.etaij) + (pad.zetaj);

        pad.DeltaAlpha = pad.g0;
        (pad.DeltaAlpha).negate();
        (pad.DeltaAlpha).scaleAdd(twoProductNoConj(temp,pad.alphaj,pad.g0)/(((pad.etatilde)/(pad.betaj))+(pad.normalphaj)),pad.alphaj);
        pad.DeltaAlpha /= pad.etatilde;
    }

    else
    {
//errstream() << "phantomxc 3\n";
        pad.DeltaAlpha = pad.g0;
        (pad.DeltaAlpha).negate();
        pad.DeltaAlpha /= pad.etaij;
    }

    return pad.DeltaAlpha;
}

template <class T> 
double lineSearch(smoVectScratch<T> &pad)
{
    double temp;

//errstream() << "\nphantomxqq 0\n";
    pad.xabs     = abs2(pad.DeltaAlpha);
    pad.thetai   = twoProductNoConj(temp,pad.alphai,pad.DeltaAlpha)/(pad.xabs);
    pad.thetaj   = twoProductNoConj(temp,pad.alphaj,pad.DeltaAlpha)/(pad.xabs);
    pad.tildeeij = twoProductNoConj(temp,pad.eij,   pad.DeltaAlpha)/(pad.xabs);
    pad.tildesM  = (pad.xabs)*(pad.sM);

    pad.CNCisq = ( pad.CNCi - pad.iota )*( pad.CNCi - pad.iota );
    pad.CNCjsq = ( pad.CNCj - pad.iota )*( pad.CNCj - pad.iota );

    pad.lambdaip = -(pad.thetai) + sqrt( ((pad.thetai)*(pad.thetai)) + pad.CNCisq - pad.normalphai );
    pad.lambdajp =  (pad.thetaj) + sqrt( ((pad.thetaj)*(pad.thetaj)) + pad.CNCjsq - pad.normalphaj );

    pad.lambda = ( pad.lambdaip < pad.lambdajp ) ? pad.lambdaip : pad.lambdajp;

    if ( pad.tildesM < pad.lambda )
    {
        pad.lambda = pad.tildesM;
    }

    pad.isidiscont = ( pad.thetai < -pad.absalphai + pad.iota );
    pad.isjdiscont = ( pad.thetaj >  pad.absalphaj - pad.iota );

    pad.tildesmin = 0;
    pad.tildesmax = pad.lambda;

    if ( pad.alphaieqnegalphaj )
    {
        pad.tti = sqrt((pad.normalphai)+(2*(pad.thetai)*(pad.absalphai))+(pad.normalphai));
        pad.ttj = sqrt((pad.normalphaj)-(2*(pad.thetaj)*(pad.absalphaj))+(pad.normalphaj));

        pad.ri = (pad.epsiloni) + 1/((pad.t)*((pad.CNCi)-(pad.tti)));
        pad.rj = (pad.epsilonj) + 1/((pad.t)*((pad.CNCj)-(pad.ttj)));

        pad.qi = pad.isidiscont ? sgn((pad.absalphai)-(pad.absalphai)) : ((pad.absalphai)+(pad.thetai))/(pad.tti);
        pad.qj = pad.isjdiscont ? sgn((pad.absalphaj)-(pad.absalphaj)) : ((pad.absalphaj)-(pad.thetaj))/(pad.ttj);

        pad.tgrad = ((pad.etaij)*(pad.absalphai)) + (pad.tildeeij) + ((pad.ri)*(pad.qi)) + ((pad.rj)*(pad.qj));
            
        if ( pad.isidiscont )
        {
            if ( ( pad.tgrad <= -((pad.ri)+(pad.rj)) ) && ( pad.absalphai > pad.tildesmin ) )
            {
//errstream() << "phantomxqq 1\n";
                pad.tildesmin = pad.absalphai;
            }

            else if ( ( pad.tgrad >= ((pad.ri)+(pad.rj)) ) && ( pad.absalphai < pad.tildesmax ) )
            {
//errstream() << "phantomxqq 2\n";
                pad.tildesmax = pad.absalphai;
            }
        }

        else
        {
            if ( ( pad.tgrad <= 0 ) && ( pad.absalphai > pad.tildesmin ) )
            {
//errstream() << "phantomxqq 3\n";
                pad.tildesmin = pad.absalphai;
            }

            else if ( ( pad.tgrad >= 0 ) && ( pad.absalphai < pad.tildesmax ) )
            {
//errstream() << "phantomxqq 4\n";
                pad.tildesmax = pad.absalphai;
            }
        }
    }

    else
    {
        pad.tti = sqrt((pad.normalphai)+(2*(pad.thetai)*(pad.absalphai))+(pad.normalphai));
        pad.ttj = sqrt((pad.normalphai)-(2*(pad.thetaj)*(pad.absalphai))+(pad.normalphaj));

        pad.ri = (pad.epsiloni) + 1/((pad.t)*((pad.CNCi)-(pad.tti)));
        pad.rj = (pad.epsilonj) + 1/((pad.t)*((pad.CNCj)-(pad.ttj)));

        pad.qi = pad.isidiscont ? sgn((pad.absalphai)-(pad.absalphai)) : ((pad.absalphai)+(pad.thetai))/(pad.tti);
        pad.qj = pad.isjdiscont ? sgn((pad.absalphai)-(pad.absalphaj)) : ((pad.absalphai)-(pad.thetaj))/(pad.ttj);

        pad.tgrad = ((pad.etaij)*(pad.absalphai)) + (pad.tildeeij) + ((pad.ri)*(pad.qi)) + ((pad.rj)*(pad.qj));
            
        if ( pad.isidiscont )
        {
            if ( ( pad.tgrad <= -(pad.ri) ) && ( pad.absalphai > pad.tildesmin ) )
            {
//errstream() << "phantomxqq 5\n";
                pad.tildesmin = pad.absalphai;
            }

            else if ( ( pad.tgrad >= (pad.ri) ) && ( pad.absalphai < pad.tildesmax ) )
            {
//errstream() << "phantomxqq 6\n";
                pad.tildesmax = pad.absalphai;
            }
        }

        else
        {
            if ( ( pad.tgrad <= 0 ) && ( pad.absalphai > pad.tildesmin ) )
            {
//errstream() << "phantomxqq 7\n";
                pad.tildesmin = pad.absalphai;
            }

            else if ( ( pad.tgrad >= 0 ) && ( pad.absalphai < pad.tildesmax ) )
            {
//errstream() << "phantomxqq 8\n";
                pad.tildesmax = pad.absalphai;
            }
        }

        pad.tti = sqrt((pad.normalphaj)+(2*(pad.thetai)*(pad.absalphaj))+(pad.normalphai));
        pad.ttj = sqrt((pad.normalphaj)-(2*(pad.thetaj)*(pad.absalphaj))+(pad.normalphaj));

        pad.ri = (pad.epsiloni) + 1/((pad.t)*((pad.CNCi)-(pad.tti)));
        pad.rj = (pad.epsilonj) + 1/((pad.t)*((pad.CNCj)-(pad.ttj)));

        pad.qi = pad.isidiscont ? sgn((pad.absalphaj)-(pad.absalphai)) : ((pad.absalphaj)+(pad.thetai))/(pad.tti);
        pad.qj = pad.isjdiscont ? sgn((pad.absalphaj)-(pad.absalphaj)) : ((pad.absalphaj)-(pad.thetaj))/(pad.ttj);

        pad.tgrad = ((pad.etaij)*(pad.absalphaj)) + (pad.tildeeij) + ((pad.ri)*(pad.qi)) + ((pad.rj)*(pad.qj));
            
        if ( pad.isjdiscont )
        {
            if ( ( pad.tgrad <= -(pad.rj) ) && ( pad.absalphaj > pad.tildesmin ) )
            {
//errstream() << "phantomxqq 9\n";
                pad.tildesmin = pad.absalphaj;
            }

            else if ( ( pad.tgrad >= (pad.rj) ) && ( pad.absalphaj < pad.tildesmax ) )
            {
//errstream() << "phantomxqq 10\n";
                pad.tildesmax = pad.absalphaj;
            }
        }

        else
        {
            if ( ( pad.tgrad <= 0 ) && ( pad.absalphaj > pad.tildesmin ) )
            {
//errstream() << "phantomxqq 11\n";
                pad.tildesmin = pad.absalphaj;
            }

            else if ( ( pad.tgrad >= 0 ) && ( pad.absalphaj < pad.tildesmax ) )
            {
//errstream() << "phantomxqq 12\n";
                pad.tildesmax = pad.absalphaj;
            }
        }
    }
//errstream() << "phantomxqq 42 " << pad.etaij << "," << pad.tildeeij << "," << pad.tildesmin << "," << pad.tildesmax << "\n";

    return findZeroCross(pad.etaij,pad.tildeeij,pad.epsiloni,pad.epsilonj,pad.thetai,pad.normalphai,pad.absalphai,pad.thetaj,pad.normalphaj,pad.absalphaj,pad.iota,pad.tildesmin,pad.tildesmax,pad.zcmaxitcnt,pad.kappa,pad.isidiscont,pad.isjdiscont,pad.CNCi,pad.CNCj,pad.t,pad.ztol)/(pad.xabs);
}








#endif
