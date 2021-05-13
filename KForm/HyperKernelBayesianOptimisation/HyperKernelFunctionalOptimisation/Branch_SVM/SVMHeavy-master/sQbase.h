/*
FIXME - have stopnow function that checks a bunch of stopping criteria and returns true of solution triggers any of them

double maxitcnt; // stop if itcnt > maxitcnt - ignore (set <= 0 to disable)
double maxruntime; // stop if runtime > maxruntime (seconds) (set <= 1 to disable)
double stopval; // stop if fval <= stopval (set valninf() to disable)
double ftolrel; // stop if abs2(dfval) <= ftolrel.abs2(fval) (set 0 to disable)
double ftolabs; // stop if abs2(dfval) <= ftolabs (set 0 to disable)
double xtolrel; // stop if forall i abs2(dalpha(i)) <= xtolrel.abs2(alpha(i)) (set 0 to disable)
double xtolabs; // stop if forall i abs2(dalpha(i)) <= xtolabs (set 0 to disable)

int feedback_cycle;
int major_feedback_cycle;


// itlevel is -1 before anything starts, then gives level in (recursively nested) optimisation
// itcnt and time recording is global over all levels
// fnval and alpha are local to each level of optimisation

int itlevel;

unsigned int itcnt;

time_used start_time;
time_used curr_time;

Vector<int> firstiter; // set 1 for first iteration at this level
Vector<double> fnvalprev;
Vector<Vector<double> > alphaprev;

// Overload to return 1 for gradient descent (or more generally if fnval is evaluated)

int havefnval(void) const
{
    return 0;
}




constructor: itlevel = -1

void setstart(void)
{
    itlevel++;

    if ( firstiter.size() < itlevel+1 )
    {
        firstiter.add(firstiter.size());
        fnvalprev.add(fnvalprev.size());
        alphaprev.add(alphaprev.size());
    }

    firstiter("&",itlevel) = 1;

    if ( !itlevel )
    {
        itcnt = 0;
        start_time = TIMECALL;
    }

    errstream() << "_";

    return;
}

void setend(void)
{
    itlevel--;

    errstream() << "\b ";
    errstream() << "\b\b";

    return;
}

int testterm(double fnval, const Vector<double> &alpha)
{
    int res = 0;

    // Feedback and itcnt update

    if ( !(++itcnt%feedback_cycle) )
    {
        if ( (itcnt/feedback_cycle)%4 == 0 )
        {
            errstream() << "\b|";
        }

        else if ( (itcnt/feedback_cycle)%4 == 1 )
        {
            errstream() << "\b/";
        }

        else if ( (itcnt/feedback_cycle)%4 == 2 )
        {
            errstream() << "\b-";
        }

        else if ( (itcnt/feedback_cycle)%4 == 3 )
        {
            errstream() << "\b\\";
        }
    }

    if ( !(itcnt%major_feedback_cycle) )
    {
        std::stringstream tbuff;

        tbuff << "==" << itcnt << "==";

        int strsize = tbuff.size();

        errstream() << tbuff.str();

        int i;

        for ( i = 0 ; i < tbuff.size() ; i++ )
        {
            errstream() << "\b";
        }
    }

    // If function value not evaluated by default *and* we need it to evaluate
    // stopping criteria *then* evaluate it using default function.  Note that
    // this only happens for active, d2c or smo, so fixHigherOrderTerms is not
    // available, so use default function.

    if ( !havefnval() && ( !testisninf(stopval) || ftolrel || ftolabs ) )
    {
        fnval = intfullfixbase(*this,htArg);
    }

    // time recording

    curr_time = TIMECALL;
    double runtime = TIMEDIFFSEC(curr_time,start_time);

    // Termination tests
    //
    // Note that if a is double then !a returns true if a == 0 (C++ standard)

    if ( ( maxitcnt > 0 ) && ( itcnt > (unsigned int) maxitcnt ) )
    {
        // Iteration count check

        res = 1;
    }

    else if ( ( maxruntime > 1 ) && ( runtime > maxruntime ) )
    {
        // Run time check

        res = 1;
    }

    else if ( !testisninf(stopval) ( fnval <= stopval ) )
    {
        // Stopping value check

        res = 1;
    }

    else if ( ftolrel && ftolabs && firstiter && ( abs2(fnval-fnvalprev) <= ftolrel*abs2(fnval) ) && ( abs2(fnval-fnvalprev) <= ftolabs ) )
    {
        // f tolerance checks - rel and abs

        res = 1;
    }

    else if ( ftolrel && firstiter && ( abs2(fnval-fnvalprev) <= ftolrel*abs2(fnval) ) )
    {
        // f tolerance checks - rel only

        res = 1;
    }

    else if ( ftolabs && ( abs2(fnval-fnvalprev) <= ftolabs ) )
    {
        // f tolerance checks - abs only

        res = 1;
    }

    else if ( xtolrel && xtolabs && firstiter && ( eabs2(alpha-alphaprev) <= xtolrel*eabs2(alpha) ) && ( eabs2(alpha-alphaprev) <= xtolabs ) )
    {
        // x tolerance checks - rel and abs

        res = 1;
    }

    else if ( xtolrel && firstiter && ( eabs2(alpha-alphaprev) <= xtolrel*eabs2(alpha) ) )
    {
        // x tolerance checks - rel only

        res = 1;
    }

    else if ( xtolabs && ( eabs2(alpha-alphaprev) <= xtolabs ) )
    {
        // x tolerance checks - abs only

        res = 1;
    }

    // Previous value recording

    if ( !testisninf(stopval) || ftolrel || ftolabs )
    {
        fnvalprev = fnval;
    }

    if ( xtolrel || xtolabs )
    {
        alphaprev = alpha;
    }

    // Clear first test flag

    firstiter = 0;

    return res;
}
*/



//
// Quadratic solver base
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _sQbase_h
#define _sQbase_h

#include "basefn.h"
#include "vector.h"
#include "matrix.h"
#include "optstate.h"

//
// Pretty self explanatory: you give is a state and relevant matrices and it
// will solve:
//
// 1/2 [ alpha ]' [ Gp   Gpn ] [ alpha ] + [ alpha ]' [ gp ] + | alpha' |' [ hp ]
//     [ beta  ]  [ Gpn' Gn  ] [ beta  ]   [ beta  ]  [ gn ]   | beta   |  [ 0  ]
//
// to within precision optsol.  It is assumed that:
//
// - Gp is positive semi-definite hermitian
// - Gn is negative semi-definite hermitian
//
// return: 0  on success
//         -1 requires external presolve first
//         1  non-optimal exit
//         2  unable to attain feasibility
//
// Typically Gn and gn will be zero and beta will be the lagrange multipliers
// associated with a set of constraints, either equality (beta unconstrained),
// greater than (beta positive) or less than (beta negative).
//
// Note that if Gp is a cover matrix (say over a kernel cache) then the cache
// must be pre-extended by size(beta) elements before calling this function.
//
// Modifications and accelerations:
//
// GpnRowTwoSigned: setting this flag will make Gpn(.,1) = sgn(alpha)
// GpnRowTwoMag: scaling factors only for GpnRowTwoMag
// hpzero: this version assumes that hp is zero, which makes things faster
// fixHigherOrderTerms: if the problem isn't strictly quadratic and you want
//     Newton search then set this to the function which updates the gradient 
//     and Hessian (it gets passed x) and returns f(alpha,beta).  It is
//     assumed that Gpn and Gn and unaffected by the non-quadraticity.
//     Return value is only used for line-search gradient-descent.
// stepscalefactor: scale factor for Newton step if fixHigherOrderTerms set.
//
// =====
//
// Presolve method (wrapsolve): if the optimiser is started in a non-feasible state (that is, 
// beta is not optimal) then the basic method used is to add slack variables xi >= 0
// to the problem:
//
// 1/2 [ [ alpha ] ]' [ [ Gp   0   ]   [ Gpn ] ] [ [ alpha ] ] + [ [ alpha ] ]' [ [ gp  ] ] + | [ alpha' ] |' [ [ hp ] ]
//     [ [ xi    ] ]  [ [ 0    a.I ]   [ +-I ] ] [ [ xi    ] ]   [ [ xi    ] ]  [ [ D.1 ] ]   | [ xi     ] |  [ [ 0  ] ]
//     [           ]  [                        ] [           ]   [           ]  [         ]   |            |  [        ]
//     [ [ beta  ] ]  [ [ Gpn' +-I ]   [ Gn  ] ] [ [ beta  ] ]   [ [ beta  ] ]  [ [ gn  ] ]   | [ beta   ] |  [ [ 0  ] ]
//
// where xi are initially selected to ensure that the beta optimality
// conditions are met.  Only the minimum number of xi required to achieve
// feasibility are added, and +- is on a per-slack basis as required to ensure:
//
//     f_i + s_i xi_i ?_i 0
//     xi_i >= 0
//     s_i = +-1
//
// where: f = Gpn'.alpha + Gn.beta + gn = betaGrad
//
// and for feasibility:
//
//     ?_i is = if beta is unconstrained (betaRestrict = 0)
//            < if beta > 0 (betaRestrict = 1)
//            > if beta < 0 (betaRestrict = 2)
//
// so:
//
// - if betaRestrict == 0, beta != 0 and f_i > +opttol() then:
//
//   s_i  = -sgn(f_i) = -1
//   xi_i = |f_i| = +f_i
//
// - if betaRestrict == 0, beta != 0 and f_i < -opttol() then:
//
//   s_i  = -sgn(f_i) = +1
//   xi_i = |f_i| = -f_i
//
// - if betaRestrict == 1, beta != 0 and f_i > +opttol() then:
//
//   s_i  = -sgn(f_i) = -1
//   xi_i = |f_i| = +f_i
//
// - if betaRestrict == 2, beta != 0 and f_i < -opttol() then:
//
//   s_i  = -sgn(f_i) = +1
//   xi_i = |f_i| = -f_i
//
// Of course the restricted cases don't need to be dealt with explitictly: if
// we add an unecessary xi_i then it will go to zero anyway, so no need to add
// complexity.
//
// During optimisation if a slack xi goes to zero it will be immediately removed
// (actually restricted to 0, which is functionally equivalent).  If all xi are
// so removed during optimisation then the routine may exit (the solution is both
// feasible and optimal).  Otherwise D is increased and the optimisation repeated.
// This continues until either feasible/optimal is attained or the max repitions
// limit is hit, implying (potential) infeasibility of problem.
//
// (for vectorial alpha/beta the method is similar except that the D cost gets
// put into hp, not gp, and signs are ignored as they are ill-defined.  We use
// hp as it is on the magnitude of xi (xi starts unconstrained), which we aim to 
// push to zero).
//
// For the scalar case we also add xi type slacks to convert betaRestrict = 1,2
// to betaRestrict = 0 for simplicity of optimisation.

// Constants used by wrap solver
//
// DVALDEFAULT:   (linear) cost used to enforce unmet constraint
// CHIDIAGOFFSET: (quadratic) cost to enforce unmet constraint
//                (this is set 1 to ensure good scaling on Gp and Gpsigma)
// BETASLACKMAX:  maximum distance from feasible region
// MAXFEASREPS:   max times that costs can be increased before giving up
//                and allowing infeasibility (more than 3 is pointless
//                unless Gp is very badly scaled, which should be avoided
//                in any case).
// 

#define DVALDEFAULT 1.0e6
#define CHIDIAGOFFSET +1.0
#define BETASLACKMAX 1.0e6
#define MAXFEASREPS 3


template <class T, class S>
class fullOptState;

double fullfixbasea(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &);
double fullfixbaseb(fullOptState<Vector<double>,double> &x, void *, const Vector<double> &, const Vector<double> &, double &);




template <class T, class S>
class fullOptState
{
public:
    // Standard constructors

    fullOptState(optState<T,S> &_x,
        const Matrix<S> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<T> &_gp, const Vector<T> &_gn, Vector<double> &_hp, 
        const Vector<double> &_lb, const Vector<double> &_ub,
        double (*_fixHigherOrderTerms)(fullOptState<T,S> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *_htArg = NULL,
        double _stepscalefactor = 1)
        : x(_x), Gpfull(_Gp), Gp(Gpfull(0,1,x.aN()-1,0,1,x.aN()-1,GpDerefer)), Gpsigmafull(_Gpsigma), Gpsigma(Gpsigmafull(0,1,x.aN()-1,0,1,x.aN()-1,GpsigmaDerefer)), Gn(_Gn), Gpn(_Gpn), 
          gp(_gp), gn(_gn), hp(_hp), lb(_lb), ub(_ub), GpnRowTwoMag(GpnRowTwoMagdef)
    {
        GpnRowTwoMagdef.resize(gp.size());
        hpdef.resize(gp.size());

        GpnRowTwoMagdef = 0.0;
        hpdef           = 0.0;

        GpnRowTwoSigned = 0;
        hpzero          = 0;

        // Because of the way c++ treats specialisations it's easier to just do this in the inherit
        //fixHigherOrderTerms = _fixHigherOrderTerms ? _fixHigherOrderTerms : fullfixbase;
        fixHigherOrderTerms = _fixHigherOrderTerms;
        htArg               = _htArg;
        stepscalefactor     = _stepscalefactor;

        maxitcnt   = DEFAULT_MAXITCNT;
        maxruntime = DEFAULT_MAXTRAINTIME;

        chistart = -1;
        repover  = 0;

        return;
    }

    fullOptState(optState<T,S> &_x,
        const Matrix<S> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<T> &_gp, const Vector<T> &_gn,
        const Vector<double> &_lb, const Vector<double> &_ub,
        double (*_fixHigherOrderTerms)(fullOptState<T,S> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *_htArg = NULL,
        double _stepscalefactor = 1)
        : x(_x), Gpfull(_Gp), Gp(Gpfull(0,1,x.aN()-1,0,1,x.aN()-1,GpDerefer)), Gpsigmafull(_Gpsigma), Gpsigma(Gpsigmafull(0,1,x.aN()-1,0,1,x.aN()-1,GpsigmaDerefer)), Gn(_Gn), Gpn(_Gpn), 
          gp(_gp), gn(_gn), hp(hpdef), lb(_lb), ub(_ub), GpnRowTwoMag(GpnRowTwoMagdef)
    {
        GpnRowTwoMagdef.resize(gp.size());
        hpdef.resize(gp.size());

        GpnRowTwoMagdef = 0.0;
        hpdef           = 0.0;

        GpnRowTwoSigned = 0;
        hpzero          = 1;

        fixHigherOrderTerms = _fixHigherOrderTerms;
        htArg               = _htArg;
        stepscalefactor     = _stepscalefactor;

        maxitcnt   = DEFAULT_MAXITCNT;
        maxruntime = DEFAULT_MAXTRAINTIME;

        chistart = -1;
        repover  = 0;

        return;
    }

    fullOptState(optState<T,S> &_x,
        const Matrix<S> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<T> &_gp, const Vector<T> &_gn, Vector<double> &_hp, 
        const Vector<double> &_lb, const Vector<double> &_ub, const Vector<double> &_GpnRowTwoMag,
        double (*_fixHigherOrderTerms)(fullOptState<T,S> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *_htArg = NULL,
        double _stepscalefactor = 1)
        : x(_x), Gpfull(_Gp), Gp(Gpfull(0,1,x.aN()-1,0,1,x.aN()-1,GpDerefer)), Gpsigmafull(_Gpsigma), Gpsigma(Gpsigmafull(0,1,x.aN()-1,0,1,x.aN()-1,GpsigmaDerefer)), Gn(_Gn), Gpn(_Gpn), 
          gp(_gp), gn(_gn), hp(_hp), lb(_lb), ub(_ub), GpnRowTwoMag(_GpnRowTwoMag)
    {
        GpnRowTwoMagdef.resize(gp.size());
        hpdef.resize(gp.size());

        GpnRowTwoMagdef = 0.0;
        hpdef           = 0.0;

        GpnRowTwoSigned = 1;
        hpzero          = 0;

        fixHigherOrderTerms = _fixHigherOrderTerms;
        htArg               = _htArg;
        stepscalefactor     = _stepscalefactor;

        maxitcnt   = DEFAULT_MAXITCNT;
        maxruntime = DEFAULT_MAXTRAINTIME;

        chistart = -1;
        repover  = 0;

        return;
    }

    virtual ~fullOptState() { return; }

    // Overwrite just the matrices/vectors, copy the rest

    virtual fullOptState *gencopy(int _chistart, 
        const Matrix<S> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<T> &_gp, const Vector<T> &_gn, Vector<double> &_hp, 
        const Vector<double> &_lb, const Vector<double> &_ub, const Vector<double> &_GpnRowTwoMag)
    {
        (void) _chistart;
        (void) _Gp;
        (void) _Gpsigma;
        (void) _Gn;
        (void) _Gpn;
        (void) _gp;
        (void) _gn;
        (void) _hp;
        (void) _lb;
        (void) _ub;
        (void) _GpnRowTwoMag;

        throw("No gencopy at base");

        return NULL;
    }

    // Wrapper around solve:
    //
    // Simplifies the problem by adding slacks to ensure that the constraints
    // embodied by beta are all met before optimisation begins, then use 
    // various tricks to ensure the slacks go to zero.  Basically this ensures
    // that the solver function only ever sees a feasible problem.  You do have
    // to pre-size Gp appropriately if you want to use these.
    //
    // chistart: if Gp has been extended then the extension starts at chistart
    //           alphas(i >= chistart) are treated as "sticky" - that is, if
    //           they are constrained at zero then they stay stuck at zero 
    //           (restriction is changed).
    // repover:  if the solver called by wrapsolve sets this then wrapsolve
    //           must repeat *after* Naug has gone to zero.

    int wrapsolve(svmvolatile int &killSwitch);

//private:

    // Actual quadratic optimiser

    virtual int solve(svmvolatile int &)
    {
        throw("Solver not defined at base");

        return 2;        
    }

    // Internal variables

    optState<T,S> &x;

    retMatrix<S>      GpDerefer;
    retMatrix<double> GpsigmaDerefer;

    const Matrix<S>      &Gpfull;
    const Matrix<S>      &Gp;
    const Matrix<double> &Gpsigmafull;
    const Matrix<double> &Gpsigma;
    const Matrix<double> &Gn;
          Matrix<double> &Gpn;

    const Vector<T>      &gp;
    const Vector<T>      &gn;
          Vector<double> &hp;

    const Vector<double> &lb;
    const Vector<double> &ub;

    const Vector<double> &GpnRowTwoMag;

    int GpnRowTwoSigned;
    int hpzero;

    double (*fixHigherOrderTerms)(fullOptState<T,S> &x, void *, const Vector<double> &, const Vector<double> &, double &);
    void *htArg;
    double stepscalefactor;

    double maxitcnt;
    double maxruntime;

    int chistart;
    int repover;

    Vector<double> GpnRowTwoMagdef;
    Vector<double> hpdef;

    virtual void copyvars(fullOptState *dest, int _chistart)
    {
        dest->GpnRowTwoSigned = GpnRowTwoSigned;
        dest->hpzero          = hpzero;

        dest->fixHigherOrderTerms = fixHigherOrderTerms;
        dest->htArg               = htArg;
        dest->stepscalefactor     = stepscalefactor;

        dest->maxitcnt   = maxitcnt;
        dest->maxruntime = maxruntime;

        dest->chistart = _chistart;
        dest->repover  = 0;

        return;
    }

    // "Dummy" replacements for fixHigherOrderTerms when there are no such terms

    double intfullfixbaseb(fullOptState<Vector<double>,double> &xx, void *p, const Vector<double> &, const Vector<double> &, double &) const
    {
        (void) xx;
        (void) p;

        return 0.0;
    }

    double intfullfixbasea(fullOptState<double,double> &xx, void *p, const Vector<double> &, const Vector<double> &, double &) const
    {
        double temp;

        (void) xx;
        (void) p;

        double res = 0.0;

        const Vector<double> &alpha = x.alpha();

        const Vector<int> &aLB = x.pivAlphaLB();
        const Vector<int> &aUB = x.pivAlphaUB();
        const Vector<int> &aF  = x.pivAlphaF();

        retVector<T> tmpva;
        retVector<T> tmpvb;
        retMatrix<S> tmpma;

        res += ((double) twoProduct(temp,alpha(aLB,tmpva),Gp(aLB,aLB,tmpma)*alpha(aLB,tmpvb)))/2.0;
        res += ((double) twoProduct(temp,alpha(aLB,tmpva),Gp(aLB,aUB,tmpma)*alpha(aUB,tmpvb)))/2.0;
        res += ((double) twoProduct(temp,alpha(aLB,tmpva),Gp(aLB,aF, tmpma)*alpha(aF, tmpvb)))/2.0;
        res += ((double) twoProduct(temp,alpha(aUB,tmpva),Gp(aUB,aLB,tmpma)*alpha(aLB,tmpvb)))/2.0;
        res += ((double) twoProduct(temp,alpha(aUB,tmpva),Gp(aUB,aUB,tmpma)*alpha(aUB,tmpvb)))/2.0;
        res += ((double) twoProduct(temp,alpha(aUB,tmpva),Gp(aUB,aF, tmpma)*alpha(aF, tmpvb)))/2.0;
        res += ((double) twoProduct(temp,alpha(aF, tmpva),Gp(aF ,aLB,tmpma)*alpha(aLB,tmpvb)))/2.0;
        res += ((double) twoProduct(temp,alpha(aF, tmpva),Gp(aF ,aUB,tmpma)*alpha(aUB,tmpvb)))/2.0;
        res += ((double) twoProduct(temp,alpha(aF, tmpva),Gp(aF ,aF, tmpma)*alpha(aF, tmpvb)))/2.0;

        res += ((double) twoProduct(temp,alpha(aLB,tmpva),gp(aLB,tmpvb)));
        res += ((double) twoProduct(temp,alpha(aUB,tmpva),gp(aUB,tmpvb)));
        res += ((double) twoProduct(temp,alpha(aF ,tmpva),gp(aF ,tmpvb)));

        if ( !hpzero && aLB.size() )
        {
            int i;

            for ( i = 0 ; i < aLB.size() ; i++ )
            {
                res += abs2((double) alpha(aLB(i)))*hp(aLB(i));
            }
        }

        if ( !hpzero && aUB.size() )
        {
            int i;

            for ( i = 0 ; i < aUB.size() ; i++ )
            {
                res += abs2((double) alpha(aUB(i)))*hp(aUB(i));
            }
        }

        if ( !hpzero && aF.size() )
        {
            int i;

            for ( i = 0 ; i < aF.size() ; i++ )
            {
                res += abs2((double) alpha(aF(i)))*hp(aF(i));
            }
        }

        return res;
    }

    double intfullfixbase(fullOptState<Vector<double>,double> &xx, void *p, const Vector<double> &, const Vector<double> &, double &) const
    {
        (void) xx;
        (void) p;

        return 0.0;
    }

    double intfullfixbase(fullOptState<double,double> &xx, void *p, const Vector<double> &, const Vector<double> &, double &) const
    {
        double temp;

        (void) xx;
        (void) p;

        double res = 0.0;

        const Vector<double> &alpha = x.alpha();

        const Vector<int> &aLB = x.pivAlphaLB();
        const Vector<int> &aUB = x.pivAlphaUB();
        const Vector<int> &aF  = x.pivAlphaF();

        retVector<T> tmpva;
        retVector<T> tmpvb;
        retMatrix<S> tmpma;

        res += ((double) twoProduct(temp,alpha(aLB,tmpva),Gp(aLB,aLB,tmpma)*alpha(aLB,tmpvb)))/2.0;
        res += ((double) twoProduct(temp,alpha(aLB,tmpva),Gp(aLB,aUB,tmpma)*alpha(aUB,tmpvb)))/2.0;
        res += ((double) twoProduct(temp,alpha(aLB,tmpva),Gp(aLB,aF ,tmpma)*alpha(aF ,tmpvb)))/2.0;
        res += ((double) twoProduct(temp,alpha(aUB,tmpva),Gp(aUB,aLB,tmpma)*alpha(aLB,tmpvb)))/2.0;
        res += ((double) twoProduct(temp,alpha(aUB,tmpva),Gp(aUB,aUB,tmpma)*alpha(aUB,tmpvb)))/2.0;
        res += ((double) twoProduct(temp,alpha(aUB,tmpva),Gp(aUB,aF ,tmpma)*alpha(aF ,tmpvb)))/2.0;
        res += ((double) twoProduct(temp,alpha(aF ,tmpva),Gp(aF ,aLB,tmpma)*alpha(aLB,tmpvb)))/2.0;
        res += ((double) twoProduct(temp,alpha(aF ,tmpva),Gp(aF ,aUB,tmpma)*alpha(aUB,tmpvb)))/2.0;
        res += ((double) twoProduct(temp,alpha(aF ,tmpva),Gp(aF ,aF ,tmpma)*alpha(aF ,tmpvb)))/2.0;

        res += ((double) twoProduct(temp,alpha(aLB,tmpva),gp(aLB,tmpvb)));
        res += ((double) twoProduct(temp,alpha(aUB,tmpva),gp(aUB,tmpvb)));
        res += ((double) twoProduct(temp,alpha(aF ,tmpva),gp(aF ,tmpvb)));

        if ( !hpzero && aLB.size() )
        {
            int i;

            for ( i = 0 ; i < aLB.size() ; i++ )
            {
                res += abs2((double) alpha(aLB(i)))*hp(aLB(i));
            }
        }

        if ( !hpzero && aUB.size() )
        {
            int i;

            for ( i = 0 ; i < aUB.size() ; i++ )
            {
                res += abs2((double) alpha(aUB(i)))*hp(aUB(i));
            }
        }

        if ( !hpzero && aF.size() )
        {
            int i;

            for ( i = 0 ; i < aF.size() ; i++ )
            {
                res += abs2((double) alpha(aF(i)))*hp(aF(i));
            }
        }

        return res;
    }
};










#endif
