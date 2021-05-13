
//
// Sparse quadratic solver - gradient descent
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _sQgradesc_h
#define _sQgradesc_h

#include "sQbase.h"
#include "vector.h"
#include "matrix.h"
#include "optstate.h"

//
// Uses gradient descent to solve general convex optimisation problems.  Basic 
// method is projected gradient descent.  For simplicity the inner (step 
// calculating) "loop" uses sQsLsAsWs to solve (where Gn should probably be
// zero):
//
// [ I    Gpn ] [ dalpha ] + [ e ] = [ 0 ]
// [ Gpn' Gn  ] [ dbeta  ]   [ 0 ]   [ 0 ]
//
// Notes:
//
// - stepscalefactor is replaced by lr, which only affects outer loop
// - higherorderterms (fixOptState) function should be set up update optState.
// - usels:
//   bit 1: controls whether line-search is used (1 for line-search)
//   bit 2: gradient descent (0) or Netwon (1)
// - GpnRowTwoMag does not work here.
//



class fullOptStateGradDesc : public fullOptState<double,double>
{
public:

    fullOptStateGradDesc(optState<double,double> &_x,
        const Matrix<double> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<double> &_gp, const Vector<double> &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub,
        double (*_fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *_htArg = NULL,
        double _stepscalefactor = DEFAULTOUTERSTEPSCALE,
        double _lrback = DEFAULTOUTERSTEPBACK, double _delta = DEFAULTOUTERDELTA, int _usels = DEFAULTUSELS,
        int _outermaxitcnt = DEFAULTOUTERMAXITCNT, double _outermaxtraintime = DEFAULTOUTERMAXTRAINTIME, double _outertol = DEFAULTOUTERTOL, double _reltol = DEFAULTOUTERRELTOL)
        : fullOptState<double,double>(_x,_Gp,_Gpsigma,_Gn,_Gpn,_gp,_gn,_hp,_lb,_ub,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        lr     = _stepscalefactor;
        lrback = _lrback;
        delta  = _delta;
        usels  = _usels;

        outermaxitcnt     = _outermaxitcnt;
        outermaxtraintime = _outermaxtraintime;
        outertol          = _outertol;
        reltol            = _reltol;

        if ( !fixHigherOrderTerms )
        {
            fixHigherOrderTerms =  fullfixbasea;
        }

        return;
    }

    fullOptStateGradDesc(optState<double,double> &_x,
        const Matrix<double> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<double> &_gp, const Vector<double> &_gn,
        const Vector<double> &_lb, const Vector<double> &_ub,
        double (*_fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *_htArg = NULL,
        double _stepscalefactor = DEFAULTOUTERSTEPSCALE,
        double _lrback = DEFAULTOUTERSTEPBACK, double _delta = DEFAULTOUTERDELTA, int _usels = DEFAULTUSELS,
        int _outermaxitcnt = DEFAULTOUTERMAXITCNT, double _outermaxtraintime = DEFAULTOUTERMAXTRAINTIME, double _outertol = DEFAULTOUTERTOL, double _reltol = DEFAULTOUTERRELTOL)
        : fullOptState<double,double>(_x,_Gp,_Gpsigma,_Gn,_Gpn,_gp,_gn,_lb,_ub,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        lr     = _stepscalefactor;
        lrback = _lrback;
        delta  = _delta;
        usels  = _usels;

        outermaxitcnt     = _outermaxitcnt;
        outermaxtraintime = _outermaxtraintime;
        outertol          = _outertol;
        reltol            = _reltol;

        if ( !fixHigherOrderTerms )
        {
            fixHigherOrderTerms =  fullfixbasea;
        }

        return;
    }

    fullOptStateGradDesc(optState<double,double> &_x,
        const Matrix<double> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<double> &_gp, const Vector<double> &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub, const Vector<double> &_GpnRowTwoMag,
        double (*_fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *_htArg = NULL,
        double _stepscalefactor = DEFAULTOUTERSTEPSCALE,
        double _lrback = DEFAULTOUTERSTEPBACK, double _delta = DEFAULTOUTERDELTA, int _usels = DEFAULTUSELS,
        int _outermaxitcnt = DEFAULTOUTERMAXITCNT, double _outermaxtraintime = DEFAULTOUTERMAXTRAINTIME, double _outertol = DEFAULTOUTERTOL, double _reltol = DEFAULTOUTERRELTOL)
        : fullOptState<double,double>(_x,_Gp,_Gpsigma,_Gn,_Gpn,_gp,_gn,_hp,_lb,_ub,_GpnRowTwoMag,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        lr     = _stepscalefactor;
        lrback = _lrback;
        delta  = _delta;
        usels  = _usels;

        outermaxitcnt     = _outermaxitcnt;
        outermaxtraintime = _outermaxtraintime;
        outertol          = _outertol;
        reltol            = _reltol;

        if ( !fixHigherOrderTerms )
        {
            fixHigherOrderTerms =  fullfixbasea;
        }

        return;
    }

    virtual ~fullOptStateGradDesc() { return; }

    // Overwrite just the matrices/vectors, copy the rest

    virtual fullOptState<double,double> *gencopy(int _chistart,
        const Matrix<double> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<double> &_gp, const Vector<double> &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub, const Vector<double> &_GpnRowTwoMag)
    {
        fullOptStateGradDesc *res;

        MEMNEW(res,fullOptStateGradDesc(x,_Gp,_Gpsigma,_Gn,_Gpn,_gp,_gn,_hp,_lb,_ub,_GpnRowTwoMag));

        copyvars(res,_chistart);

        return static_cast<fullOptState<double,double> *>(res);
    }

    double lr;     // learning rate
    double lrback; // lr rollback factor
    double delta;  // delta factor
    int usels;     // use line-search

    int outermaxitcnt;        // max iterations (outer loop)
    double outermaxtraintime; // max train time (outer loop)
    double outertol;          // optimality tolerance (outer loop)
    double reltol;            // reltol stopping condition

//private:

    // Actual quadratic optimiser

    virtual int solve(svmvolatile int &killSwitch);

    virtual void copyvars(fullOptState<double,double> *dest, int _chistart)
    {
        fullOptState<double,double>::copyvars(dest,_chistart);

        fullOptStateGradDesc *ddest = static_cast<fullOptStateGradDesc *>(dest);

        ddest->lr     = lr;
        ddest->lrback = lrback;
        ddest->usels  = usels;

        ddest->outermaxitcnt     = outermaxitcnt;
        ddest->outermaxtraintime = outermaxtraintime;
        ddest->outertol          = outertol;
        ddest->reltol            = reltol;

        return;
    }
};





#endif
