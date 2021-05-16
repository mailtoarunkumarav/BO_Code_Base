
//
// Sparse quadratic solver - large scale, active set, warm start
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _sQsLsAsWs_h
#define _sQsLsAsWs_h

#include "sQbase.h"
#include "vector.h"
#include "matrix.h"
#include "optstate.h"


class fullOptStateActive : public fullOptState<double,double>
{
public:

    fullOptStateActive(optState<double,double> &_x,
        const Matrix<double> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<double> &_gp, const Vector<double> &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub,
        double (*_fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *_htArg = NULL,
        double _stepscalefactor = 1)
        : fullOptState<double,double>(_x,_Gp,_Gpsigma,_Gn,_Gpn,_gp,_gn,_hp,_lb,_ub,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        NiceAssert( x.keepfact() );
        NiceAssert( !_fixHigherOrderTerms );

        linbreak = 0;

        return;
    }

    fullOptStateActive(optState<double,double> &_x,
        const Matrix<double> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<double> &_gp, const Vector<double> &_gn,
        const Vector<double> &_lb, const Vector<double> &_ub,
        double (*_fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *_htArg = NULL,
        double _stepscalefactor = 1)
        : fullOptState<double,double>(_x,_Gp,_Gpsigma,_Gn,_Gpn,_gp,_gn,_lb,_ub,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        NiceAssert( x.keepfact() );
        NiceAssert( !_fixHigherOrderTerms );

        linbreak = 0;

        return;
    }

    fullOptStateActive(optState<double,double> &_x,
        const Matrix<double> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<double> &_gp, const Vector<double> &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub, const Vector<double> &_GpnRowTwoMag,
        double (*_fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *_htArg = NULL,
        double _stepscalefactor = 1)
        : fullOptState<double,double>(_x,_Gp,_Gpsigma,_Gn,_Gpn,_gp,_gn,_hp,_lb,_ub,_GpnRowTwoMag,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        NiceAssert( x.keepfact() );
        NiceAssert( !_fixHigherOrderTerms );

        linbreak = 0;

        return;
    }

    virtual ~fullOptStateActive() { return; }

    // Overwrite just the matrices/vectors, copy the rest

    virtual fullOptState<double,double> *gencopy(int _chistart,
        const Matrix<double> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<double> &_gp, const Vector<double> &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub, const Vector<double> &_GpnRowTwoMag)
    {
        fullOptStateActive *res;

        MEMNEW(res,fullOptStateActive(x,_Gp,_Gpsigma,_Gn,_Gpn,_gp,_gn,_hp,_lb,_ub,_GpnRowTwoMag));

        copyvars(res,_chistart);

        res->linbreak = linbreak;;

        return static_cast<fullOptState<double,double> *>(res);
    }

private:

    // Actual quadratic optimiser

    virtual int solve(svmvolatile int &killSwitch);

    virtual void copyvars(fullOptState<double,double> *dest, int _chistart)
    {
        fullOptState<double,double>::copyvars(dest,_chistart);

        return;
    }

public:
    int linbreak; // set 1 to stop if linear step occurs
};

#endif
