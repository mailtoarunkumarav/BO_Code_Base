
//
// Sparse quadratic solver - special case SMO optimiser
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _sQsmo_h
#define _sQsmo_h

#include "sQbase.h"
#include "vector.h"
#include "matrix.h"
#include "optstate.h"

//
// Pretty self explanatory: you give is a state and relevant matrices and it
// will solve:
//
// [ alpha ]' [ Gp   Gpn ] [ alpha ] + [ alpha ]' [ gp ] + | alpha' |' [ hp ]
// [ beta  ]  [ Gpn' Gn  ] [ beta  ]   [ beta  ]  [ gn ]   | beta   |  [ 0  ]
//
// to within precision optsol.  It is assumed that:
//
// - Gp is positive semi-definite hermitian
// - Gpn is a size(Gp)*1 matrix of all 1s (the only effect is in sQsmo.cc, trial_step_SMO)
// - Gn is a 1*1 zero matrix
// - GpnRowTwoSigned = 0
//
// Will return 0 on success or an error code otherwise
//


class fullOptStateSMO : public fullOptState<double,double>
{
public:

    fullOptStateSMO(optState<double,double> &_x,
        const Matrix<double> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<double> &_gp, const Vector<double> &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub,
        double (*_fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *_htArg = NULL,
        double _stepscalefactor = 1)
        : fullOptState<double,double>(_x,_Gp,_Gpsigma,_Gn,_Gpn,_gp,_gn,_hp,_lb,_ub,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        NiceAssert( !_fixHigherOrderTerms );

        return;
    }

    fullOptStateSMO(optState<double,double> &_x,
        const Matrix<double> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<double> &_gp, const Vector<double> &_gn,
        const Vector<double> &_lb, const Vector<double> &_ub,
        double (*_fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *_htArg = NULL,
        double _stepscalefactor = 1)
        : fullOptState<double,double>(_x,_Gp,_Gpsigma,_Gn,_Gpn,_gp,_gn,_lb,_ub,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        NiceAssert( !_fixHigherOrderTerms );

        return;
    }

    fullOptStateSMO(optState<double,double> &_x,
        const Matrix<double> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<double> &_gp, const Vector<double> &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub, const Vector<double> &_GpnRowTwoMag,
        double (*_fixHigherOrderTerms)(fullOptState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *_htArg = NULL,
        double _stepscalefactor = 1)
        : fullOptState<double,double>(_x,_Gp,_Gpsigma,_Gn,_Gpn,_gp,_gn,_hp,_lb,_ub,_GpnRowTwoMag,_fixHigherOrderTerms,_htArg,_stepscalefactor)
    {
        NiceAssert( !_fixHigherOrderTerms );

        return;
    }

    virtual ~fullOptStateSMO() { return; }

    // Overwrite just the matrices/vectors, copy the rest

    virtual fullOptState<double,double> *gencopy(int _chistart,
        const Matrix<double> &_Gp, const Matrix<double> &_Gpsigma, const Matrix<double> &_Gn, Matrix<double> &_Gpn,
        const Vector<double> &_gp, const Vector<double> &_gn, Vector<double> &_hp,
        const Vector<double> &_lb, const Vector<double> &_ub, const Vector<double> &_GpnRowTwoMag)
    {
        fullOptStateSMO *res;

        MEMNEW(res,fullOptStateSMO(x,_Gp,_Gpsigma,_Gn,_Gpn,_gp,_gn,_hp,_lb,_ub,_GpnRowTwoMag));

        copyvars(res,_chistart);

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
};




// Don't call these.

int examineExample_SMO(int i2, int tau2, int &f_zero, optState<double,double> &probdef, const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, const Vector<double> &lb, const Vector<double> &ub, double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *htArg = NULL, double stepscalefactor = 1.0);
int takeStep_SMO(int i1, int i2, int tau1, int tau2, double &E1, double &E2, optState<double,double> &probdef, const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, const Vector<double> &lb, const Vector<double> &ub, double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *htArg = NULL, double stepscalefactor = 1.0);
int trial_step_SMO(double &d_J_epart, int i1, int i2, int tau1, int tau2, double &e1, double &e2, double &d_alpha1, double &d_alpha2, double &d_b, optState<double,double> &probdef, const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, const Vector<double> &lb, const Vector<double> &ub, double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *htArg = NULL, double stepscalefactor = 1.0);
int actually_take_step_SMO(int i1, int i2, int tau1, int tau2, double &d_alpha1, double &d_alpha2, double &d_b, optState<double,double> &probdef, const Matrix<double> &Gp, const Matrix<double> &Gpsigma, const Matrix<double> &Gn, const Matrix<double> &Gpn, const Vector<double> &gp, const Vector<double> &gn, const Vector<double> &hp, const Vector<double> &lb, const Vector<double> &ub, double (*fixHigherOrderTerms)(optState<double,double> &x, void *, const Vector<double> &, const Vector<double> &, double &) = NULL, void *htArg = NULL, double stepscalefactor = 1.0);

#endif
