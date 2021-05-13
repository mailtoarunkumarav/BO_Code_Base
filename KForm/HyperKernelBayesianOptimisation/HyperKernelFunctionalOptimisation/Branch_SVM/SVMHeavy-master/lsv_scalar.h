
//
// LS-SVM scalar class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _lsv_scalar_h
#define _lsv_scalar_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "lsv_generic.h"


class LSV_Scalar;

// Swap and zeroing (restarting) functions

inline void qswap(LSV_Scalar &a, LSV_Scalar &b);
inline LSV_Scalar &setzero(LSV_Scalar &a);

class LSV_Scalar : public LSV_Generic
{
public:

    // Constructors, destructors, assignment etc..

    LSV_Scalar();
    LSV_Scalar(const LSV_Scalar &src);
    LSV_Scalar(const LSV_Scalar &src, const ML_Base *srcx);
    LSV_Scalar &operator=(const LSV_Scalar &src) { assign(src); return *this; }
    virtual ~LSV_Scalar() { return; }

    virtual int prealloc(int expectedN);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    // Information functions (training data):

    virtual int type(void)      const { return 500; }
    virtual int subtype(void)   const { return 0;   }
    virtual int tspaceDim(void) const { return 1;   }

    virtual char gOutType(void) const { return 'R'; }
    virtual char hOutType(void) const { return 'R'; }
    virtual char targType(void) const { return 'R'; }

    virtual int isUnderlyingScalar(void) const { return 1; }
    virtual int isUnderlyingVector(void) const { return 0; }
    virtual int isUnderlyingAnions(void) const { return 0; }

    virtual int getInternalClass(const gentype &y) const;

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) { return LSV_Generic::addTrainingVector (i,y,x,Cweigh,epsweigh); }
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) { return LSV_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh); }

    virtual int removeTrainingVector(int i) { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);
    virtual int removeTrainingVector(int i, int num);

    virtual int sety(int                i, const gentype         &y);
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y);
    virtual int sety(                      const Vector<gentype> &y);

    virtual int sety(int                i, const double         &y);
    virtual int sety(const Vector<int> &i, const Vector<double> &y);
    virtual int sety(                      const Vector<double> &y);

    virtual int setd(int                i, int                nd);
    virtual int setd(const Vector<int> &i, const Vector<int> &nd);
    virtual int setd(                      const Vector<int> &nd);

    // General modification and autoset functions

    virtual int scale(double a);
    virtual int reset(void);
    virtual int restart(void) { LSV_Scalar temp; *this = temp; return 1; }

    // Training functions:

    virtual int train(int &res, svmvolatile int &killSwitch);
    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

    // Use functions

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;

    virtual int covTrainingVector(gentype &resv, gentype &resmu, int i, int j, gentype ***pxyprodi = NULL, gentype ***pxyprodj = NULL, gentype **pxyprodij = NULL) const;

//    virtual void dgTrainingVector(Vector<double> &res, int i) const;

    // ================================================================
    //     Common functions for all LS-SVMs
    // ================================================================

    virtual int setgamma(const Vector<gentype> &newgamma);
    virtual int setdelta(const gentype         &newdelta);

    // ================================================================
    //     Required by K2xfer
    // ================================================================

    virtual const double         &biasR (void) const { return dbiasR;  }
    virtual const Vector<double> &alphaR(void) const { return dalphaR; }

private:

    virtual gentype &makezero(gentype &val) 
    { 
        val.force_double() = 0.0;

        return val; 
    }

    Vector<double> dalphaR;
    double dbiasR;

    Vector<double> alltraintargR;

    LSV_Scalar *thisthis;
    LSV_Scalar **thisthisthis;
};

inline void qswap(LSV_Scalar &a, LSV_Scalar &b)
{
    a.qswapinternal(b);

    return;
}

inline LSV_Scalar &setzero(LSV_Scalar &a)
{
    a.restart();

    return a;
}

inline void LSV_Scalar::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    LSV_Scalar &b = dynamic_cast<LSV_Scalar &>(bb.getML());

    LSV_Generic::qswapinternal(b);

    qswap(dalphaR      ,b.dalphaR      );
    qswap(dbiasR       ,b.dbiasR       );
    qswap(alltraintargR,b.alltraintargR);

    return;
}

inline void LSV_Scalar::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const LSV_Scalar &b = dynamic_cast<const LSV_Scalar &>(bb.getMLconst());

    LSV_Generic::semicopy(b);

    dalphaR       = b.dalphaR;
    dbiasR        = b.dbiasR;
//    alltraintargR = b.alltraintargR;

    return;
}

inline void LSV_Scalar::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const LSV_Scalar &src = dynamic_cast<const LSV_Scalar &>(bb.getMLconst());

    LSV_Generic::assign(src,onlySemiCopy);

    dalphaR       = src.dalphaR;
    dbiasR        = src.dbiasR;
    alltraintargR = src.alltraintargR;

    return;
}

#endif
