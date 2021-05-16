
//
// Binary Classification GP (by EP)
//
// Version: 7
// Date: 18/12/2019
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _gpr_binary_h
#define _gpr_binary_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "gpr_scalar.h"








class GPR_Binary;


// Swap function

inline void qswap(GPR_Binary &a, GPR_Binary &b);


class GPR_Binary : public GPR_Scalar
{
public:

    GPR_Binary();
    GPR_Binary(const GPR_Binary &src);
    GPR_Binary(const GPR_Binary &src, const ML_Base *xsrc);
    GPR_Binary &operator=(const GPR_Binary &src) { assign(src); return *this; }
    virtual ~GPR_Binary();

    virtual int prealloc(int expectedN);
    virtual int preallocsize(void) const;

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual int tspaceDim(void)  const { return 1;   }
    virtual int numClasses(void) const { return 2;   }
    virtual int type(void)       const { return 409; }
    virtual int subtype(void)    const { return 0;   }

    virtual char gOutType(void) const { return 'R'; }
    virtual char hOutType(void) const { return 'Z'; }
    virtual char targType(void) const { return 'Z'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual int numInternalClasses(void) const { return 2; }

    virtual const Vector<gentype> &y(void) const { return bintraintarg;  }

    virtual int isClassifier(void) const { return 1; }
    virtual int isRegression(void) const { return 0; }

    // Modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);

    virtual int sety(int                i, const gentype         &nv) { NiceAssert( nv.isCastableToIntegerWithoutLoss() ); NiceAssert( (int) nv >= -1 ); NiceAssert( (int) nv <= +1 ); return setd(i,(int) nv); }
    virtual int sety(const Vector<int> &i, const Vector<gentype> &nv);
    virtual int sety(                      const Vector<gentype> &nv);

    virtual int sety(int                i, double                nv) { NiceAssert( nv == (int) nv ); NiceAssert( (int) nv >= -1 ); NiceAssert( (int) nv <= +1 ); return setd(i,(int) nv); }
    virtual int sety(const Vector<int> &i, const Vector<double> &nv);
    virtual int sety(                      const Vector<double> &nv);

    virtual int setd(int                i, int                nd);
    virtual int setd(const Vector<int> &i, const Vector<int> &nd);
    virtual int setd(                      const Vector<int> &nd);

    virtual int restart(void) { GPR_Binary temp; *this = temp; return 1; }

    // Evaluation:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;

private:

    Vector<gentype> bintraintarg;
};

inline void qswap(GPR_Binary &a, GPR_Binary &b)
{
    a.qswapinternal(b);

    return;
}

inline void GPR_Binary::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    GPR_Binary &b = dynamic_cast<GPR_Binary &>(bb.getML());

    GPR_Scalar::qswapinternal(b);

    qswap(bintraintarg,b.bintraintarg);

    return;
}

inline void GPR_Binary::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const GPR_Binary &b = dynamic_cast<const GPR_Binary &>(bb.getMLconst());

    GPR_Scalar::semicopy(b);

    bintraintarg = b.bintraintarg;

    return;
}

inline void GPR_Binary::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const GPR_Binary &src = dynamic_cast<const GPR_Binary &>(bb.getMLconst());

    GPR_Scalar::assign(static_cast<const GPR_Scalar &>(src),onlySemiCopy);

    bintraintarg = src.bintraintarg;

    return;
}

#endif
