
//
// Binary Classification SSV
//
// Version: 7
// Date: 01/12/2017
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _ssv_binary_h
#define _ssv_binary_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ssv_scalar.h"








class SSV_Binary;


// Swap function

inline void qswap(SSV_Binary &a, SSV_Binary &b);


class SSV_Binary : public SSV_Scalar
{
public:

    SSV_Binary();
    SSV_Binary(const SSV_Binary &src);
    SSV_Binary(const SSV_Binary &src, const ML_Base *xsrc);
    SSV_Binary &operator=(const SSV_Binary &src) { assign(src); return *this; }
    virtual ~SSV_Binary();

    virtual int prealloc(int expectedN);
    virtual int preallocsize(void) const;

    virtual int restart(void) { SSV_Binary temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual int tspaceDim(void)  const { return 1;   }
    virtual int numClasses(void) const { return 2;   }
    virtual int type(void)       const { return 701; }
    virtual int subtype(void)    const { return 0;   }

    virtual char gOutType(void) const { return 'R'; }
    virtual char hOutType(void) const { return 'Z'; }
    virtual char targType(void) const { return 'Z'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual int isClassifier(void) const { return 1; }

    // Training set control

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int removeTrainingVector(int i) { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);

    virtual int sety(int i, const gentype &z);
    virtual int sety(const Vector<int> &i, const Vector<gentype> &z);
    virtual int sety(const Vector<gentype> &z);

    virtual int setd(int i, int d);
    virtual int setd(const Vector<int> &i, const Vector<int> &d);
    virtual int setd(const Vector<int> &d);

    // Evaluation:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;

    // Other functions

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);

    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

private:

    // Largely just copied from SVM_Binary

    int setdinternal(int i, int d);
};

inline void qswap(SSV_Binary &a, SSV_Binary &b)
{
    a.qswapinternal(b);

    return;
}

inline void SSV_Binary::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SSV_Binary &b = dynamic_cast<SSV_Binary &>(bb.getML());

    SSV_Scalar::qswapinternal(b);

    return;
}

inline void SSV_Binary::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SSV_Binary &b = dynamic_cast<const SSV_Binary &>(bb.getMLconst());

    SSV_Scalar::semicopy(b);

    return;
}

inline void SSV_Binary::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SSV_Binary &src = dynamic_cast<const SSV_Binary &>(bb.getMLconst());

    SSV_Scalar::assign(static_cast<const SSV_Scalar &>(src),onlySemiCopy);

    return;
}

#endif
