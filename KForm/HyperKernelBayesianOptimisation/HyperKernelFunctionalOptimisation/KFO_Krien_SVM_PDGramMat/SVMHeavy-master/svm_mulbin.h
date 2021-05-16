
//
// Multi-user Binary Classification SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_mulbin_h
#define _svm_mulbin_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_mvrank.h"








class SVM_MulBin;


// Swap function

inline void qswap(SVM_MulBin &a, SVM_MulBin &b);


class SVM_MulBin : public SVM_MvRank
{
public:

    SVM_MulBin();
    SVM_MulBin(const SVM_MulBin &src);
    SVM_MulBin(const SVM_MulBin &src, const ML_Base *xsrc);
    SVM_MulBin &operator=(const SVM_MulBin &src) { assign(src); return *this; }
    virtual ~SVM_MulBin();

    virtual int prealloc(int expectedN);
    virtual int preallocsize(void) const;

    virtual int restart(void) { SVM_MulBin temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual int type(void)    const { return 18; }
    virtual int subtype(void) const { return 0;  }

    virtual const Vector<gentype> &y(void)  const { return locy;  }
    virtual const Vector<double>  &zR(void) const { return loczR; }

    // Training set control

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int addTrainingVector (int i, int d, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, double z = 0.0);
    virtual int qaddTrainingVector(int i, int d,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, double z = 0.0);

    virtual int addTrainingVector (int i, const Vector<int> &d, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<double> &z);
    virtual int qaddTrainingVector(int i, const Vector<int> &d,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<double> &z);

    virtual int removeTrainingVector(int i) { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);

    virtual int sety(int i, double z);
    virtual int sety(const Vector<int> &i, const Vector<double> &d);
    virtual int sety(const Vector<double> &z);

    virtual int sety(int i, const gentype &z);
    virtual int sety(const Vector<int> &i, const Vector<gentype> &z);
    virtual int sety(const Vector<gentype> &z);

    virtual int setd(int i, int d);
    virtual int setd(const Vector<int> &i, const Vector<int> &d);
    virtual int setd(const Vector<int> &d);

    // Other functions

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);

    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

private:

    Vector<double> loczR;
    Vector<gentype> locy;
};

inline void qswap(SVM_MulBin &a, SVM_MulBin &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_MulBin::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_MulBin &b = dynamic_cast<SVM_MulBin &>(bb.getML());

    SVM_MvRank::qswapinternal(b);

    qswap(loczR,b.loczR);
    qswap(locy, b.locy );

    return;
}

inline void SVM_MulBin::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_MulBin &b = dynamic_cast<const SVM_MulBin &>(bb.getMLconst());

    SVM_MvRank::semicopy(b);

    //y,zR

    return;
}

inline void SVM_MulBin::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_MulBin &src = dynamic_cast<const SVM_MulBin &>(bb.getMLconst());

    SVM_MvRank::assign(static_cast<const SVM_MvRank &>(src),onlySemiCopy);

    locy  = src.locy;
    loczR = src.loczR;

    return;
}

#endif
