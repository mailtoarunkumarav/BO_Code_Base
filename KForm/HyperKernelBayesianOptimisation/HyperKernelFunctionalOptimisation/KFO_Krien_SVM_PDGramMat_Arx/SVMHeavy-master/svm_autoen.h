
//
// Auto Encoder SVM
//
// Version: 6
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_autoen_h
#define _svm_autoen_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_vector.h"







class SVM_AutoEn;


// Swap function

inline void qswap(SVM_AutoEn &a, SVM_AutoEn &b);


class SVM_AutoEn : public SVM_Vector
{
public:

    // Constructors, destructors, assignment operators and similar

    SVM_AutoEn();
    SVM_AutoEn(const SVM_AutoEn &src);
    SVM_AutoEn(const SVM_AutoEn &src, const ML_Base *xsrc);
    SVM_AutoEn &operator=(const SVM_AutoEn &src) { assign(src); return *this; }
    virtual ~SVM_AutoEn();

    virtual int prealloc(int expectedN);
    virtual int preallocsize(void) const;

    virtual int restart(void) { SVM_AutoEn temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual int type(void)    const { return 6; }
    virtual int subtype(void) const { return 0; }

    virtual char gOutType(void) const { return 'V'; }
    virtual char hOutType(void) const { return 'V'; }
    virtual char targType(void) const { return 'N'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual int isClassifier(void) const { return 0; }

    // Training set control

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int addTrainingVector (int i, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);
    virtual int qaddTrainingVector(int i,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);

    virtual int addTrainingVector (int i, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);
    virtual int qaddTrainingVector(int i,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);

    virtual int removeTrainingVector(int i) { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);

    // Other functions

    virtual void assign(const ML_Base &src, int isOnlySemi = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    // Overloads to make sure target space aligns with input space

    virtual int addxspaceFeat(int i)    { return SVM_Vector::addtspaceFeat(i);    }
    virtual int removexspaceFeat(int i) { return SVM_Vector::removetspaceFeat(i); }
};

inline void qswap(SVM_AutoEn &a, SVM_AutoEn &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_AutoEn::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_AutoEn &b = dynamic_cast<SVM_AutoEn &>(bb.getML());

    SVM_Vector::qswapinternal(b);

    return;
}

inline void SVM_AutoEn::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_AutoEn &b = dynamic_cast<const SVM_AutoEn &>(bb.getMLconst());

    SVM_Vector::semicopy(b);

    return;
}

inline void SVM_AutoEn::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_AutoEn &src = dynamic_cast<const SVM_AutoEn &>(bb.getMLconst());

    SVM_Vector::assign(src,onlySemiCopy);

    return;
}

#endif
