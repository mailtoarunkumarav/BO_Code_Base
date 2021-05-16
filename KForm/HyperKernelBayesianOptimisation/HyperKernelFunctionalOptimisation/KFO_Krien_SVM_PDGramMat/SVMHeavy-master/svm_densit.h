
//
// Density estimation SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_densit_h
#define _svm_densit_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_scalar.h"






class SVM_Densit;


// Swap function

inline void qswap(SVM_Densit &a, SVM_Densit &b);

class SVM_Densit : public SVM_Scalar
{
public:

    // Constructors, destructors, assignment operators and similar

    SVM_Densit();
    SVM_Densit(const SVM_Densit &src);
    SVM_Densit(const SVM_Densit &src, const ML_Base *xsrc);
    SVM_Densit &operator=(const SVM_Densit &src) { assign(src); return *this; }
    virtual ~SVM_Densit() { return; }

    virtual int restart(void) { SVM_Densit temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual int tspaceDim(void)  const { return 1; }
    virtual int numClasses(void) const { return 0; }
    virtual int type(void)       const { return 7; }
    virtual int subtype(void)    const { return 0; }
    virtual int order(void)      const { return 0; }

    virtual char gOutType(void) const { return 'R'; }
    virtual char hOutType(void) const { return 'R'; }
    virtual char targType(void) const { return 'N'; }

    virtual int isClassifier(void) const { return 0; }

    // Modification and autoset functions

    virtual int sety(int i, double z)                               { (void) i; (void) z; throw("sety not defined for density estimation\n"); return 1; }
    virtual int sety(const Vector<int> &i, const Vector<double> &z) { (void) i; (void) z; throw("sety not defined for density estimation\n"); return 1; }
    virtual int sety(const Vector<double> &z)                       { (void) z;           throw("sety not defined for density estimation\n"); return 1; }

    // Training set control

    virtual int addTrainingVector (int i, double z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);
    virtual int qaddTrainingVector(int i, double z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);

    virtual int addTrainingVector (int i, const Vector<double> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);
    virtual int qaddTrainingVector(int i, const Vector<double> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);

    virtual int addTrainingVector (int i, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);
    virtual int qaddTrainingVector(int i,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);

    virtual int addTrainingVector (int i, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);
    virtual int qaddTrainingVector(int i,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);

    virtual int removeTrainingVector(int i) { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int sety(int i, const gentype &z) { (void) i; (void) z; throw("sety  not defined for density estimation\n"); return 0; }
    virtual int sety(const Vector<int> &i, const Vector<gentype> &z) { (void) i; (void) z; throw("sety  not defined for density estimation\n"); return 0; }
    virtual int sety(const Vector<gentype> &z) { (void) z; throw("sety  not defined for density estimation\n"); return 0; }

    virtual int setd(int i, int d);
    virtual int setd(const Vector<int> &i, const Vector<int> &d);
    virtual int setd(const Vector<int> &d);

    // Train the SVM

    virtual int train(int &res, svmvolatile int &killSwitch);
    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

    // Evaluation:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;

    // Other functions

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);

    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

private:

    virtual int gTrainingVector(double &res, int &unusedvar, int i, int raw = 0, gentype ***pxyprodi = NULL) const;

    void fixz(void);
};



inline void qswap(SVM_Densit &a, SVM_Densit &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_Densit::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Densit &b = dynamic_cast<SVM_Densit &>(bb.getML());

    SVM_Scalar::qswapinternal(b);

    return;
}

inline void SVM_Densit::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Densit &b = dynamic_cast<const SVM_Densit &>(bb.getMLconst());

    SVM_Scalar::semicopy(b);

    return;
}

inline void SVM_Densit::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Densit &src = dynamic_cast<const SVM_Densit &>(bb.getMLconst());

    SVM_Scalar::assign(src,onlySemiCopy);

    return;
}

#endif
