
//
// Pareto-frontier style 1-class Classification SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


//NB: changed so that x -> -x.  Net result is kernel 65x -> 60x

//
// - K(x,y) = k(min(x_i-y_i))
// - k(a) assumed decreasing, range +1 -> 0
// - So use kernels 402-404
// - Use 1-norm binary SVM, fixed bias 0, epsilon 0.5, with all examples in
//   class +1.
// - Then g(x) = sum_j alpha_j K(x,x_j) and alpha >= 0
// - Optimisation problem solved is:
//
//      min  sum_j alpha_j + C sum_j xi_j
//      s.t. g(x_i) + xi_i >= 0.5 for all i
//           alpha >= 0
//           xi >= 0
//
// NOW:
//
// - boundary is in negative quadrant.
// - inside is anything above/right of boundary
// - outside is anything below/left of boundary
// - g(x) = 0.5  on boundary
//          ->0  outside boundary (x -> -\infty)
//          ->NS inside  boundary (x -> +\infty)
//
// RESCALE:
//
// - we want it to be:
//   gReturned(x)  < 0 inside boundary
//   gReturned(x)  = 0 on boundary
//   gReturned(x) -> 1 outside boundary (x -> -\infty)
//
// Scale:
//
// gReturned(x) = -2.g(x) + 1
//




#ifndef _svm_pfront_h
#define _svm_pfront_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_binary.h"


class SVM_PFront;


// Swap function

inline void qswap(SVM_PFront &a, SVM_PFront &b);


class SVM_PFront : public SVM_Binary
{
public:

    // Constructors, destructors, assignment operators and similar

    SVM_PFront();
    SVM_PFront(const SVM_PFront &src);
    SVM_PFront(const SVM_PFront &src, const ML_Base *xsrc);
    SVM_PFront &operator=(const SVM_PFront &src) { assign(src); return *this; }
    virtual ~SVM_PFront();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);

    virtual int restart(void)   { SVM_PFront temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual char gOutType(void) const { return 'R'; }
    virtual char hOutType(void) const { return 'Z'; }
    virtual char targType(void) const { return 'N'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const;

    virtual int numInternalClasses(void) const { return SVM_Generic::numInternalClasses(); }

    virtual int type(void)    const { return 8; }
    virtual int subtype(void) const { return 0; }

    virtual SVM_Generic &getSVM(void)                  { return static_cast<      SVM_Generic &>(*this); }
    virtual const SVM_Generic &getSVMconst(void) const { return static_cast<const SVM_Generic &>(*this); }

    virtual int isClassifier(void) const { return 1; }

    // Training set control

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int addTrainingVector (int i, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    // Training functions:

    virtual int train(int &res) { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch);

    // Evaluation:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;

    // Other functions

    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);
};

inline void qswap(SVM_PFront &a, SVM_PFront &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_PFront::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_PFront &b = dynamic_cast<SVM_PFront &>(bb.getML());

    SVM_Binary::qswapinternal(b);

    return;
}

inline void SVM_PFront::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_PFront &b = dynamic_cast<const SVM_PFront &>(bb.getMLconst());

    SVM_Binary::semicopy(b);

    return;
}

inline void SVM_PFront::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_PFront &src = dynamic_cast<const SVM_PFront &>(bb.getMLconst());

    SVM_Binary::assign(src,onlySemiCopy);

    return;
}

#endif
