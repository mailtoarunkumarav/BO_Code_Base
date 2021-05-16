
//
// Ranking SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_birank_h
#define _svm_birank_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_binary.h"



// Ranking svm: constructs
//
// g(x) = w'phi(x)
//
// subjet to constraints:
//
// g(x_i) - g(x_j) >= 1
//
// (that is g(x_i) > g(x_j)).  Training data is in two classes:
//
// Class 0:  the training vectors themselves
// Class +1: the inequalities, where x = [ i j ] refers to x_i and x_j (which
//           are assumed to be class 0).
// Class -1: like class +1 but but enforced <= constraints.
//           g(x_i) - g(x_j) <= -1
//
// It does this by replacing the kernel (for training and in test mode) by:
//
// K(x,y) = K([i0,j0],[i1,j1])
//        = K(x_i0,x_i1) + K(x_j0,x_j1) - 2K(x_i0,x_j1)
//
// and for testing:
//
// K(x,y) = K([i0,j0],y)
//        = K(x_i0,y) - K(x_j0,y)
//
// To see how this works, note that in the form of the constraints we have
// the terms:
//
// w'.(phi(x_ik) - phi(x_jk) >= 1
//
// So when we form the dual we get:
//
// w = sum_k alpha_k ( phi(x_ik) - phi(x_jk) )
//
// During training/testing we have terms w'.w, hence the first kernel sub.
// During run/use we have terms w'.phi(y), hence the second kernel sub.
//
// Note that vectors of class 0 are automatically excluded by SVM_Binary, so
// we use these to store data.  Vectors of class +1 are included but the
// calculation of kernels is referred back to this level.  gh is referred
// to this level for run/use, left at lower level for train/test.



class SVM_BiRank;


std::ostream &operator<<(std::ostream &output, const SVM_BiRank &src );
std::istream &operator>>(std::istream &input,        SVM_BiRank &dest);

// Swap function

inline void qswap(SVM_BiRank &a, SVM_BiRank &b);


class SVM_BiRank : public SVM_Binary
{
public:

    SVM_BiRank();
    SVM_BiRank(const SVM_BiRank &src);
    SVM_BiRank(const SVM_BiRank &src, const ML_Base *xsrc);
    SVM_BiRank &operator=(const SVM_BiRank &src) { assign(src); return *this; }
    virtual ~SVM_BiRank();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual int restart(void) { SVM_BiRank temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual int type(void)       const { return 9; }
    virtual int subtype(void)    const { return 0; }

    // Kernel Modification - this does all the work by changing how K is
    // evaluated.

    virtual double &KReal(double &res, int i, int j) const;

    // Evaluation:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg) const;

    virtual void  eTrainingVector(gentype         &res,                int i) const;
    virtual void dgTrainingVector(Vector<gentype> &res, gentype &resn, int i) const;
    virtual void dgTrainingVector(SparseVector<gentype> &resx, int i) const;
    virtual void drTrainingVector(Vector<gentype> &res, gentype &resn, int i) const;

    // General modification and autoset functions
    //
    // These are blocked.

    virtual int setVarBias(void)                   { return SVM_Generic::setVarBias();          }
    virtual int setPosBias(void)                   { return SVM_Generic::setPosBias();          }
    virtual int setNegBias(void)                   { return SVM_Generic::setNegBias();          }
    virtual int setFixedBias(double newbias = 0.0) { return SVM_Generic::setFixedBias(newbias); }

private:

    // This sets mode for KReal.
    //
    // locTestMode = 0: K([i0,j0],[i1,k1]) = K(x_i0,x_i1) + K(x_j0,x_j1) - 2K(x_i0,x_j1)
    // locTestMode = 1: K([i0,j0],y) = K(x_i0,y) - K(x_i0,y)
      
    int locTestMode;

    SVM_BiRank *thisthis;
    SVM_BiRank **thisthisthis;
};

inline void qswap(SVM_BiRank &a, SVM_BiRank &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_BiRank::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_BiRank &b = dynamic_cast<SVM_BiRank &>(bb.getML());

    SVM_Binary::qswapinternal(b);
    
    qswap(locTestMode,b.locTestMode);

    return;
}

inline void SVM_BiRank::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_BiRank &b = dynamic_cast<const SVM_BiRank &>(bb.getMLconst());

    SVM_Binary::semicopy(b);

    locTestMode = b.locTestMode;

    return;
}

inline void SVM_BiRank::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_BiRank &src = dynamic_cast<const SVM_BiRank &>(bb.getMLconst());

    SVM_Binary::assign(static_cast<const SVM_Binary &>(src),onlySemiCopy);

    locTestMode = src.locTestMode;

    return;
}

#endif
