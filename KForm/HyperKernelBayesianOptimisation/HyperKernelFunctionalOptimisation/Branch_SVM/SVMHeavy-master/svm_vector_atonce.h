
//
// Vector (at once) regression SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_vector_atonce_h
#define _svm_vector_atonce_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_vector_atonce_template.h"

double evalKSVM_Vector_atonce(int i, int j, const gentype **pxyprod, const void *owner);
double evalSigmaSVM_Vector_atonce(int i, int j, const gentype **pxyprod, const void *owner);



class SVM_Vector_atonce;


// Swap function

inline void qswap(SVM_Vector_atonce &a, SVM_Vector_atonce &b);


class SVM_Vector_atonce : public SVM_Vector_atonce_temp<double>
{
public:

    // Constructors, destructors, assignment operators and similar

    SVM_Vector_atonce();
    SVM_Vector_atonce(const SVM_Vector_atonce &src);
    SVM_Vector_atonce(const SVM_Vector_atonce &src, const ML_Base *xsrc);
    SVM_Vector_atonce &operator=(const SVM_Vector_atonce &src) { assign(src); return *this; }
    virtual ~SVM_Vector_atonce();

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Other functions

    virtual void assign(const ML_Base &src, int isOnlySemi = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);
    virtual void setN(int newN) { (void) newN; throw("setN undefined for non-generic SVM."); return; }

    virtual int subtype(void)    const { return 0;         }
};

inline void qswap(SVM_Vector_atonce &a, SVM_Vector_atonce &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_Vector_atonce::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Vector_atonce &b = dynamic_cast<SVM_Vector_atonce &>(bb.getML());

    SVM_Vector_atonce_temp<double>::qswapinternal(b);

    return;
}

inline void SVM_Vector_atonce::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Vector_atonce &b = dynamic_cast<const SVM_Vector_atonce &>(bb.getMLconst());

    SVM_Vector_atonce_temp<double>::semicopy(b);

    return;
}

inline void SVM_Vector_atonce::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Vector_atonce &src = dynamic_cast<const SVM_Vector_atonce &>(bb.getMLconst());

    SVM_Vector_atonce_temp<double>::assign(src,onlySemiCopy);

    return;
}

#endif
