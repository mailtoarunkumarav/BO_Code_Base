
//
// Vector+Scoring SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_vescor_h
#define _svm_vescor_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_scscor.h"
#include "svm_vector_redbin.h"





class SVM_VeScor;


std::ostream &operator<<(std::ostream &output, const SVM_VeScor &src );
std::istream &operator>>(std::istream &input,        SVM_VeScor &dest);

// Swap function

inline void qswap(SVM_VeScor &a, SVM_VeScor &b);


class SVM_VeScor : public SVM_Vector_redbin<SVM_ScScor>
{
public:

    SVM_VeScor();
    SVM_VeScor(const SVM_VeScor &src);
    SVM_VeScor(const SVM_VeScor &src, const ML_Base *xsrc);
    SVM_VeScor &operator=(const SVM_VeScor &src) { assign(src); return *this; }
    virtual ~SVM_VeScor();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual int restart(void) { SVM_VeScor temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual int type(void)       const { return 14; }
    virtual int subtype(void)    const { return 0;  }

private:

    SVM_VeScor *thisthis;
    SVM_VeScor **thisthisthis;
};

inline void qswap(SVM_VeScor &a, SVM_VeScor &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_VeScor::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_VeScor &b = dynamic_cast<SVM_VeScor &>(bb.getML());

    SVM_Vector_redbin<SVM_ScScor>::qswapinternal(b);

    return;
}

inline void SVM_VeScor::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_VeScor &b = dynamic_cast<const SVM_VeScor &>(bb.getMLconst());

    SVM_Vector_redbin<SVM_ScScor>::semicopy(b);

    return;
}

inline void SVM_VeScor::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_VeScor &src = dynamic_cast<const SVM_VeScor &>(bb.getMLconst());

    SVM_Vector_redbin<SVM_ScScor>::assign(static_cast<const SVM_Vector_redbin<SVM_ScScor> &>(src),onlySemiCopy);

    return;
}

#endif
