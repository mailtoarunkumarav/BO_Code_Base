
//
// Vector+Ranking SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_verank_h
#define _svm_verank_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_scrank.h"
#include "svm_vector_redbin.h"


// Like SVM_ScRank but for vectors.



class SVM_VeRank;


std::ostream &operator<<(std::ostream &output, const SVM_VeRank &src );
std::istream &operator>>(std::istream &input,        SVM_VeRank &dest);

// Swap function

inline void qswap(SVM_VeRank &a, SVM_VeRank &b);


class SVM_VeRank : public SVM_Vector_redbin<SVM_ScRank>
{
public:

    SVM_VeRank();
    SVM_VeRank(const SVM_VeRank &src);
    SVM_VeRank(const SVM_VeRank &src, const ML_Base *xsrc);
    SVM_VeRank &operator=(const SVM_VeRank &src) { assign(src); return *this; }
    virtual ~SVM_VeRank();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual int restart(void) { SVM_VeRank temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information:

    virtual int type(void)       const { return 11; }
    virtual int subtype(void)    const { return 0;  }

private:

    SVM_VeRank *thisthis;
    SVM_VeRank **thisthisthis;
};

inline void qswap(SVM_VeRank &a, SVM_VeRank &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_VeRank::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_VeRank &b = dynamic_cast<SVM_VeRank &>(bb.getML());

    SVM_Vector_redbin<SVM_ScRank>::qswapinternal(b);

    return;
}

inline void SVM_VeRank::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_VeRank &b = dynamic_cast<const SVM_VeRank &>(bb.getMLconst());

    SVM_Vector_redbin<SVM_ScRank>::semicopy(b);

    return;
}

inline void SVM_VeRank::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_VeRank &src = dynamic_cast<const SVM_VeRank &>(bb.getMLconst());

    SVM_Vector_redbin<SVM_ScRank>::assign(static_cast<const SVM_Vector_redbin<SVM_ScRank> &>(src),onlySemiCopy);

    return;
}

#endif
