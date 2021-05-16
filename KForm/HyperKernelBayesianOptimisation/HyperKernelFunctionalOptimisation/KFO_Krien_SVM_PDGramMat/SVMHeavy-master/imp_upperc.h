
//
// Upper Confidence Bound
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _imp_upperc_h
#define _imp_upperc_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "imp_generic.h"




class IMP_UpperC;


inline void qswap(IMP_UpperC &a, IMP_UpperC &b);


class IMP_UpperC : public IMP_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    IMP_UpperC(int isIndPrune = 0);
    IMP_UpperC(const IMP_UpperC &src, int isIndPrune = 0);
    IMP_UpperC(const IMP_UpperC &src, const ML_Base *xsrc, int isIndPrune = 0);
    IMP_UpperC &operator=(const IMP_UpperC &src) { assign(src); return *this; }
    virtual ~IMP_UpperC();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    virtual int type(void) const { return 602; }

    // Information functions (training data):

    virtual char gOutType(void) const { return 'R'; }
    virtual char hOutType(void) const { return 'R'; }

    virtual int isUnderlyingScalar(void) const { return 1; }
    virtual int isUnderlyingVector(void) const { return 0; }
    virtual int isUnderlyingAnions(void) const { return 0; }

    // Training function (pre-calculates min_i(x(i)) for x(i) enabled)

    virtual int train(int &res, svmvolatile int &killSwitch);

    // General modification and autoset functions

    virtual int reset(void)   { untrain(); return 1;                     }
    virtual int restart(void) { IMP_UpperC temp; *this = temp; return 1; }

    // Evaluation Functions:
    //
    // Output g(x) = h(x) is the input.
    // Output imp(E(x),var(x)) is upper confidence bound on increase in
    // -g(x), negated to fit with minimisation.
    //

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;
    virtual int imp(gentype &resi, const SparseVector<gentype> &xxmean, const gentype &xxvar) const;
};

inline void qswap(IMP_UpperC &a, IMP_UpperC &b)
{
    a.qswapinternal(b);

    return;
}

inline void IMP_UpperC::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    IMP_UpperC &b = dynamic_cast<IMP_UpperC &>(bb.getML());

    IMP_Generic::qswapinternal(b);

    return;
}

inline void IMP_UpperC::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const IMP_UpperC &b = dynamic_cast<const IMP_UpperC &>(bb.getMLconst());

    IMP_Generic::semicopy(b);

    return;
}

inline void IMP_UpperC::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const IMP_UpperC &src = dynamic_cast<const IMP_UpperC &>(bb.getMLconst());

    IMP_Generic::assign(src,onlySemiCopy);

    return;
}

#endif
