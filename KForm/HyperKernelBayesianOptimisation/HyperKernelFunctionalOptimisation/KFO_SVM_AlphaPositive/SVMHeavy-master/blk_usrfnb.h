
//
// Simple function block
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_usrfnb_h
#define _blk_usrfnb_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.h"


// Defines a very basic set of blocks for use in machine learning.
//
// g(x) = fn(x)
// sampling: applies finalisation to fn


class BLK_UsrFnB;


// Swap and zeroing (restarting) functions

inline void qswap(BLK_UsrFnB &a, BLK_UsrFnB &b);


class BLK_UsrFnB : public BLK_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_UsrFnB(int isIndPrune = 0);
    BLK_UsrFnB(const BLK_UsrFnB &src, int isIndPrune = 0);
    BLK_UsrFnB(const BLK_UsrFnB &src, const ML_Base *xsrc, int isIndPrune = 0);
    BLK_UsrFnB &operator=(const BLK_UsrFnB &src) { assign(src); return *this; }
    virtual ~BLK_UsrFnB();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    virtual int N(void)       const { return 0;   }
    virtual int NNC(int)      const { return 0;   }
    virtual int type(void)    const { return 207; }
    virtual int subtype(void) const { return 0;   }

    // Evaluation Functions:
    //
    // Output g(x) is scalar value g(x)
    // Output h(x) is scalar value g(x)

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;
};

inline void qswap(BLK_UsrFnB &a, BLK_UsrFnB &b)
{
    a.qswapinternal(b);

    return;
}

inline void BLK_UsrFnB::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_UsrFnB &b = dynamic_cast<BLK_UsrFnB &>(bb.getML());

    BLK_Generic::qswapinternal(b);

    return;
}

inline void BLK_UsrFnB::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_UsrFnB &b = dynamic_cast<const BLK_UsrFnB &>(bb.getMLconst());

    BLK_Generic::semicopy(b);

    return;
}

inline void BLK_UsrFnB::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_UsrFnB &src = dynamic_cast<const BLK_UsrFnB &>(bb.getMLconst());

    BLK_Generic::assign(src,onlySemiCopy);

    return;
}

#endif
