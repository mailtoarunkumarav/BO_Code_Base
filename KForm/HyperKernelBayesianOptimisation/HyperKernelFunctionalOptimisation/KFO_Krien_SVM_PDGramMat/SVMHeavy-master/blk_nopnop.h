
//
// Do nothing block
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_nopnop_h
#define _blk_nopnop_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.h"


// Defines a very basic set of blocks for use in machine learning.


class BLK_Nopnop;


// Swap and zeroing (restarting) functions

inline void qswap(BLK_Nopnop &a, BLK_Nopnop &b);
inline void qswap(BLK_Nopnop *&a, BLK_Nopnop *&b);



class BLK_Nopnop : public BLK_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_Nopnop(int isIndPrune = 0);
    BLK_Nopnop(const BLK_Nopnop &src, int isIndPrune = 0);
    BLK_Nopnop(const BLK_Nopnop &src, const ML_Base *xsrc, int isIndPrune = 0);
    BLK_Nopnop &operator=(const BLK_Nopnop &src) { assign(src); return *this; }
    virtual ~BLK_Nopnop();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    virtual int type(void)    const { return 200; }
    virtual int subtype(void) const { return 0;   }
};

inline void qswap(BLK_Nopnop &a, BLK_Nopnop &b)
{
    a.qswapinternal(b);

    return;
}

inline void qswap(BLK_Nopnop *&a, BLK_Nopnop *&b)
{
    BLK_Nopnop *temp;

    temp = a;
    a = b;
    b = temp;

    return;
}

inline void BLK_Nopnop::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_Nopnop &b = dynamic_cast<BLK_Nopnop &>(bb.getML());

    BLK_Generic::qswapinternal(b);

    return;
}

inline void BLK_Nopnop::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_Nopnop &b = dynamic_cast<const BLK_Nopnop &>(bb.getMLconst());

    BLK_Generic::semicopy(b);

    return;
}

inline void BLK_Nopnop::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_Nopnop &src = dynamic_cast<const BLK_Nopnop &>(bb.getMLconst());

    BLK_Generic::assign(src,onlySemiCopy);

    return;
}

#endif
