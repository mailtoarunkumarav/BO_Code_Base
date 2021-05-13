
//
// Function callback block
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_calbak_h
#define _blk_calbak_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.h"


// Basic user I/O


class BLK_CalBak;


// Swap and zeroing (restarting) functions

inline void qswap(BLK_CalBak &a, BLK_CalBak &b);


class BLK_CalBak : public BLK_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_CalBak(int isIndPrune = 0);
    BLK_CalBak(const BLK_CalBak &src, int isIndPrune = 0);
    BLK_CalBak(const BLK_CalBak &src, const ML_Base *xsrc, int isIndPrune = 0);
    BLK_CalBak &operator=(const BLK_CalBak &src) { assign(src); return *this; }
    virtual ~BLK_CalBak();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    virtual int type(void)    const { return 208; }
    virtual int subtype(void) const { return 0;   }

    // Evaluation Functions:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;
};

inline void qswap(BLK_CalBak &a, BLK_CalBak &b)
{
    a.qswapinternal(b);

    return;
}

inline void BLK_CalBak::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_CalBak &b = dynamic_cast<BLK_CalBak &>(bb.getML());

    BLK_Generic::qswapinternal(b);

    return;
}

inline void BLK_CalBak::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_CalBak &b = dynamic_cast<const BLK_CalBak &>(bb.getMLconst());

    BLK_Generic::semicopy(b);

    return;
}

inline void BLK_CalBak::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_CalBak &src = dynamic_cast<const BLK_CalBak &>(bb.getMLconst());

    BLK_Generic::assign(src,onlySemiCopy);

    return;
}

#endif
