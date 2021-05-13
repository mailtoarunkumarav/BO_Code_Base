
//
// Simple system callback block.
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_system_h
#define _blk_system_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.h"


// Defines a very basic set of blocks for use in machine learning.


class BLK_System;


// Swap and zeroing (restarting) functions

inline void qswap(BLK_System &a, BLK_System &b);


class BLK_System : public BLK_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_System(int isIndPrune = 0);
    BLK_System(const BLK_System &src, int isIndPrune = 0);
    BLK_System(const BLK_System &src, const ML_Base *xsrc, int isIndPrune = 0);
    BLK_System &operator=(const BLK_System &src) { assign(src); return *this; }
    virtual ~BLK_System();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    virtual int type(void)    const { return 213; }
    virtual int subtype(void) const { return 0;   }

    // Evaluation Functions:
    //
    // Output g(x) is scalar value g(x)
    // Output h(x) is scalar value g(x)

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const;
};

inline void qswap(BLK_System &a, BLK_System &b)
{
    a.qswapinternal(b);

    return;
}

inline void BLK_System::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_System &b = dynamic_cast<BLK_System &>(bb.getML());

    BLK_Generic::qswapinternal(b);

    return;
}

inline void BLK_System::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_System &b = dynamic_cast<const BLK_System &>(bb.getMLconst());

    BLK_Generic::semicopy(b);

    return;
}

inline void BLK_System::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_System &src = dynamic_cast<const BLK_System &>(bb.getMLconst());

    BLK_Generic::assign(src,onlySemiCopy);

    return;
}

#endif
