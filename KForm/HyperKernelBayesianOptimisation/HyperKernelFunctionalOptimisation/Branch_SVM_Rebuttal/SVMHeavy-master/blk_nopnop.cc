
//
// Do nothing block
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_nopnop.h"


BLK_Nopnop::BLK_Nopnop(int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(NULL);

    return;
}

BLK_Nopnop::BLK_Nopnop(const BLK_Nopnop &src, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(NULL);

    assign(src,0);

    return;
}

BLK_Nopnop::BLK_Nopnop(const BLK_Nopnop &src, const ML_Base *xsrc, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

BLK_Nopnop::~BLK_Nopnop()
{
    return;
}

std::ostream &BLK_Nopnop::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "No-op wrapper block\n";

    return BLK_Generic::printstream(output,dep+1);
}

std::istream &BLK_Nopnop::inputstream(std::istream &input )
{
    return BLK_Generic::inputstream(input);
}
