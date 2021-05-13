
//
// Callback Function
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
#include "blk_calbak.h"


BLK_CalBak::BLK_CalBak(int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(NULL);

    return;
}

BLK_CalBak::BLK_CalBak(const BLK_CalBak &src, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(NULL);

    assign(src,0);

    return;
}

BLK_CalBak::BLK_CalBak(const BLK_CalBak &src, const ML_Base *xsrc, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

BLK_CalBak::~BLK_CalBak()
{
    return;
}

std::ostream &BLK_CalBak::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Callback wrapper block\n";

    return BLK_Generic::printstream(output,dep+1);
}

std::istream &BLK_CalBak::inputstream(std::istream &input )
{
    return BLK_Generic::inputstream(input);
}





































int BLK_CalBak::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    (void) retaltg;
    (void) pxyprodi;

    int res = (*callback())(resg,x(i),callbackfndata());

    resh = resg;

    return res;
}



