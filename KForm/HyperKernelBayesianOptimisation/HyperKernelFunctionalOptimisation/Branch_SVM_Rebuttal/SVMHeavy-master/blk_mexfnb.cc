
//
// Simple function block
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
#include "blk_mexfnb.h"


BLK_MexFnB::BLK_MexFnB(int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(NULL);

    return;
}

BLK_MexFnB::BLK_MexFnB(const BLK_MexFnB &src, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(NULL);

    assign(src,0);

    return;
}

BLK_MexFnB::BLK_MexFnB(const BLK_MexFnB &src, const ML_Base *xsrc, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

BLK_MexFnB::~BLK_MexFnB()
{
    return;
}

std::ostream &BLK_MexFnB::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "User wrapper block\n";

    return BLK_Generic::printstream(output,dep+1);
}

std::istream &BLK_MexFnB::inputstream(std::istream &input )
{
    return BLK_Generic::inputstream(input);
}





































int BLK_MexFnB::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    (void) retaltg;
    (void) pxyprodi;

    int j;

    const SparseVector<gentype> &xv = x(i);

    int xdim = xv.size();

    gentype src;

    (src.force_vector()).resize(xdim);

    for ( j = 0 ; j < xdim ; j++ )
    {
        (src.dir_vector())("&",j) = xv(j);
    }

    resg.makeString(getmexcall());

    (*(BLK_Generic::getsetExtVar))(resg,src,getmexcallid());

    resh = resg;

    return 0;
}

