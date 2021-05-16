
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
#include "blk_mexfna.h"


BLK_MexFnA::BLK_MexFnA(int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(NULL);

    return;
}

BLK_MexFnA::BLK_MexFnA(const BLK_MexFnA &src, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(NULL);

    assign(src,0);

    return;
}

BLK_MexFnA::BLK_MexFnA(const BLK_MexFnA &src, const ML_Base *xsrc, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

BLK_MexFnA::~BLK_MexFnA()
{
    return;
}

std::ostream &BLK_MexFnA::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "User wrapper block\n";

    return BLK_Generic::printstream(output,dep+1);
}

std::istream &BLK_MexFnA::inputstream(std::istream &input )
{
    return BLK_Generic::inputstream(input);
}





































int BLK_MexFnA::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    (void) retaltg;
    (void) pxyprodi;

    resg.force_vector(xindsize(i));

    if ( xindsize(i) )
    {
        int j;
        gentype temp;

        for ( j = 0 ; j < xindsize(i) ; j++ )
        {
            ((resg.dir_vector())("&",j)).makeString(getmexcall());

            (*(BLK_Generic::getsetExtVar))((resg.dir_vector())("&",j),xelm(temp,i,j),getmexcallid());
        }
    }

    resh = resg;

    return 0;
}

