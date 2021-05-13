
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
#include "blk_usrfna.h"


BLK_UsrFnA::BLK_UsrFnA(int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(NULL);

    return;
}

BLK_UsrFnA::BLK_UsrFnA(const BLK_UsrFnA &src, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(NULL);

    assign(src,0);

    return;
}

BLK_UsrFnA::BLK_UsrFnA(const BLK_UsrFnA &src, const ML_Base *xsrc, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

BLK_UsrFnA::~BLK_UsrFnA()
{
    return;
}

std::ostream &BLK_UsrFnA::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "User wrapper block\n";

    return BLK_Generic::printstream(output,dep+1);
}

std::istream &BLK_UsrFnA::inputstream(std::istream &input )
{
    return BLK_Generic::inputstream(input);
}





































int BLK_UsrFnA::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
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
            resg.dir_vector()("&",j) = outfn()(xelm(temp,i,j));
            resg.dir_vector()("&",j);
        }
    }

    resg.finalise(2);
    resg.finalise(1);
    resg.finalise();
    resh = resg;

    return 0;
}

