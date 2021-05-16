
//
// LS-SVM planar class
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
#include "lsv_mvrank.h"

LSV_MvRank::LSV_MvRank() : LSV_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    grunt.setQuadraticCost();
    grunt.fudgeOn();

    return;
}

LSV_MvRank::LSV_MvRank(const LSV_MvRank &src) : LSV_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    grunt.setQuadraticCost();
    grunt.fudgeOn();

    assign(src,0);

    return;
}

LSV_MvRank::LSV_MvRank(const LSV_MvRank &src, const ML_Base *srcx) : LSV_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(srcx);

    grunt.setQuadraticCost();
    grunt.fudgeOn();

    assign(src,0);

    return;
}

std::ostream &LSV_MvRank::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "LSV planar regression\n";

    grunt.printstream(output,dep+1);

    return output;
}

std::istream &LSV_MvRank::inputstream(std::istream &input )
{
    grunt.inputstream(input);

    return input;
}


