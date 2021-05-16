
//
// LS-SVM gentype class
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
#include "lsv_gentyp.h"

LSV_Gentyp::LSV_Gentyp() : LSV_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    grunt.setQuadraticCost();
    grunt.fudgeOn();

    return;
}

LSV_Gentyp::LSV_Gentyp(const LSV_Gentyp &src) : LSV_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    grunt.setQuadraticCost();
    grunt.fudgeOn();

    assign(src,0);

    return;
}

LSV_Gentyp::LSV_Gentyp(const LSV_Gentyp &src, const ML_Base *srcx) : LSV_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(srcx);

    grunt.setQuadraticCost();
    grunt.fudgeOn();

    assign(src,0);

    return;
}

std::ostream &LSV_Gentyp::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "LSV gentype regression\n";

    grunt.printstream(output,dep+1);

    return output;
}

std::istream &LSV_Gentyp::inputstream(std::istream &input )
{
    grunt.inputstream(input);

    return input;
}


