
//
// Anionic regression GP
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "gpr_anions.h"

GPR_Anions::GPR_Anions() : GPR_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setsigma(DEFAULT_SIGMA);

    getKernel_unsafe().setType(3);
    resetKernel();

    setZeromuBias();

    setaltx(NULL);

    return;
}

GPR_Anions::GPR_Anions(const GPR_Anions &src) : GPR_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setsigma(DEFAULT_SIGMA);

    getKernel_unsafe().setType(3);
    resetKernel();

    setZeromuBias();

    setaltx(NULL);
    assign(src,0);

    return;
}

GPR_Anions::GPR_Anions(const GPR_Anions &src, const ML_Base *srcx) : GPR_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    setsigma(DEFAULT_SIGMA);

    getKernel_unsafe().setType(3);
    resetKernel();

    setZeromuBias();

    setaltx(srcx);
    assign(src,1);

    return;
}

std::ostream &GPR_Anions::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "GPR (Anions)\n";

    GPR_Generic::printstream(output,dep+1);

    return output;
}

std::istream &GPR_Anions::inputstream(std::istream &input )
{
    GPR_Generic::inputstream(input);

    return input;
}

