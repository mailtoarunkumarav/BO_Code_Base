
//
// Gentype regression GP
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "gpr_gentyp.h"

GPR_Gentyp::GPR_Gentyp() : GPR_Generic()
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

GPR_Gentyp::GPR_Gentyp(const GPR_Gentyp &src) : GPR_Generic()
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

GPR_Gentyp::GPR_Gentyp(const GPR_Gentyp &src, const ML_Base *srcx) : GPR_Generic()
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

std::ostream &GPR_Gentyp::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "GPR (Gentype)\n";

    GPR_Generic::printstream(output,dep+1);

    return output;
}

std::istream &GPR_Gentyp::inputstream(std::istream &input )
{
    GPR_Generic::inputstream(input);

    return input;
}

