
//
// Auto-encoding GP
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "gpr_autoen.h"

std::ostream &operator<<(std::ostream &output, const GPR_AutoEn &src)
{
    return src.printstream(output);
}

std::istream &operator>>(std::istream &input, GPR_AutoEn &dest)
{
    return dest.inputstream(input);
}

GPR_AutoEn::GPR_AutoEn() : GPR_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    getKernel_unsafe().setType(3);
    resetKernel();

    setZeromuBias();

    setaltx(NULL);

    return;
}

GPR_AutoEn::GPR_AutoEn(const GPR_AutoEn &src) : GPR_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    getKernel_unsafe().setType(3);
    resetKernel();

    setZeromuBias();

    setaltx(NULL);
    assign(src,0);

    return;
}

GPR_AutoEn::GPR_AutoEn(const GPR_AutoEn &src, const ML_Base *srcx) : GPR_Generic()
{
    thisthis = this;
    thisthisthis = &thisthis;

    getKernel_unsafe().setType(3);
    resetKernel();

    setZeromuBias();

    setaltx(srcx);
    assign(src,1);

    return;
}

std::ostream &GPR_AutoEn::printstream(std::ostream &output) const
{
    GPR_Generic::printstream(output);

    return output;
}

std::istream &GPR_AutoEn::inputstream(std::istream &input )
{
    GPR_Generic::inputstream(input);

    return input;
}

