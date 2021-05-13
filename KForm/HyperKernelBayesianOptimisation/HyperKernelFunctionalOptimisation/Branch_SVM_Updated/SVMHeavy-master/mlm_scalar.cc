
//
// Scalar regression Type-II multi-layer kernel-machine
//
// Version: 7
// Date: 07/07/2018
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "mlm_scalar.h"

// Boilerplate

MLM_Scalar::MLM_Scalar() : MLM_Generic()
{
    fixMLTree();

    thisthis = this;
    thisthisthis = &thisthis;

    return;
}

MLM_Scalar::MLM_Scalar(const MLM_Scalar &src) : MLM_Generic()
{
    fixMLTree();

    thisthis = this;
    thisthisthis = &thisthis;

    assign(src,0);

    return;
}

MLM_Scalar::MLM_Scalar(const MLM_Scalar &src, const ML_Base *srcx) : MLM_Generic()
{
    fixMLTree();

    thisthis = this;
    thisthisthis = &thisthis;

    assign(src,1);
    setaltx(srcx);

    return;
}

std::ostream &MLM_Scalar::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Scalar MLM\n\n";

    MLM_Generic::printstream(output,dep+1);

    return output;
}

std::istream &MLM_Scalar::inputstream(std::istream &input )
{
    MLM_Generic::inputstream(input);

    return input;
}


// Actual stuff

int MLM_Scalar::train(int &res, svmvolatile int &killSwitch)
{
    QQQ.train(res,killSwitch);

    xistrained = 1;

    return res;
}

