
//
// 1-norm 1-class Pareto SVM measure
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
#include "imp_parsvm.h"


IMP_ParSVM::IMP_ParSVM(int isIndPrune) : IMP_Generic(isIndPrune)
{
    setaltx(NULL);

    return;
}

IMP_ParSVM::IMP_ParSVM(const IMP_ParSVM &src, int isIndPrune) : IMP_Generic(isIndPrune)
{
    setaltx(NULL);

    assign(src,0);

    return;
}

IMP_ParSVM::IMP_ParSVM(const IMP_ParSVM &src, const ML_Base *xsrc, int isIndPrune) : IMP_Generic(isIndPrune)
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

IMP_ParSVM::~IMP_ParSVM()
{
    return;
}

std::ostream &IMP_ParSVM::printstream(std::ostream &output,int dep) const
{
    repPrint(output,'>',dep) << "SVM Pareto improvement block\n";

    repPrint(output,'>',dep) << "Pareto SVM: " << content << "\n";

    return IMP_Generic::printstream(output,dep+1);
}

std::istream &IMP_ParSVM::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> content;

    return IMP_Generic::inputstream(input);
}







int IMP_ParSVM::imp(gentype &resi, const SparseVector<gentype> &xxmean, const gentype &xxxvar) const
{
    NiceAssert( isTrained() );

    gentype xxvar = xxxvar;

    gentype tempresh;
    gentype yymean;

    gh(tempresh,yymean,xxmean);

    //SparseVector<gentype> mgrad;
    //const static gentype dummy('N');
    //
    //dgg(mgrad,dummy,xxmean);

    gentype mgrad;
    SparseVector<gentype> xxmeangrad(xxmean);

    xxmeangrad.fff("&",6).force_int() = 1;
    gg(mgrad,xxmeangrad);

    xxvar *= norm2(mgrad);

    resi = yymean;
    resi.negate();

    return ( ((double) resi) == 0 ) ? 0 : ( ( -((double) resi) > 0 ) ? +1 : -1 );
}
